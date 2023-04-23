import torch
import torch.nn as nn
from quantize.int_linear import QuantLinear
from quantize.int_matmul import QuantMatMul
from quantize.reorder_layer_norm import ReorderLayerNorm
from models.int_opt_layer import QuantOPTDecoderLayer
from quantize.quant_transformer_layer import quant_layer
from quantize.reorder_utils import (
    tensor_calc_reorder_index,
    ic_maxmin_dict,
    oc_maxmin_dict,
    oc_maxmin_dict_debug,
    layer_i0max_hook,
    layer_omax_hook,
)


R_DEBUG_BIT = 0
DEBUG_BREAK_LAYER = -1


def R1_reorder(layer_norm, qproj, kproj, vproj, index, counts):
    layer_norm.register_buffer("reorder_index", index)
    layer_norm.out_quantizer.cluster_dim = 2
    layer_norm.out_quantizer.cluster_counts = counts
    if R_DEBUG_BIT:
        layer_norm.out_quantizer.change_n_bits(R_DEBUG_BIT)

    qproj.weight.data = torch.index_select(qproj.weight.data, 1, index)
    qproj.set_ic_cluster_counts(counts, a_dim=None)

    kproj.weight.data = torch.index_select(kproj.weight.data, 1, index)
    kproj.set_ic_cluster_counts(counts, a_dim=None)
    vproj.weight.data = torch.index_select(vproj.weight.data, 1, index)
    vproj.set_ic_cluster_counts(counts, a_dim=None)


def R2_reorder(qproj, kproj, qkt_matmul, index, counts):
    qproj.weight.data = torch.index_select(qproj.weight.data, 0, index)
    qproj.bias.data = torch.index_select(qproj.bias.data, 0, index)
    kproj.weight.data = torch.index_select(kproj.weight.data, 0, index)
    kproj.bias.data = torch.index_select(kproj.bias.data, 0, index)

    qkt_matmul.set_ic_cluster_counts(counts, x1_dim=2, x2_dim=2)
    if R_DEBUG_BIT:
        qkt_matmul.x1_quantizer.change_n_bits(R_DEBUG_BIT)
        qkt_matmul.x2_quantizer.change_n_bits(R_DEBUG_BIT)


def R3_reorder(vproj, pv_matmul, out_proj, index, counts):
    vproj.weight.data = torch.index_select(vproj.weight.data, 0, index)
    vproj.bias.data = torch.index_select(vproj.bias.data, 0, index)
    pv_matmul.set_ic_cluster_counts(counts, cluster_x1=False)
    out_proj.weight.data = torch.index_select(out_proj.weight.data, 1, index)
    out_proj.set_ic_cluster_counts(counts)
    if R_DEBUG_BIT:
        pv_matmul.x2_quantizer.change_n_bits(R_DEBUG_BIT)
        out_proj.act_quantizer.change_n_bits(R_DEBUG_BIT)


def R4_reorder(layer_norm, fc1, index, counts):
    layer_norm.register_buffer("reorder_index", index)

    layer_norm.out_quantizer.cluster_dim = 1
    layer_norm.out_quantizer.cluster_counts = counts

    fc1.weight.data = torch.index_select(fc1.weight.data, 1, index)
    fc1.set_ic_cluster_counts(counts, a_dim=None)
    if R_DEBUG_BIT:
        layer_norm.out_quantizer.change_n_bits(R_DEBUG_BIT)


def R5_reorder(fc1, fc2, index, counts):
    fc1.weight.data = torch.index_select(fc1.weight.data, 0, index)
    fc1.bias.data = torch.index_select(fc1.bias.data, 0, index)

    fc2.weight.data = torch.index_select(fc2.weight.data, 1, index)
    fc2.set_ic_cluster_counts(counts, a_dim=1)
    if R_DEBUG_BIT:
        fc2.act_quantizer.change_n_bits(R_DEBUG_BIT)


@torch.no_grad()
def opt_reorder_quantize(
    lm,
    args,
    dataloader,
    n_clusters={"R1": 4, "R2": 4, "R3": 4, "R4": 32, "R5": 4},
    reorder="12345",
):
    print("Starting ...")

    model = lm.model
    dev = lm.device

    use_cache = model.config.use_cache
    model.config.use_cache = False
    layers = model.model.decoder.layers

    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.to(dev)
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.to(dev)
    if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.to(dev)
    if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.to(dev)
    layers[0] = layers[0].to(dev)

    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros(
        (args.nsamples, lm.seqlen, model.config.hidden_size), dtype=dtype, device=dev
    )
    cache = {"i": 0, "attention_mask": None}

    # only catch the first layer input
    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            inps[cache["i"]] = inp
            cache["i"] += 1
            cache["attention_mask"] = kwargs["attention_mask"]
            raise ValueError

    layers[0] = Catcher(layers[0])

    for batch in dataloader:
        if cache["i"] >= args.nsamples:
            break
        try:
            model(batch[0].to(dev))

        except ValueError:
            pass

    layers[0] = layers[0].module
    layers[0] = layers[0].cpu()
    model.model.decoder.embed_tokens = model.model.decoder.embed_tokens.cpu()
    model.model.decoder.embed_positions = model.model.decoder.embed_positions.cpu()
    if hasattr(model.model.decoder, "project_out") and model.model.decoder.project_out:
        model.model.decoder.project_out = model.model.decoder.project_out.cpu()
    if hasattr(model.model.decoder, "project_in") and model.model.decoder.project_in:
        model.model.decoder.project_in = model.model.decoder.project_in.cpu()
    torch.cuda.empty_cache()

    outs = torch.zeros_like(inps)
    attention_mask = cache["attention_mask"]

    enable_R1 = True if "1" in reorder else False
    enable_R2 = True if "2" in reorder else False
    enable_R3 = True if "3" in reorder else False
    enable_R4 = True if "4" in reorder else False
    enable_R5 = True if "5" in reorder else False
    print(f"Ready for reorder {reorder}.")

    for i in range(len(layers)):
        if i == DEBUG_BREAK_LAYER:
            break
        print(f"=== Start quantize layer {i} ===")
        layer = layers[i].to(dev)
        qlayer = QuantOPTDecoderLayer(lm.model.config, layer, args)

        # register hook for data
        handlers = []
        for name, module in layer.named_modules():
            if (
                enable_R1
                and isinstance(module, nn.LayerNorm)
                and "attn_layer_norm" in name
            ):
                # print(f"register R1 hook for layer_norm {name}")
                module.name = name
                handler = module.register_forward_hook(layer_omax_hook)
                handlers.append(handler)
            if (
                enable_R2
                and isinstance(module, nn.Linear)
                and ("q_proj" in name or "k_proj" in name)
            ):
                # print(f"register R2 hook for layer_norm {name}")
                module.name = name
                handler = module.register_forward_hook(layer_omax_hook)
                handlers.append(handler)
            if enable_R3 and isinstance(module, nn.Linear) and "out_proj" in name:
                # print(f"register R3 hook for layer_norm {name}")
                module.name = name
                handler = module.register_forward_hook(layer_i0max_hook)
                handlers.append(handler)
            if (
                enable_R4
                and isinstance(module, nn.LayerNorm)
                and "final_layer_norm" in name
            ):
                # print(f"register R4 hook for layer_norm {name}")
                module.name = name
                handler = module.register_forward_hook(layer_omax_hook)
                handlers.append(handler)
            if enable_R5 and isinstance(module, nn.Linear) and "fc2" in name:
                # print(f"register R5 hook for layer_norm {name}")
                module.name = name
                handler = module.register_forward_hook(layer_i0max_hook)
                handlers.append(handler)

        # inference to collect data for reordering
        for j in range(args.nsamples):
            outs[j] = layer(
                inps[j].unsqueeze(0).to(dev), attention_mask=attention_mask.to(dev)
            )[0]
        for handler in handlers:
            handler.remove()

        if enable_R1:
            feature_max, feature_min = oc_maxmin_dict[f"self_attn_layer_norm"]

            R1_index, counts = tensor_calc_reorder_index(
                feature_max, feature_min, n_clusters["R1"]
            )
            # print("R1 index counts", counts)
            R1_reorder(
                qlayer.self_attn_layer_norm,
                qlayer.self_attn.q_proj,
                qlayer.self_attn.k_proj,
                qlayer.self_attn.v_proj,
                R1_index,
                counts,
            )

        if enable_R2:
            qmax, qmin = oc_maxmin_dict[f"self_attn.q_proj"]
            kmax, kmin = oc_maxmin_dict[f"self_attn.k_proj"]
            R2_index, counts = tensor_calc_reorder_index(
                [qmax, kmax], [qmin, kmin], n_clusters["R2"], qlayer.self_attn.num_heads
            )
            # print("R2 index counts", counts)
            R2_reorder(
                qlayer.self_attn.q_proj,
                qlayer.self_attn.k_proj,
                qlayer.self_attn.qkt_matmul,
                R2_index,
                counts,
            )

        if enable_R3:
            feature_max, feature_min = ic_maxmin_dict[f"self_attn.out_proj"]
            R3_index, counts = tensor_calc_reorder_index(
                feature_max, feature_min, n_clusters["R3"], qlayer.self_attn.num_heads
            )
            # print("R3 index counts", counts)
            R3_reorder(
                qlayer.self_attn.v_proj,
                qlayer.self_attn.pv_matmul,
                qlayer.self_attn.out_proj,
                R3_index,
                counts,
            )

        if enable_R4:
            feature_max, feature_min = oc_maxmin_dict[f"final_layer_norm"]

            R4_index, counts = tensor_calc_reorder_index(
                feature_max, feature_min, n_clusters["R4"]
            )
            # print("R4 index counts", counts)
            R4_reorder(
                qlayer.final_layer_norm,
                qlayer.fc1,
                R4_index,
                counts,
            )

        if enable_R5:
            feature_max, feature_min = ic_maxmin_dict[f"fc2"]
            R5_index, counts = tensor_calc_reorder_index(
                feature_max, feature_min, n_clusters["R5"]
            )
            # print("R5 index counts", counts)
            R5_reorder(
                qlayer.fc1,
                qlayer.fc2,
                R5_index,
                counts,
            )

        outs = quant_layer(qlayer, args, outs, inps, attention_mask, dev)

        ic_maxmin_dict.clear()
        oc_maxmin_dict.clear()
        layers[i] = qlayer.to("cpu")
        del layer
        torch.cuda.empty_cache()

        inps, outs = outs, inps
        print(
            lm._device,
            "memory_allocated",
            i,
            torch.cuda.memory_allocated(lm._device) / 1024 / 1024,
            "max memory_allocated",
            torch.cuda.max_memory_allocated(lm._device) / 1024**2,
        )

    del inps, outs
    model.config.use_cache = use_cache
    return model


if __name__ == "__main__":
    tensor = torch.rand([30])
    index, counts = tensor_calc_reorder_index(tensor, 2, 3)
    print(index, counts)
