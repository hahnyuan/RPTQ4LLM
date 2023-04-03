import torch
from quantize.int_linear import QuantLinear
from quantize.int_matmul import QuantMatMul
from quantize.reorder_layer_norm import ReorderLayerNorm
from quantize.gptq import GPTQ
from quantize.mem_packer import MemoryPacker
from torch.nn import Parameter


def gptq_add_batch(m, i, o):
    if m.recorded_quant_input is not None:
        m.gptq.add_batch(m.recorded_quant_input, o.data)
    else:
        m.gptq.add_batch(i[0].data, o.data)


@torch.no_grad()
def quant_layer(qlayer, args, outs, inps, attention_mask, dev):
    handlers = []
    gptqs = {}
    qlayer.set_quant_state(weight_quant=True, act_quant=True)
    for name, module in qlayer.named_modules():
        if isinstance(module, QuantLinear):
            if args.disable_w_quant or module.weight_quantizer.n_bits>=16:
                continue
            if args.w_quantizer == "normal":
                module.weight_quantizer.set_calibration_mode()
                """caculate the step size and zero point for weight quantizer"""
                fake_quantized_weight = module.weight_quantizer(module.weight)
                module.weight_quantizer.set_eval_mode()
                del module.weight
                module.register_buffer('weight',fake_quantized_weight)
                # module.weight = fake_quantized_weight
                module.replace_weight_with_quantized = True
            elif args.w_quantizer == "gptq":
                gptqs[name] = GPTQ(module, module.weight_quantizer)
                module.gptq = gptqs[name]
                module.record_quant_input = True
                handler = module.register_forward_hook(gptq_add_batch)
                handlers.append(handler)
    qlayer.set_quant_state(weight_quant=False, act_quant=True)
    # for activation quantize
    a_quantizers = {}
    w_quantizers = {}
    for name, m in qlayer.named_modules():
        if isinstance(m, (QuantLinear)) and not m.disable_input_quant:
            a_quantizers[name] = m.act_quantizer
            w_quantizers[name] = m.weight_quantizer
        if isinstance(m, ReorderLayerNorm) and m.out_quantizer is not None:
            a_quantizers[name] = m.out_quantizer
        if isinstance(m, QuantMatMul):
            a_quantizers[name + "x1"] = m.x1_quantizer
            a_quantizers[name + "x2"] = m.x2_quantizer

    for name, quantizer in a_quantizers.items():
        quantizer.set_calibration_mode()
    for j in range(args.nsamples):
        outs[j] = qlayer(
            inps[j].unsqueeze(0).to(dev), attention_mask=attention_mask.to(dev)
        )[0]

    for name, quantizer in a_quantizers.items():
        quantizer.set_eval_mode()
        if args.metric == "layer_mse":
            quantizer.layer_mse_param_search_update(
                qlayer, a_quantizers, inps, outs, attention_mask
            )
        quantizer.free()

    if args.w_quantizer == "gptq" and not args.disable_w_quant:
        # for weight quantize
        for handler in handlers:
            handler.remove()
        gptq_losses = {}
        print(f"GPTQ Quantizing ...")
        for name, module in qlayer.named_modules():
            if isinstance(module, QuantLinear) and module.weight_quantizer.n_bits < 16:
                module.record_quant_input = False
                module.recorded_quant_input = None

                fake_quantized_weight, gptq_loss = gptqs[name].fasterquant(
                    percdamp=args.percdamp
                )
                
                gptq_losses[name] = gptq_loss
                if args.pack_weight:
                    module.mem_packer = MemoryPacker(
                        module.weight_quantizer.scale,
                        module.weight_quantizer.round_zero_point,
                        module.weight_quantizer.n_bits,
                    )
                    w_packed = module.mem_packer.pack_tensor(fake_quantized_weight)
                    del module.weight
                    module.register_buffer(w_packed,w_packed)
                    module.is_weight_packed = True
                else:
                    # module.weight.data = fake_quantized_weight.to(module.weight.dtype)
                    del module.weight
                    module.register_buffer("weight", fake_quantized_weight.half())
                    module.replace_weight_with_quantized = True
                gptqs[name].free()
                module.gptq = None
                module.weight_quantizer = None

        if len(gptq_losses):
            print("GPTQ losses", gptq_losses)

            for j in range(args.nsamples):
                outs[j] = qlayer(
                    inps[j].unsqueeze(0).to(dev), attention_mask=attention_mask.to(dev)
                )[0]
            gptqs.clear()
    for name, quantizer in w_quantizers.items():
        quantizer.free()

    return outs
