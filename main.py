import os
import sys

import random
import numpy as np
from models.opt import OPTClass
import torch
import time
from datautils import get_loaders
from lm_evaluation.lm_eval import tasks, evaluator
from quantize.opt_reorder_quantize import opt_reorder_quantize
import datetime
from models.int_opt_layer import QuantOPTAttention
from pprint import pprint
from parallel_utils import map_layers_to_multi_gpus, get_lowest_occupied_gpu
import torch.nn as nn
from quantize.opt_reorder_quantize import opt_reorder_quantize
from tqdm import tqdm

torch.backends.cudnn.benchmark = True

net_choices = [
    "opt-125m",
    "opt-1.3b",
    "opt-6.7b",
    "opt-13b",
    "opt-30b",
    "opt-66b",
    # "llama-7b",
    # "llama-13b",
    # "bloom-3b",
]

# tasks lambada_openai,piqa,arc_easy,arc_challenge,openbookqa,boolq


@torch.no_grad()
def evaluate(lm, args):
    for name, m in lm.model.named_modules():
        if isinstance(m, (QuantOPTAttention,)):
            m.name = name
            # m.register_forward_hook(mem_test_hook)
    results = {}
    if args.multigpu:
        if "opt" in args.model:
            map_layers_to_multi_gpus(lm.model.model.decoder.layers)
            input_device = lm.model.model.decoder.layers[0].device
            output_device = lm.model.model.decoder.layers[-1].device
            lm._device = input_device
            assert input_device == output_device
            lm.model.model.decoder.embed_positions.to(input_device)

            lm.model.model.decoder.embed_tokens.to(input_device)
            lm.model.model.decoder.final_layer_norm.to(output_device)
            lm.model.lm_head.to(output_device)

        elif "llama" in args.model:
            map_layers_to_multi_gpus(lm.model.model.layers)
            input_device = lm.model.model.layers[0].device
            output_device = lm.model.model.layers[-1].device
            assert input_device == output_device
            lm._device = input_device
    else:
        if "opt" in args.model:
            lm.model.model.decoder = lm.model.model.decoder.to(lm.device)
        elif "llama" in args.model:
            lm.model.model = lm.model.model.to(lm.device)

    if args.eval_ppl:
        for dataset in ["wikitext2", "ptb", "c4"]:
            # for dataset in ['c4']:
            if "opt" in args.model:
                cache_testloader = f"/tmp/{dataset}_testloader_opt_all.cache"
                if os.path.exists(cache_testloader):
                    testloader = torch.load(cache_testloader)
                    # print(f"load calibration from {cache_testloader}")
                else:
                    dataloader, testloader = get_loaders(
                        dataset,
                        seed=args.seed,
                        model=args.model,
                        seqlen=lm.seqlen,
                        cache_dir=args.cache_dir,
                    )
                    torch.save(testloader, cache_testloader)
            elif "llama" in args.model:
                cache_testloader = f"/tmp/{dataset}_testloader_llama_all.cache"
                if os.path.exists(cache_testloader):
                    testloader = torch.load(cache_testloader)
                    # print(f"load calibration from {cache_testloader}")
                else:
                    dataloader, testloader = get_loaders(
                        dataset,
                        seed=args.seed,
                        model=args.model,
                        seqlen=lm.seqlen,
                        cache_dir=args.cache_dir,
                    )
                    torch.save(testloader, cache_testloader)
            # print(dataset)
            if "c4" == dataset:
                testenc = testloader
            else:
                testenc = testloader.input_ids

            nsamples = testenc.numel() // lm.seqlen
            use_cache = lm.model.config.use_cache
            lm.model.config.use_cache = False
            lm.model.eval()
            nlls = []

            for i in tqdm(range(nsamples)):
                batch = testenc[:, (i * lm.seqlen) : ((i + 1) * lm.seqlen)].to(
                    lm.device
                )
                if "opt" in args.model:
                    outputs = lm.model.model.decoder(batch)
                elif "llama" in args.model:
                    outputs = lm.model.model(batch)
                hidden_states = outputs[0]
                logits = lm.model.lm_head(hidden_states)
                shift_logits = logits[:, :-1, :]
                shift_labels = testenc[:, (i * lm.seqlen) : ((i + 1) * lm.seqlen)][
                    :, 1:
                ].to(lm.model.lm_head.weight.device)
                loss_fct = nn.CrossEntropyLoss()
                loss = loss_fct(
                    shift_logits.view(-1, shift_logits.size(-1)),
                    shift_labels.view(-1),
                )
                neg_log_likelihood = loss.float() * lm.seqlen
                nlls.append(neg_log_likelihood)
                if i == args.limit:
                    break

            ppl = torch.exp(torch.stack(nlls).sum() / (nsamples * lm.seqlen))
            print(dataset, ppl.item())
            lm.model.config.use_cache = use_cache
            # pprint(args.model)
            results[dataset] = ppl.item()
    if args.tasks != "":
        t_results = evaluator.simple_evaluate(
            lm,
            tasks=args.tasks,
            num_fewshot=args.num_fewshot,
            limit=None if args.limit == -1 else args.limit,
        )
        results.update(t_results)
        pprint(results)
    return results


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("net", type=str, choices=net_choices)
    parser.add_argument(
        "--cache_dir", default="./data", type=str, help="OPT model cache_dir"
    )
    parser.add_argument(
        "--calib_dataset",
        type=str,
        default="mix",
        choices=["wikitext2", "ptb", "c4", "mix"],
        help="Where to extract calibration data from.",
    )
    parser.add_argument(
        "--nsamples", type=int, default=128, help="Number of calibration data samples."
    )
    parser.add_argument(
        "--percdamp",
        type=float,
        default=0.01,
        help="Percent of the average Hessian diagonal to use for dampening.",
    )
    parser.add_argument(
        "--seed", type=int, default=2, help="Seed for sampling the calibration data."
    )
    parser.add_argument(
        "--metric",
        type=str,
        default="ema_minmax",
        choices=["minmax", "ema_minmax", "mse", "layer_mse"],
    )

    parser.add_argument("--tasks", default="")
    parser.add_argument("--eval_ppl", action="store_true")
    parser.add_argument("--num_fewshot", type=int, default=0)
    parser.add_argument("--output_path", default="./output")
    parser.add_argument("--wbits", type=int, default=4)
    parser.add_argument("--abits", type=int, default=4)
    parser.add_argument("--load", type=str, default="")
    parser.add_argument("--disable_w_quant", action="store_true")
    parser.add_argument("--disable_a_quant", action="store_true")
    parser.add_argument("--R1_clusters", type=int, default=32)
    parser.add_argument("--R2_clusters", type=int, default=4)
    parser.add_argument("--R3_clusters", type=int, default=4)
    parser.add_argument("--R4_clusters", type=int, default=32)
    parser.add_argument("--R5_clusters", type=int, default=32)
    parser.add_argument("--reorder", type=str, default="12345", help="like 12345 or 1")
    parser.add_argument(
        "--w_quantizer", type=str, default="gptq", choices=["gptq", "normal"]
    )
    parser.add_argument("--limit", type=int, default=-1)
    parser.add_argument("--a_dynamic", action="store_true")
    parser.add_argument("--eval_base_ppl", action="store_true")
    parser.add_argument("--act_dist_plot", action="store_true")
    parser.add_argument("--only_quant_kv", action="store_true")
    parser.add_argument(
        "--pack_weight",
        action="store_true",
        help="enable this to reduce memory consumption",
    )
    parser.add_argument(
        "--multigpu", action="store_true", help="at eval, map model to multiple gpus"
    )

    args = parser.parse_args()
    args.batch_size = 1  # BS=1 is used for zeroShot tasks!
    print(args)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    if "opt" in args.net:
        args.model = f"facebook/{args.net}"
        if not os.path.exists(f"{args.cache_dir}/{args.net.split('-')[0]}/"):
            os.makedirs(f"{args.cache_dir}/{args.net.split('-')[0]}/")
        args.cache_dir = (
            f"{args.cache_dir}/{args.net.split('-')[0]}/{args.net.split('-')[1]}"
        )
        print(args.cache_dir)
        cache_file = f"{args.cache_dir}/torch_model.pth"
        if os.path.exists(cache_file):
            lm = torch.load(cache_file)
        else:
            lm = OPTClass(args)
            torch.save(lm, cache_file)
        lm.model.eval()
    else:
        raise NotImplementedError

    print("=== start quantization ===")
    if args.load:
        print("Loading checkpoint from {}...".format(args.load))
        lm.model.load_state_dict(torch.load(args.load))

    tick = time.time()

    if "opt" in args.model:
        cache_dataloader = (
            f"/tmp/dataloader_opt_{args.calib_dataset}_{args.nsamples}.cache"
        )
        if os.path.exists(cache_dataloader):
            dataloader = torch.load(cache_dataloader)
            print(f"load calibration from {cache_dataloader}")
        else:
            dataloader, testloader = get_loaders(
                args.calib_dataset,
                nsamples=args.nsamples,
                seed=args.seed,
                model=args.model,
                seqlen=lm.seqlen,
                cache_dir=args.cache_dir,
            )
            torch.save(dataloader, cache_dataloader)
        lm.model.eval()
    else:
        raise NotImplementedError()

    args.weight_quant_params = {
        "n_bits": args.wbits,
        "per_channel_axes": [0],
        "symmetric": False,
        "metric": "minmax",
    }
    args.act_quant_params = {
        "n_bits": 16 if args.only_quant_kv else args.abits,
        "per_channel_axes": [],
        "symmetric": False,
        "metric": args.metric,
        "dynamic": args.a_dynamic,
    }
    args.q_quant_params = {
        "n_bits": 16 if args.only_quant_kv else args.abits,
        "per_channel_axes": [],
        "symmetric": False,
        "metric": args.metric,
        "dynamic": args.a_dynamic,
    }
    args.k_quant_params = {
        "n_bits": args.abits,
        "per_channel_axes": [],
        "symmetric": False,
        "metric": args.metric,
        "dynamic": args.a_dynamic,
    }
    args.v_quant_params = {
        "n_bits": args.abits,
        "per_channel_axes": [],
        "symmetric": False,
        "metric": args.metric,
        "dynamic": args.a_dynamic,
    }
    args.layer_norm_out_quant_params = {
        "n_bits": 16 if args.only_quant_kv else max(8, args.abits),
        "per_channel_axes": [],
        "symmetric": False,
        "metric": args.metric,
        "dynamic": args.a_dynamic,
    }
    args.p_quant_params = {
        "n_bits": 16 if args.only_quant_kv else max(8, args.abits),
        "metric": "fix0to1",
    }
    n_clusters = {
        "R1": args.R1_clusters,
        "R2": args.R2_clusters,
        "R3": args.R3_clusters,
        "R4": args.R4_clusters,
        "R5": args.R5_clusters,
    }
    if args.multigpu:
        gpu_id = get_lowest_occupied_gpu(wait_memory=5000)
        lm._device = f"cuda:{gpu_id}"
        print(f"set quantization in gpu {gpu_id}")
    if "opt" in args.model:
        opt_reorder_quantize(
            lm,
            args,
            dataloader,
            n_clusters,
            args.reorder,
        )

        for layer in lm.model.model.decoder.layers:
            if hasattr(layer, "set_quant_state"):
                layer.set_quant_state(
                    not args.disable_w_quant, not args.disable_a_quant
                )

    print(time.time() - tick)

    results = evaluate(lm, args)
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)
    with open(
        f"{args.output_path}/{args.net}.txt",
        "a+",
    ) as f:
        now = datetime.datetime.now()
        formatted_time = now.strftime("%Y-%m-%d %H:%M:%S")
        f.write(
            f"{' '.join(sys.argv)} {formatted_time} \n {args} \n w{args.wbits}a{args.abits} {results}\n\n"
        )


if __name__ == "__main__":
    print(sys.argv)
    main()
