# RPTQ: Reorder-Based Post-Training Quantization for Large Language Models
Large-scale language models (LLMs) have shown exceptional performance on various tasks. However, the deployment of LLMs is challenging due to their enormous size. One of the main challenges in quantizing LLMs is the different ranges between the channels, which affects the accuracy and compression ratio of the quantized model.
In our work, we propose a novel reorder-based quantization approach called RPTQ. The RPTQ approach involves rearranging the channels in the activations and then quantizing them in clusters, thereby reducing the impact of the range difference between channels. We also reduce the storage and computation overhead by avoiding explicit reordering.
By implementing the RPTQ approach, we achieved a significant breakthrough by pushing LLM models to 3 bit activation for the first time. Our approach provides an effective solution to quantize LLMs without siginificant accuracy drop.

### Requirements
python packages
- torch >= 2.0.0
- transformers==4.28.0
- omegaconf pycountry sqlitedict lm-eval


### Usage
The RPTQ approach can be applied to OPT models.
```
python main.py opt-1.3b --wbits 4 --abits 4 --eval_ppl --tasks lambada_openai,piqa,arc_easy,arc_challenge,openbookqa,boolq
```

Only quantize K/V cache:
```
python main.py opt-1.3b --wbits 4 --abits 4 --only_quant_kv --eval_ppl --tasks lambada_openai,piqa,arc_easy,arc_challenge,openbookqa,boolq
```

### Results

Perplexity

| Model  | OPT-1.3b |        |        | OPT-6.7b |        |        | OPT-13b |        |        | OPT-30b |        |        | OPT-66b |        |         |
|--------|----------|--------|--------|----------|--------|--------|---------|--------|--------|---------|--------|--------|---------|--------|---------|
| Task   | WIKI     | PT     | C4     | WIKI     | PT     | C4     | WIKI    | PT     | C4     | WIKI    | PT     | C4     | WIKI    | PT     | C4      |
| FP16   | 14.63    | 16.96  | 14.72  | 10.86    | 13.09  | 11.74  | 10.13   | 12.34  | 11.20  | 9.56    | 11.84  | 10.69  | 9.34    | 11.36  | 10.28   |
| W4A16  | 14.78    | 17.21  | 14.92  | 11.18    | 13.62  | 12.07  | 10.29   | 12.45  | 11.27  | 9.55    | 11.91  | 10.74  | 9.30    | 11.42  | 10.31   |
| W4A8   | 15.16    | 17.96  | 15.43  | 11.11    | 14.51  | 12.11  | 10.92   | 14.53  | 11.64  | 10.29   | 12.90  | 11.03  | 9.23    | 11.87  | 10.58   |
| W4A4   | 16.97    | 19.58  | 16.60  | 11.88    | 15.73  | 12.78  | 12.64   | 17.63  | 14.00  | 11.09   | 15.09  | 13.14  | 12.31   | 18.32  | 16.15   |
| W4A4KV | 15.28    | 17.71  | 15.33  | 11.21    | 13.46  | 12.04  | 10.59   | 12.93  | 11.54  | 9.98    | 12.27  | 11.03  | 9.75    | 11.72  | 10.62   |
| W4A3KV | 17.28    | 20.27  | 16.87  | 11.90    | 14.35  | 12.62  | 11.29   | 14.11  | 12.05  | 11.71   | 15.33  | 12.01  | 10.99   | 15.23  | 11.46   |
| W3A3KV | 18.70    | 22.40  | 18.17  | 12.29    | 15.06  | 13.10  | 11.73   | 14.65  | 12.42  | 12.03   | 15.51  | 12.30  | 11.59   | 15.79  | 11.81   |

Zero-shot tasks

| Task   | lambada_openai |        |        |        |        | piqa          |        |        |        |        |
| ------ | -------------- | ------ | ------ | ------ | ------ | ------------- | ------ | ------ | ------ | ------ |
| Model  | 1.3b           | 6.7b   | 13b    | 30b    | 66b    | 1.3b          | 6.7b   | 13b    | 30b    | 66b    |
| FP16   | 57.98%         | 61.84% | 68.60% | 71.41% | 67.14% | 72.47%        | 74.53% | 76.87% | 78.01% | 78.12% |
| W4A16  | 57.46%         | 60.78% | 68.50% | 71.37% | 67.06% | 71.59%        | 74.80% | 76.93% | 78.29% | 78.18% |
| W4A8   | 55.96%         | 61.92% | 68.54% | 71.29% | 66.87% | 71.65%        | 74.91% | 76.93% | 78.45% | 77.80% |
| W4A4   | 53.38%         | 59.59% | 66.71% | 69.78% | 68.56% | 69.74%        | 74.10% | 76.38% | 77.80% | 76.38% |
| W4A4KV | 57.55%         | 61.18% | 68.67% | 71.25% | 70.09% | 71.16%        | 74.53% | 76.16% | 78.23% | 76.87% |
| W4A3KV | 51.54%         | 59.82% | 66.01% | 64.19% | 65.06% | 70.34%        | 73.06% | 75.62% | 68.55% | 74.26% |
| W3A3KV | 47.99%         | 56.95% | 65.47% | 63.32% | 68.13% | 68.93%        | 72.68% | 73.83% | 67.46% | 75.13% |
| Task   | arc_easy       |        |        |        |        | arc_challenge |        |        |        |        |
| Model  | 1.3b           | 6.7b   | 13b    | 30b    | 66b    | 1.3b          | 6.7b   | 13b    | 30b    | 66b    |
| FP16   | 51.05%         | 58.03% | 61.91% | 65.31% | 64.68% | 29.69%        | 33.61% | 35.66% | 38.05% | 38.99% |
| W4A16  | 51.17%         | 57.02% | 61.82% | 65.10% | 64.89% | 30.03%        | 32.59% | 35.49% | 37.96% | 38.99% |
| W4A8   | 49.95%         | 58.37% | 61.57% | 65.44% | 65.31% | 29.18%        | 32.59% | 36.26% | 37.45% | 38.65% |
| W4A4   | 47.64%         | 56.39% | 58.67% | 64.56% | 63.63% | 28.15%        | 31.91% | 34.81% | 37.79% | 37.71% |
| W4A4KV | 49.83%         | 57.11% | 58.41% | 63.13% | 63.63% | 28.32%        | 32.08% | 35.40% | 37.45% | 37.71% |
| W4A3KV | 46.38%         | 55.89% | 56.60% | 46.54% | 56.39% | 27.30%        | 31.99% | 34.30% | 29.60% | 34.89% |
| W3A3KV | 45.20%         | 54.67% | 55.05% | 46.75% | 56.86% | 26.45%        | 29.77% | 33.61% | 29.35% | 33.87% |
| Task   | openbookqa     |        |        |        |        | boolq         |        |        |        |        |
| Model  | 1.3b           | 6.7b   | 13b    | 30b    | 66b    | 1.3b          | 6.7b   | 13b    | 30b    | 66b    |
| FP16   | 33.00%         | 38.00% | 39.00% | 40.20% | 41.60% | 57.73%        | 67.03% | 65.90% | 70.45% | 70.85% |
| W4A16  | 31.80%         | 37.40% | 39.20% | 40.60% | 42.00% | 58.99%        | 59.72% | 66.66% | 70.70% | 70.55% |
| W4A8   | 33.60%         | 38.00% | 38.40% | 39.20% | 41.20% | 56.85%        | 64.00% | 65.84% | 72.35% | 70.45% |
| W4A4   | 31.80%         | 37.40% | 38.00% | 40.80% | 42.40% | 52.84%        | 59.75% | 62.99% | 68.44% | 69.44% |
| W4A4KV | 32.60%         | 39.60% | 39.60% | 39.40% | 42.00% | 56.57%        | 63.76% | 65.13% | 67.73% | 69.63% |
| W4A3KV | 31.40%         | 38.00% | 37.80% | 29.40% | 39.40% | 53.76%        | 62.53% | 61.10% | 63.60% | 66.42% |
| W3A3KV | 31.00%         | 34.60% | 37.60% | 30.80% | 39.40% | 53.21%        | 65.71% | 62.20% | 63.70% | 63.08% |



### Citation
If you use our RPTQ approach in your research, please cite our paper:
@inproceedings{rptq2023,
  title={RPTQ: Reorder-based Post-training Quantization for Large Language Models},
  author={Zhihang Yuan, Lin Niu, },
  year={2023}
}