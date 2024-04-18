---
license: mit
---

This is a reproduction of the <a href="https://arxiv.org/abs/2402.17764"> BitNet b1.58</a> paper. The models are trained with <a href="https://github.com/togethercomputer/RedPajama-Data">RedPajama dataset</a> for 100B tokens. The hypers, as well as two-stage LR and weight decay, are implemented as suggested in their following <a href="https://github.com/microsoft/unilm/blob/master/bitnet/The-Era-of-1-bit-LLMs__Training_Tips_Code_FAQ.pdf">paper</a>. All models are open-source in the <a href="https://huggingface.co/1bitLLM">repo</a>. We will train larger models and/or more tokens when resource is available.

## Results
PPL and zero-shot accuracy:
| Models | PPL| ARCe| ARCc| HS | BQ | OQ | PQ | WGe | Avg
|-------|-------|-------|-------|-------|-------|-------|-------|-------|-------|
| FP16 700M (reported) | 12.33 | 54.7 | 23.0 | 37.0 | 60.0 | 20.2 | 68.9 | 54.8 | 45.5 |
| BitNet b1.58 700M (reported) | 12.87 | 51.8 | 21.4 | 35.1 | 58.2 | 20.0 | 68.1 | 55.2 | 44.3 |
| BitNet b1.58 700M (reproduced) | 12.78 | 51.4 | 21.8 | 35.0 | 59.6 | 20.6 | 67.5 | 55.4 | 44.5 |
| FP16 1.3B (reported)    | 11.25  | 56.9 | 23.5 | 38.5 | 59.1 | 21.6 | 70.0 | 53.9 | 46.2
| BitNet b1.58 1.3B (reported)    | 11.29  | 54.9 | 24.2 | 37.7 | 56.7 | 19.6 | 68.8 | 55.8 | 45.4 |
| BitNet b1.58 1.3B (reproduced)    | 11.19 | 55.8 | 23.7 | 37.6 | 59.0 | 20.2 | 69.2 | 56.0 | 45.9
| FP16 3B (reported)    | 10.04   | 62.1 | 25.6 | 43.3 | 61.8 | 24.6 | 72.1 | 58.2 | 49.7
| BitNet b1.58 3B (reported)    | 9.91   | 61.4 | 28.3 | 42.9 | 61.5 | 26.6 | 71.5 | 59.3 | 50.2
| BitNet b1.58 3B (reproduced)    | 9.88 | 60.9 | 28.0 | 42.3 | 58.3 | 26.0 | 71.4 | 60.3 | 49.6 |

The differences between the reported numbers and the reproduced results are possibly variances from the training data processing, seeds, or other random factors.

## Evaluation
The evaluation pipelines are from the paper authors. Here is the commands to run the evaluation:
```
pip install lm-eval==0.3.0
```
```
python eval_ppl.py --hf_path 1bitLLM/bitnet_b1_58-3B --seqlen 2048
```
```
python eval_task.py --hf_path 1bitLLM/bitnet_b1_58-3B \
    --batch_size 1 \
    --tasks \
    --output_path result.json \
    --num_fewshot 0 \
    --ctx_size 2048
```
