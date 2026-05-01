# DocVQA GRPO Workspace

This repository starts with a baseline evaluation for `Qwen/Qwen3-VL-8B-Instruct` on DocVQA. Training and full evaluation are expected to run on the server.

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For best Qwen3-VL compatibility, use a recent `transformers` release. If the server image is behind the required release, install from source:

```bash
pip install git+https://github.com/huggingface/transformers
```

## Baseline Smoke Test

Run a small validation subset first:

```bash
python scripts/eval_docvqa_qwen3vl.py \
  --model-id Qwen/Qwen3-VL-8B-Instruct \
  --dataset-name lmms-lab/DocVQA \
  --dataset-config DocVQA \
  --split validation \
  --limit 20 \
  --output-dir outputs/baseline_qwen3vl8b_docvqa_smoke
```

## Hard-Example Baseline

For a more useful early signal, run a limited validation subset biased toward likely harder examples:

```bash
python scripts/eval_docvqa_qwen3vl.py \
  --model-id Qwen/Qwen3-VL-8B-Instruct \
  --dataset-name lmms-lab/DocVQA \
  --dataset-config DocVQA \
  --split validation \
  --selection hard \
  --limit 100 \
  --batch-size 8 \
  --output-dir outputs/baseline_qwen3vl8b_docvqa_hard100
```

The hard subset is selected with a lightweight heuristic over the ground-truth metadata: longer questions, longer/multi-token answers, numeric/date-like answers, and examples where provided answers differ.

## Thinking-Mode Hard Baseline

Qwen thinking mode can be tested separately. It is slower, so use a small hard subset first:

```bash
python scripts/eval_docvqa_qwen3vl.py \
  --model-id Qwen/Qwen3-VL-8B-Instruct \
  --dataset-name lmms-lab/DocVQA \
  --dataset-config DocVQA \
  --split validation \
  --selection hard \
  --limit 20 \
  --thinking-mode on \
  --max-new-tokens 256 \
  --batch-size 4 \
  --temperature 0.6 \
  --top-p 0.95 \
  --top-k 20 \
  --output-dir outputs/baseline_qwen3vl8b_docvqa_hard20_think
```

The script stores the raw `prediction` and a `scored_prediction` with `<think>...</think>` blocks removed before exact match and ANLS scoring.

## Faster A100 Baseline

On an 80G A100, start with batching:

```bash
python scripts/eval_docvqa_qwen3vl.py \
  --model-id Qwen/Qwen3-VL-8B-Instruct \
  --dataset-name lmms-lab/DocVQA \
  --dataset-config DocVQA \
  --split validation \
  --batch-size 16 \
  --torch-dtype bfloat16 \
  --attn-implementation flash_attention_2 \
  --output-dir outputs/baseline_qwen3vl8b_docvqa_val_bs16
```

If evaluation is still image-preprocessing bound, add a pixel cap. This is faster but may change accuracy:

```bash
--max-pixels 1003520
```

For thinking mode, use a smaller batch size because it generates many more tokens:

```bash
--thinking-mode on --max-new-tokens 256 --batch-size 4
```

## Full Baseline

```bash
python scripts/eval_docvqa_qwen3vl.py \
  --model-id Qwen/Qwen3-VL-8B-Instruct \
  --dataset-name lmms-lab/DocVQA \
  --dataset-config DocVQA \
  --split validation \
  --batch-size 16 \
  --output-dir outputs/baseline_qwen3vl8b_docvqa_val
```

The evaluator writes:

- `predictions.jsonl`: one row per example with question, prediction, answers, exact match, and ANLS.
- `metrics.json`: aggregate exact match and ANLS.

## Notes

- DocVQA is usually reported with ANLS. This script computes max-normalized ANLS against all provided ground-truth answers.
- Use `--torch-dtype bfloat16` on modern GPUs. Use `--torch-dtype float16` if BF16 is unavailable.
- Add `--attn-implementation flash_attention_2` if the server has FlashAttention installed.
- Use `--limit` for quick server sanity checks before running the full validation split.
