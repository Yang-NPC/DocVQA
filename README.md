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
  --output-dir outputs/baseline_qwen3vl8b_docvqa_hard100
```

The hard subset is selected with a lightweight heuristic over the ground-truth metadata: longer questions, longer/multi-token answers, numeric/date-like answers, and examples where provided answers differ.

## Full Baseline

```bash
python scripts/eval_docvqa_qwen3vl.py \
  --model-id Qwen/Qwen3-VL-8B-Instruct \
  --dataset-name lmms-lab/DocVQA \
  --dataset-config DocVQA \
  --split validation \
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
