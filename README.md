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

Qwen thinking mode should be tested with the thinking checkpoint, `Qwen/Qwen3-VL-8B-Thinking`. It is slower, so use a small hard subset first:

```bash
python scripts/eval_docvqa_qwen3vl.py \
  --model-id Qwen/Qwen3-VL-8B-Thinking \
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

For thinking-mode runs, `metrics.json` also reports `thinking_tag_count`, `thinking_tag_rate`, and `missing_think_tag_count`, so you can see how often the model failed to include a complete `<think>...</think>` block.

For the vLLM OpenAI-compatible evaluator, `--thinking-mode on` sends `chat_template_kwargs.enable_thinking=true` in `extra_body`. This is required for Qwen thinking mode; appending `/think` alone may not produce `<think>...</think>` traces. The served vLLM model must also be `Qwen/Qwen3-VL-8B-Thinking`; serving `Qwen/Qwen3-VL-8B-Instruct` will usually produce zero think tags.

## Acceleration Strategy

Use this order when scaling from quick checks to full validation:

1. Start with a hard smoke test:

```bash
--selection hard --limit 20
```

This catches obvious prompt/model/dataset issues without spending a full validation pass.

2. Batch the Transformers evaluator:

```bash
--batch-size 16 --torch-dtype bfloat16
```

On an 80G A100, try `--batch-size 16`, then `24`, then `32`. If memory is fine but throughput is still low, the bottleneck is likely image preprocessing or generation scheduling rather than model weights.

3. Cap image pixels for faster sweeps:

```bash
--max-pixels 1003520
```

This reduces multimodal preprocessing and vision tokens. It can change accuracy, so report whether a run used a pixel cap.

4. Use FlashAttention if installed:

```bash
--attn-implementation flash_attention_2
```

If FlashAttention is not installed correctly, omit the flag rather than blocking baseline runs.

5. Use vLLM for fastest visual inference:

```bash
VLLM_USE_V1=0 vllm serve Qwen/Qwen3-VL-8B-Instruct \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 8192
```

Then start conservatively:

```bash
--concurrency 8
```

If stable, increase `--concurrency`, `--max-num-seqs`, and `--max-num-batched-tokens` one at a time. Qwen3-VL can hit a vLLM V1 DeepStack scheduler crash under high multimodal concurrency, so keep `VLLM_USE_V1=0` if you see `Requested more deepstack tokens than available in buffer`.

6. Keep thinking mode separate and use the Thinking checkpoint:

```bash
--model-id Qwen/Qwen3-VL-8B-Thinking --thinking-mode on --max-new-tokens 256
```

Thinking mode is slower because it generates more tokens. Use smaller batches or lower vLLM concurrency, for example `--batch-size 4` in Transformers or `--concurrency 2` in vLLM. Check `missing_think_tag_count` to see how often a thinking run omitted a complete `<think>...</think>` trace.

## Faster A100 Baseline

On an 80G A100, start with batching in the Transformers evaluator:

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

## Fast vLLM Server Baseline

For the fastest VLM baseline on an A100, serve Qwen3-VL with vLLM and run the OpenAI-compatible evaluator. This keeps the task visual, unlike an OCR plus text-only LLM pipeline.

Start the server:

```bash
pip install -U vllm
vllm serve Qwen/Qwen3-VL-8B-Instruct \
  --dtype bfloat16 \
  --max-model-len 8192 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 8192
```

Then evaluate with high request concurrency:

```bash
python scripts/eval_docvqa_qwen3vl_vllm_server.py \
  --model-id Qwen/Qwen3-VL-8B-Instruct \
  --base-url http://localhost:8000/v1 \
  --dataset-name lmms-lab/DocVQA \
  --dataset-config DocVQA \
  --split validation \
  --concurrency 8 \
  --max-pixels 1003520 \
  --output-dir outputs/baseline_qwen3vl8b_docvqa_val_vllm
```

If vLLM V1 hits a Qwen3-VL DeepStack scheduler error under multimodal concurrency, restart the server with `VLLM_USE_V1=0`. Then raise `--concurrency`, `--max-num-seqs`, and `--max-num-batched-tokens` gradually.

### vLLM Thinking Server

Stop the Instruct vLLM server before serving the Thinking checkpoint on the same port:

```bash
MODEL_ID=Qwen/Qwen3-VL-8B-Thinking \
PORT=8000 \
PID_FILE=/tmp/vllm_qwen3vl_thinking.pid \
LOG_FILE=/tmp/vllm_qwen3vl_thinking.log \
scripts/run_vllm_qwen3vl_thinking.sh
```

The Thinking runner starts vLLM with `--reasoning-parser qwen3`, so the OpenAI response can expose reasoning separately. The evaluator wraps that field back into `<think>...</think>` before writing `predictions.jsonl` and counting think tags.

Then evaluate with the same Thinking model id:

```bash
python scripts/eval_docvqa_qwen3vl_vllm_server.py \
  --model-id Qwen/Qwen3-VL-8B-Thinking \
  --base-url http://localhost:8000/v1 \
  --dataset-name lmms-lab/DocVQA \
  --dataset-config DocVQA \
  --split validation \
  --selection hard \
  --limit 200 \
  --thinking-mode on \
  --max-tokens 1024 \
  --temperature 0.6 \
  --top-p 0.95 \
  --top-k 20 \
  --concurrency 2 \
  --max-pixels 1003520 \
  --output-dir outputs/baseline_qwen3vl8b_docvqa_hard200_vllm_think
```

If `thinking_tag_count` remains zero, verify `/v1/models` reports `Qwen/Qwen3-VL-8B-Thinking` rather than `Qwen/Qwen3-VL-8B-Instruct`.

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

## GRPO LoRA Training Scaffold

The GRPO reward logic is in `rewards/docvqa_grpo_reward.py`.

Reward design:

- Absolute accuracy: extract exactly one `<answer>...</answer>` block and exact-match against DocVQA answers after lowercase/whitespace normalization. Correct answers get `+1.5`.
- Visual grounding: inspect complete `<think>...</think>` blocks. If a sufficiently long ground-truth answer appears in the thinking text, add `+0.5`.
- Short answers below the grounding threshold skip the grounding reward to reduce false positives.

Validate reward behavior:

```bash
python -m unittest tests/test_docvqa_reward.py -v
```

EasyR1 uses the adapter in `rewards/easyr1_docvqa_reward.py`.

Prepare a local EasyR1 image-text JSONL dataset:

```bash
python scripts/prepare_docvqa_easyr1.py \
  --split validation \
  --limit 200 \
  --output-dir data/docvqa_easyr1 \
  --output-name train.jsonl \
  --num-workers 8 \
  --skip-existing-images
```

For a quick validation split, prepare a separate file:

```bash
python scripts/prepare_docvqa_easyr1.py \
  --split validation \
  --limit 50 \
  --output-dir data/docvqa_easyr1 \
  --output-name val.jsonl \
  --num-workers 8 \
  --skip-existing-images
```

EasyR1 training config:

```bash
configs/easyr1/qwen3vl_8b_docvqa_grpo_lora.yaml
```

Launch from an EasyR1 checkout with this repository on `PYTHONPATH` or copied into the EasyR1 workspace:

```bash
bash scripts/run_easyr1_docvqa_grpo.sh
```

EasyR1 installation can take several minutes because its requirements include heavy runtime packages such as `flash-attn`, `ray`, and `vllm`. In Colab, avoid quiet/captured pip output for the EasyR1 install so dependency resolution or CUDA extension builds are visible.

For Colab runtimes with Python 3.12, Torch `2.10.0+cu128`, and CUDA 12.8, install a prebuilt `flash-attn` wheel before installing EasyR1 to avoid compiling from source:

```bash
python -m pip install --no-deps \
  "https://github.com/lesj0610/flash-attention/releases/download/v2.8.3-cu12-torch2.10-cp312/flash_attn-2.8.3%2Bcu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl"
```

Local Colab notebook:

```bash
colab_docvqa_grpo_training.ipynb
```

Notebook files are ignored by git.
