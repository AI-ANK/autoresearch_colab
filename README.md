# autoresearch_colab

**Run [autoresearch](https://github.com/karpathy/autoresearch) on free GPUs with free LLM APIs. Total cost: $0.**

A fork of Andrej Karpathy's [autoresearch](https://github.com/karpathy/autoresearch) that runs autonomously on Google Colab's free T4 GPU (16GB VRAM). Instead of requiring an H100 and a paid Claude/Codex API, this fork uses free LLM APIs (OpenRouter, Groq, Gemini) with daisy-chain failover. It handles everything automatically: GPU detection, hyperparameter tuning, Google Drive persistence, and disconnect recovery.

## Quick Start (Colab)

1. **Open the notebook**: Open [`colab_runner.ipynb`](colab_runner.ipynb) in Google Colab
2. **Select GPU**: `Runtime > Change runtime type > T4 GPU`
3. **Add API key(s)**: Click the key icon in the left sidebar (Secrets) and add at least one:
   - `OPENROUTER_API_KEY` — Get free at [openrouter.ai](https://openrouter.ai)
   - `GROQ_API_KEY` — Get free at [console.groq.com](https://console.groq.com)
   - `GEMINI_API_KEY` — Get free at [aistudio.google.com](https://aistudio.google.com/apikey)
4. **Run all cells** — the notebook handles everything else

The notebook runs a smoke test first, then enters the autonomous experiment loop. Each experiment takes ~5-6 minutes. You can close the tab and come back later — state is saved to Google Drive.

## What changed from upstream

This fork adds automatic GPU detection and adaptation. The original H100 codepath is untouched — if you run on an H100, everything works exactly as before.

**On GPUs with <20GB VRAM (T4, P100, V100):**

- **Attention**: Uses PyTorch's built-in SDPA instead of Flash Attention 3 (which requires Ampere+)
- **Dtype**: float16 + GradScaler instead of bfloat16 (T4 doesn't support bf16)
- **Hyperparams**: Auto-reduced for 16GB — `DEPTH=4`, `DEVICE_BATCH_SIZE=32`, `MAX_SEQ_LEN=512`, `WINDOW_PATTERN="L"`
- **MFU**: Per-GPU peak FLOPS lookup (H100/H200/A100/T4/V100/P100) for accurate utilization reporting

**New files:**

- **`colab_runner.ipynb`** — Full Colab notebook with autonomous experiment loop, Google Drive persistence, anti-disconnect keep-alive, and disconnect recovery
- **`program_colab.md`** — T4-specific agent instructions with experiment ideas tuned for 16GB VRAM

## Free LLM APIs

The notebook uses a daisy-chain client that tries each provider in sequence. If one hits rate limits, the next picks up automatically.

| Provider | Model | Free Tier | Sign Up |
|----------|-------|-----------|---------|
| **OpenRouter** | Qwen3 Coder (free) | ~200 req/day | [openrouter.ai](https://openrouter.ai) |
| **Groq** | Llama 3.3 70B | 30 req/min, 6000 req/day | [console.groq.com](https://console.groq.com) |
| **Gemini** | Gemini 2.5 Flash | 10 req/min, 250-500 req/day | [aistudio.google.com](https://aistudio.google.com/apikey) |

Any single provider is enough for overnight runs (~200 experiments). Adding more providers gives better resilience against rate limits.

## How it works

Give an AI agent a small but real LLM training setup and let it experiment autonomously. It modifies the code, trains for 5 minutes, checks if the result improved, keeps or discards, and repeats. You come back to a log of experiments and (hopefully) a better model.

The repo is deliberately kept small with just a few files that matter:

- **`prepare.py`** — fixed constants, one-time data prep (downloads training data, trains a BPE tokenizer), and runtime utilities (dataloader, evaluation). Not modified by the agent.
- **`train.py`** — the single file the agent edits. Contains the full GPT model, optimizer (Muon + AdamW), and training loop. Everything is fair game: architecture, hyperparameters, optimizer, batch size, etc.
- **`program_colab.md`** — T4-specific agent instructions for the autonomous LLM orchestrator.
- **`colab_runner.ipynb`** — the Colab notebook that orchestrates the autonomous loop.

Training runs for a **fixed 5-minute time budget** (wall clock, excluding startup/compilation). The metric is **val_bpb** (validation bits per byte) — lower is better.

## Project structure

```
colab_runner.ipynb  — Colab notebook: setup, autonomous loop, recovery
program_colab.md    — T4-specific agent instructions
prepare.py          — constants, data prep + runtime utilities (do not modify)
train.py            — model, optimizer, training loop (agent modifies this)
program.md          — original agent instructions (for H100)
pyproject.toml      — dependencies
```

## Running locally (H100/A100)

If you have an H100 or A100, everything works as upstream:

```bash
# 1. Install uv project manager (if you don't already have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install dependencies
uv sync

# 3. Download data and train tokenizer (one-time, ~2 min)
uv run prepare.py

# 4. Manually run a single training experiment (~5 min)
uv run train.py
```

Then point your Claude/Codex agent at `program.md` and let it go.

## Design choices

- **Single file to modify.** The agent only touches `train.py`. This keeps the scope manageable and diffs reviewable.
- **Fixed time budget.** Training always runs for exactly 5 minutes, regardless of your specific platform. This makes experiments directly comparable regardless of what the agent changes (model size, batch size, architecture, etc.), and means autoresearch will find the most optimal model for your platform in that time budget.
- **Self-contained.** No external dependencies beyond PyTorch and a few small packages. No distributed training, no complex configs. One GPU, one file, one metric.
- **Zero cost.** This fork is designed so that anyone can run autoresearch with no paid APIs and no paid GPU. Free Colab T4 + free LLM APIs = $0.

## Credits

This is a fork of [karpathy/autoresearch](https://github.com/karpathy/autoresearch). All credit for the original concept, training code, and experiment framework goes to Andrej Karpathy.

## License

MIT
