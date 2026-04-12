# mHC + Attention Residuals

Root-level research implementation of a tiny decoder-only language model that combines:

- Block Attention Residuals
- static manifold-constrained hyper-connections (`mHC-lite`)

## Repository Layout

- `src/mhc_attnres/`: package code
- `scripts/`: CLI entrypoints
- `tests/`: unit and smoke tests
- `paper/`: LaTeX paper
- `artifacts/`: checkpoints, metrics, plots, and samples

The embedded directories `attn_res/`, `open-attention-residuals/`, and `mHC-manifold-constrained-hyper-connections/` are treated as local references only.

## Variants

- `baseline`: single-stream residual transformer
- `attnres_block`: single-stream transformer with Block Attention Residuals
- `mhc`: multi-stream transformer with static constrained residual routing
- `attnres_mhc`: combined Block Attention Residuals + static mHC-lite

## Datasets

- Main: `TinyStories`
- Smoke/regression: `tiny Shakespeare`

Both paths use GPT-2 BPE via `tiktoken`.

## Setup

Create the environment and install dependencies with `uv`:

```bash
uv sync
```

If the environment has not been bootstrapped yet, install the dependencies first:

```bash
uv add torch --index-url https://download.pytorch.org/whl/cu128
uv add datasets tiktoken matplotlib numpy
uv add --dev pytest ruff
```

## Quick Start

Run the architecture on dummy token IDs without training:

```bash
uv run python scripts/dummy_run.py --all-variants --seq-len 32
```

Run a single variant:

```bash
uv run python scripts/dummy_run.py --variant attnres_mhc --seq-len 64
```

This validates:

- forward pass shape contracts
- scalar loss computation on dummy targets
- depth-routing summaries for AttnRes variants
- doubly-stochastic and simplex constraints for mHC variants

## Optional Scripts

Training and evaluation scripts remain in the repo, but they are not required for the current architecture-only flow:

```bash
uv run python scripts/train.py --preset smoke --variant baseline
uv run python scripts/eval.py --checkpoint <checkpoint>
uv run python scripts/sample.py --checkpoint <checkpoint>
uv run python scripts/visualize_routes.py --checkpoint <checkpoint>
```

## Notes

- Full AttnRes is not part of the v1 experiment matrix.
- The mHC implementation here is the static-per-sublayer simplified formulation from the local reference code, not the dynamic input-conditioned version from the paper.
- The current requested validation mode is architecture-only with dummy data and no training.
