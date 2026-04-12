# Autonomous Research Agent

## Current Understanding

- The root project is now a new implementation target, separate from the three embedded reference repos.
- Two source papers were read before implementation:
  - `Attention Residuals.pdf`
  - `mHC - Manifold-Constrained Hyper-Connections.pdf`
- The local machine is a single-GPU system with an RTX 5070 Laptop GPU (8 GB VRAM), 16 GB RAM, and Python 3.12.3.
- The current user request is architecture-only validation. Do not train anything unless the user asks again.

## Active Decision Set

- Main implementation target: tiny nanoGPT-style decoder LM
- AttnRes scope: Block Attention Residuals only
- mHC scope: static `mHC-lite`, not the dynamic paper formulation
- Variant matrix:
  - `baseline`
  - `attnres_block`
  - `mhc`
  - `attnres_mhc`

## Architecture Plan

- Keep the hidden state as `B x T x S x D` whenever mHC is enabled.
- For AttnRes, maintain:
  - completed block summaries
  - an in-progress partial block
- Depth-attention weights are stream-shared:
  - score each source after RMS normalization
  - average key features over streams
  - apply one learned pseudo-query vector per sublayer
- For mHC-lite, use per-sublayer static parameters:
  - `H_res_logits`
  - `H_pre_logits`
  - `H_post_logits`
- Project `H_res` with Sinkhorn.
- Constrain `H_pre` and `H_post` with softmax.

## Immediate Next Steps

1. Keep the architecture runnable on dummy token data for all four variants.
2. Avoid dataset downloads and avoid checkpoint-producing training runs.
3. Preserve training utilities as optional code paths only.
4. Update docs to point at the dummy-run entrypoint as the primary validation path.

## Open Questions

- None for the current architecture-only scope.

## Quality Checklist

- [x] Current architecture design makes sense
- [x] Code runs end to end on dummy data
- [x] Outputs are practically sensible for random initialization
- [x] Logging is sufficient for architecture validation
- [x] `README.md` is accurate
- [x] `AGENTS.md` is accurate
- [ ] Root repo is committed and pushed
- [x] Tests are passing
