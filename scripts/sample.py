from __future__ import annotations

import argparse
from pathlib import Path

import torch

from mhc_attnres.experiment import load_checkpoint, sample_text


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--max-new-tokens", type=int, default=80)
    args = parser.parse_args()

    device = torch.device("cpu")
    model, _, _ = load_checkpoint(Path(args.checkpoint), device)
    print(sample_text(model, args.prompt, args.max_new_tokens, device))


if __name__ == "__main__":
    main()
