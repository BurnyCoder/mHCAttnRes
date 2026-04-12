from __future__ import annotations

import argparse
from pathlib import Path

import torch

from mhc_attnres.data import load_corpora
from mhc_attnres.experiment import evaluate_loss, load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    args = parser.parse_args()

    device = torch.device("cpu")
    model, cfg, _ = load_checkpoint(Path(args.checkpoint), device)
    _, val_corpus, _ = load_corpora(cfg.train)
    metrics = evaluate_loss(model, val_corpus, cfg, device)
    print(metrics)


if __name__ == "__main__":
    main()
