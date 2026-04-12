from __future__ import annotations

import argparse

from mhc_attnres.config import build_preset
from mhc_attnres.experiment import run_training


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="main", choices=["smoke", "main"])
    parser.add_argument(
        "--variant",
        default="baseline",
        choices=["baseline", "attnres_block", "mhc", "attnres_mhc"],
    )
    parser.add_argument("--train-steps", type=int, default=None)
    parser.add_argument("--run-name", default=None)
    args = parser.parse_args()

    cfg = build_preset(args.preset, args.variant)
    if args.train_steps is not None:
        cfg.train.train_steps = args.train_steps
    if args.run_name is not None:
        cfg.train.run_name = args.run_name

    run_dir = run_training(cfg)
    print(run_dir)


if __name__ == "__main__":
    main()
