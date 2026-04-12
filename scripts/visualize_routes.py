from __future__ import annotations

import argparse
from pathlib import Path

from mhc_attnres.visualize import plot_routing_heatmap


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--text",
        default="Once upon a time there was a small dragon who wanted to learn how to read.",
    )
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    checkpoint = Path(args.checkpoint)
    output = Path(args.output) if args.output else checkpoint.parent / "routing_heatmap.png"
    print(plot_routing_heatmap(checkpoint, args.text, output))


if __name__ == "__main__":
    main()
