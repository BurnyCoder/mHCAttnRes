from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import torch

from mhc_attnres.data import get_tokenizer
from mhc_attnres.experiment import load_checkpoint

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def plot_routing_heatmap(checkpoint: Path, text: str, output_path: Path) -> Path:
    device = torch.device("cpu")
    model, _, _ = load_checkpoint(checkpoint, device)
    model.eval()

    tokenizer = get_tokenizer()
    input_ids = torch.tensor(tokenizer.encode_ordinary(text), dtype=torch.long, device=device).unsqueeze(0)
    with torch.no_grad():
        model(input_ids[:, : model.cfg.context_length], capture_routing=True)

    snapshots = model.routing_snapshots()
    max_sources = max(len(snapshot.weights) for snapshot in snapshots)
    matrix = np.full((len(snapshots), max_sources), np.nan)

    labels = []
    for row, snapshot in enumerate(snapshots):
        labels.append(snapshot.label)
        matrix[row, : len(snapshot.weights)] = snapshot.weights

    fig, ax = plt.subplots(figsize=(max(8, max_sources), max(4, len(snapshots) * 0.35)))
    masked = np.ma.masked_invalid(matrix)
    cmap = plt.cm.viridis.copy()
    cmap.set_bad(color="#202020")
    image = ax.imshow(masked, aspect="auto", cmap=cmap, interpolation="nearest")
    ax.set_title("Depth Routing Weights")
    ax.set_xlabel("Source Index")
    ax.set_ylabel("Sublayer")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=7)
    plt.colorbar(image, ax=ax, shrink=0.8, label="Mean softmax weight")
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path
