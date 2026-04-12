from __future__ import annotations

import argparse

import torch

from mhc_attnres.config import VALID_VARIANTS, build_preset
from mhc_attnres.model import ResidualTransformer
from mhc_attnres.utils import count_parameters, select_device, set_seed


def summarize_variant(
    variant: str,
    batch_size: int,
    seq_len: int,
    device: torch.device,
) -> None:
    cfg = build_preset("smoke", variant)
    cfg.model.context_length = max(cfg.model.context_length, seq_len)

    model = ResidualTransformer(cfg.model).to(device)
    model.eval()

    tokens = torch.randint(
        low=0,
        high=cfg.model.vocab_size,
        size=(batch_size, seq_len),
        device=device,
    )

    with torch.no_grad():
        logits, loss = model(tokens, tokens, capture_routing=True)

    print(f"variant={variant}")
    print(f"  params={count_parameters(model):,}")
    print(f"  input_shape={tuple(tokens.shape)}")
    print(f"  logits_shape={tuple(logits.shape)}")
    print(f"  loss={loss.item():.6f}")

    snapshots = model.routing_snapshots()
    if snapshots:
        print("  routing:")
        for snapshot in snapshots[:4]:
            rounded = [round(weight, 4) for weight in snapshot.weights]
            print(f"    {snapshot.label}: {rounded}")

    mhc_layer = next((layer for layer in model.layers if layer.mhc is not None), None)
    if mhc_layer is not None:
        h_res, h_pre, h_post = mhc_layer.mhc.project()
        print("  mhc_constraints:")
        print(f"    h_res_row_sums={[round(v, 4) for v in h_res.sum(dim=-1).tolist()]}")
        print(f"    h_res_col_sums={[round(v, 4) for v in h_res.sum(dim=-2).tolist()]}")
        print(f"    h_pre={[round(v, 4) for v in h_pre.tolist()]}")
        print(f"    h_post={[round(v, 4) for v in h_post.tolist()]}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=VALID_VARIANTS, default=None)
    parser.add_argument("--all-variants", action="store_true")
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seq-len", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_seed(args.seed)
    device = select_device(args.device)

    if args.all_variants:
        variants = list(VALID_VARIANTS)
    elif args.variant is not None:
        variants = [args.variant]
    else:
        variants = ["attnres_mhc"]

    print(f"device={device}")
    for idx, variant in enumerate(variants):
        if idx:
            print()
        summarize_variant(
            variant=variant,
            batch_size=args.batch_size,
            seq_len=args.seq_len,
            device=device,
        )


if __name__ == "__main__":
    main()
