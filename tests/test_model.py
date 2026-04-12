import torch

from mhc_attnres.config import build_preset
from mhc_attnres.model import ResidualTransformer


def test_all_variants_forward_shape():
    variants = ["baseline", "attnres_block", "mhc", "attnres_mhc"]
    for variant in variants:
        cfg = build_preset("smoke", variant)
        model = ResidualTransformer(cfg.model)
        x = torch.randint(0, cfg.model.vocab_size, (2, 32))
        logits, loss = model(x, x)
        assert logits.shape == (2, 32, cfg.model.vocab_size)
        assert loss is not None


def test_static_mhc_constraints():
    cfg = build_preset("smoke", "attnres_mhc")
    model = ResidualTransformer(cfg.model)
    layer = next(layer for layer in model.layers if layer.mhc is not None)
    h_res, h_pre, h_post = layer.mhc.project()

    assert torch.all(h_res >= 0)
    torch.testing.assert_close(h_res.sum(dim=-1), torch.ones(cfg.model.num_streams), atol=1e-3, rtol=0)
    torch.testing.assert_close(h_res.sum(dim=-2), torch.ones(cfg.model.num_streams), atol=1e-3, rtol=0)
    torch.testing.assert_close(h_pre.sum(), torch.tensor(1.0), atol=1e-6, rtol=0)
    torch.testing.assert_close(h_post.sum(), torch.tensor(1.0), atol=1e-6, rtol=0)


def test_attnres_routes_capture():
    cfg = build_preset("smoke", "attnres_block")
    model = ResidualTransformer(cfg.model)
    x = torch.randint(0, cfg.model.vocab_size, (1, 32))
    model(x, capture_routing=True)
    snapshots = model.routing_snapshots()
    assert snapshots
    assert all(abs(sum(snapshot.weights) - 1.0) < 1e-5 for snapshot in snapshots)
