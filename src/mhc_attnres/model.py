from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from mhc_attnres.config import ModelConfig


def rms_normalize(x: Tensor, eps: float = 1e-6) -> Tensor:
    scale = torch.rsqrt(x.float().pow(2).mean(dim=-1, keepdim=True) + eps)
    return (x.float() * scale).to(x.dtype)


def sinkhorn_log(logits: Tensor, num_iters: int = 20, tau: float = 0.05) -> Tensor:
    n = logits.shape[-1]
    z = logits / tau
    log_marginal = torch.full((n,), -math.log(n), device=logits.device, dtype=logits.dtype)
    u = torch.zeros(n, device=logits.device, dtype=logits.dtype)
    v = torch.zeros(n, device=logits.device, dtype=logits.dtype)

    for _ in range(num_iters):
        u = log_marginal - torch.logsumexp(z + v.unsqueeze(0), dim=1)
        v = log_marginal - torch.logsumexp(z + u.unsqueeze(1), dim=0)

    return torch.exp(z + u.unsqueeze(1) + v.unsqueeze(0)) * n


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return rms_normalize(x, eps=self.eps) * self.weight


class CausalSelfAttention(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.n_head = cfg.n_head
        self.head_dim = cfg.head_dim
        self.qkv = nn.Linear(cfg.n_embd, 3 * cfg.n_embd, bias=cfg.bias)
        self.proj = nn.Linear(cfg.n_embd, cfg.n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: Tensor) -> Tensor:
        bsz, seqlen, dim = x.shape
        qkv = self.qkv(x)
        q, k, v = qkv.split(dim, dim=-1)
        q = q.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        k = k.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        v = v.view(bsz, seqlen, self.n_head, self.head_dim).transpose(1, 2)
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=None,
            dropout_p=self.dropout.p if self.training else 0.0,
            is_causal=True,
        )
        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, dim)
        return self.proj(self.dropout(y))


class MLP(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        hidden = cfg.mlp_mult * cfg.n_embd
        self.fc = nn.Linear(cfg.n_embd, hidden, bias=cfg.bias)
        self.proj = nn.Linear(hidden, cfg.n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.proj(F.gelu(self.fc(x))))


@dataclass
class RoutingSnapshot:
    label: str
    weights: list[float]


class StaticMHC(nn.Module):
    def __init__(self, streams: int, cfg: ModelConfig) -> None:
        super().__init__()
        self.streams = streams
        self.sinkhorn_iters = cfg.sinkhorn_iters
        self.sinkhorn_tau = cfg.sinkhorn_tau

        h_res_init = torch.full((streams, streams), -8.0)
        h_res_init.fill_diagonal_(0.0)
        self.h_res_logits = nn.Parameter(h_res_init)

        h_pre_init = torch.full((streams,), -8.0)
        h_pre_init[0] = 0.0
        self.h_pre_logits = nn.Parameter(h_pre_init)
        self.h_post_logits = nn.Parameter(torch.zeros(streams))

    def project(self) -> tuple[Tensor, Tensor, Tensor]:
        h_res = sinkhorn_log(self.h_res_logits, num_iters=self.sinkhorn_iters, tau=self.sinkhorn_tau)
        h_pre = self.h_pre_logits.softmax(dim=-1)
        h_post = self.h_post_logits.softmax(dim=-1)
        return h_res, h_pre, h_post

    def reduce_inputs(self, routed: Tensor) -> Tensor:
        _, h_pre, _ = self.project()
        return torch.einsum("s,btsd->btd", h_pre, routed)

    def inject_delta(self, delta: Tensor) -> Tensor:
        _, _, h_post = self.project()
        return torch.einsum("s,btd->btsd", h_post, delta)

    def mix_residuals(self, routed: Tensor) -> Tensor:
        h_res, _, _ = self.project()
        return torch.einsum("ij,btid->btjd", h_res, routed)


class ResidualSubLayer(nn.Module):
    def __init__(self, cfg: ModelConfig, kind: str, layer_idx: int) -> None:
        super().__init__()
        self.cfg = cfg
        self.kind = kind
        self.layer_idx = layer_idx
        self.norm = RMSNorm(cfg.n_embd)
        self.query = nn.Parameter(torch.zeros(cfg.n_embd)) if cfg.attnres_enabled else None
        self.mhc = StaticMHC(cfg.num_streams, cfg) if cfg.mhc_enabled else None
        self.core = CausalSelfAttention(cfg) if kind == "attn" else MLP(cfg)
        self.last_route: RoutingSnapshot | None = None

    def route_sources(self, sources: list[Tensor], capture: bool) -> Tensor:
        if len(sources) == 1:
            if capture:
                self.last_route = RoutingSnapshot(label=self.label, weights=[1.0])
            return sources[0]

        stacked = torch.stack(sources, dim=0)
        keys = rms_normalize(stacked)
        pooled = keys.mean(dim=3)
        logits = torch.einsum("d,nbtd->nbt", self.query, pooled)
        weights = logits.softmax(dim=0)
        if capture:
            route = weights.float().mean(dim=(1, 2)).tolist()
            self.last_route = RoutingSnapshot(label=self.label, weights=route)
        return torch.einsum("nbt,nbtsd->btsd", weights, stacked)

    @property
    def label(self) -> str:
        block = self.layer_idx // 2
        return f"{self.kind}_{block}"

    def forward(self, x: Tensor, sources: list[Tensor] | None = None, capture: bool = False) -> tuple[Tensor, Tensor]:
        routed = self.route_sources(sources, capture=capture) if sources is not None else x

        if self.mhc is not None:
            reduced = self.mhc.reduce_inputs(routed)
            delta = self.core(self.norm(reduced))
            delta_stream = self.mhc.inject_delta(delta)
            mixed = self.mhc.mix_residuals(routed)
            return mixed + delta_stream, delta_stream

        reduced = routed.squeeze(2)
        delta = self.core(self.norm(reduced))
        delta_stream = delta.unsqueeze(2)
        return routed + delta_stream, delta_stream


class ResidualTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.token_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.pos_emb = nn.Embedding(cfg.context_length, cfg.n_embd)
        self.stream_emb = nn.Parameter(torch.zeros(cfg.num_streams, cfg.n_embd))
        self.drop = nn.Dropout(cfg.dropout)
        self.layers = nn.ModuleList(
            ResidualSubLayer(cfg, kind="attn" if idx % 2 == 0 else "mlp", layer_idx=idx)
            for idx in range(cfg.total_sublayers)
        )
        self.final_norm = RMSNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def routing_snapshots(self) -> list[RoutingSnapshot]:
        out: list[RoutingSnapshot] = []
        for layer in self.layers:
            if layer.last_route is not None:
                out.append(layer.last_route)
        return out

    def forward(self, idx: Tensor, targets: Tensor | None = None, capture_routing: bool = False) -> tuple[Tensor, Tensor | None]:
        bsz, seqlen = idx.shape
        if seqlen > self.cfg.context_length:
            raise ValueError("input exceeds configured context length")

        pos = torch.arange(seqlen, device=idx.device)
        x = self.token_emb(idx) + self.pos_emb(pos)[None, :, :]
        x = self.drop(x)
        x = x.unsqueeze(2).repeat(1, 1, self.cfg.num_streams, 1)
        x = x + self.stream_emb[None, None, :, :]

        completed_blocks: list[Tensor] = [x] if self.cfg.attnres_enabled else []
        partial_block: Tensor | None = None

        for sub_idx, layer in enumerate(self.layers):
            sources = None
            if self.cfg.attnres_enabled:
                sources = list(completed_blocks)
                if partial_block is not None:
                    sources.append(partial_block)
            x, delta_stream = layer(x, sources=sources, capture=capture_routing)

            if self.cfg.attnres_enabled:
                partial_block = delta_stream if partial_block is None else partial_block + delta_stream
                is_boundary = (sub_idx + 1) % self.cfg.block_sublayer_size == 0
                if is_boundary and (sub_idx + 1) < self.cfg.total_sublayers:
                    completed_blocks.append(partial_block)
                    partial_block = None

        merged = x.sum(dim=2)
        logits = self.lm_head(self.final_norm(merged))
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss
