from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

VALID_VARIANTS = ("baseline", "attnres_block", "mhc", "attnres_mhc")
VALID_DATASETS = ("tinystories", "shakespeare")
VALID_PRESETS = ("smoke", "main")


@dataclass
class ModelConfig:
    vocab_size: int = 50_304
    context_length: int = 256
    n_layer: int = 6
    n_head: int = 8
    n_embd: int = 256
    dropout: float = 0.0
    bias: bool = False
    mlp_mult: int = 4
    variant: str = "baseline"
    num_streams: int = 1
    attnres_num_blocks: int = 4
    sinkhorn_iters: int = 20
    sinkhorn_tau: float = 0.05

    def __post_init__(self) -> None:
        if self.variant not in VALID_VARIANTS:
            raise ValueError(f"unsupported variant: {self.variant}")
        if self.n_embd % self.n_head != 0:
            raise ValueError("n_embd must be divisible by n_head")
        if self.attnres_num_blocks < 1:
            raise ValueError("attnres_num_blocks must be >= 1")
        if self.num_streams < 1:
            raise ValueError("num_streams must be >= 1")

    @property
    def attnres_enabled(self) -> bool:
        return self.variant in {"attnres_block", "attnres_mhc"}

    @property
    def mhc_enabled(self) -> bool:
        return self.variant in {"mhc", "attnres_mhc"}

    @property
    def total_sublayers(self) -> int:
        return self.n_layer * 2

    @property
    def block_sublayer_size(self) -> int:
        return max(1, (self.total_sublayers + self.attnres_num_blocks - 1) // self.attnres_num_blocks)

    @property
    def head_dim(self) -> int:
        return self.n_embd // self.n_head


@dataclass
class TrainConfig:
    dataset: str = "tinystories"
    preset: str = "main"
    run_name: str = "main-baseline"
    out_root: str = "artifacts/runs"
    seed: int = 42
    device: str = "auto"
    dtype: str = "bfloat16"
    train_examples: int = 5_000
    val_examples: int = 500
    micro_batch_size: int = 4
    grad_accum_steps: int = 8
    train_steps: int = 2_000
    eval_interval: int = 100
    log_interval: int = 10
    sample_interval: int = 200
    sample_prompt: str = "Once upon a time"
    sample_tokens: int = 80
    eval_batches: int = 16
    learning_rate: float = 3e-4
    weight_decay: float = 0.1
    warmup_steps: int = 100
    grad_clip: float = 1.0

    def __post_init__(self) -> None:
        if self.dataset not in VALID_DATASETS:
            raise ValueError(f"unsupported dataset: {self.dataset}")
        if self.preset not in VALID_PRESETS:
            raise ValueError(f"unsupported preset: {self.preset}")

    @property
    def out_dir(self) -> Path:
        return Path(self.out_root) / self.run_name


@dataclass
class ExperimentConfig:
    model: ModelConfig
    train: TrainConfig

    def to_dict(self) -> dict[str, Any]:
        return {"model": asdict(self.model), "train": asdict(self.train)}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ExperimentConfig":
        return cls(model=ModelConfig(**payload["model"]), train=TrainConfig(**payload["train"]))


def build_preset(preset: str, variant: str) -> ExperimentConfig:
    if preset not in VALID_PRESETS:
        raise ValueError(f"unsupported preset: {preset}")

    if preset == "smoke":
        dataset = "shakespeare"
        model = ModelConfig(
            variant=variant,
            context_length=128,
            n_layer=4,
            n_head=4,
            n_embd=128,
            dropout=0.0,
            num_streams=4 if variant in {"mhc", "attnres_mhc"} else 1,
            attnres_num_blocks=2,
        )
        train = TrainConfig(
            dataset=dataset,
            preset=preset,
            run_name=f"{preset}-{variant}",
            train_examples=1,
            val_examples=1,
            micro_batch_size=8,
            grad_accum_steps=1,
            train_steps=40,
            eval_interval=10,
            log_interval=5,
            sample_interval=20,
            eval_batches=4,
            learning_rate=5e-4,
            warmup_steps=10,
            sample_tokens=40,
        )
        return ExperimentConfig(model=model, train=train)

    model = ModelConfig(
        variant=variant,
        context_length=256,
        n_layer=6,
        n_head=8,
        n_embd=256,
        dropout=0.0,
        num_streams=4 if variant in {"mhc", "attnres_mhc"} else 1,
        attnres_num_blocks=4,
    )
    train = TrainConfig(
        dataset="tinystories",
        preset=preset,
        run_name=f"{preset}-{variant}",
        train_examples=5_000,
        val_examples=500,
        micro_batch_size=4,
        grad_accum_steps=8,
        train_steps=2_000,
        eval_interval=100,
        log_interval=10,
        sample_interval=200,
        eval_batches=16,
        learning_rate=3e-4,
        warmup_steps=100,
        sample_tokens=80,
    )
    return ExperimentConfig(model=model, train=train)
