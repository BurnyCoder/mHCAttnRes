from __future__ import annotations

import math
from pathlib import Path

import torch

from mhc_attnres.config import ExperimentConfig, build_preset
from mhc_attnres.data import get_tokenizer, load_corpora
from mhc_attnres.model import ResidualTransformer
from mhc_attnres.utils import (
    append_jsonl,
    autocast_context,
    cosine_lr,
    count_parameters,
    ensure_dir,
    save_json,
    select_device,
    set_seed,
    timestamp,
)


def evaluate_loss(
    model: ResidualTransformer,
    corpus,
    cfg: ExperimentConfig,
    device: torch.device,
    batches: int | None = None,
) -> dict[str, float]:
    batches = batches or cfg.train.eval_batches
    losses = []
    model.eval()
    with torch.no_grad():
        for _ in range(batches):
            x, y = corpus.get_batch(cfg.train.micro_batch_size, cfg.model.context_length, device)
            with autocast_context(device, cfg.train.dtype):
                _, loss = model(x, y)
            losses.append(float(loss.item()))
    model.train()
    mean_loss = sum(losses) / len(losses)
    return {"loss": mean_loss, "ppl": math.exp(mean_loss)}


@torch.no_grad()
def sample_text(
    model: ResidualTransformer,
    prompt: str,
    max_new_tokens: int,
    device: torch.device,
    temperature: float = 1.0,
    top_k: int = 40,
) -> str:
    tokenizer = get_tokenizer()
    tokens = tokenizer.encode_ordinary(prompt)
    idx = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)

    for _ in range(max_new_tokens):
        idx_cond = idx[:, -model.cfg.context_length :]
        logits, _ = model(idx_cond)
        logits = logits[:, -1, :] / max(temperature, 1e-4)
        if top_k is not None:
            values, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < values[:, [-1]]] = -float("inf")
        probs = torch.softmax(logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        idx = torch.cat([idx, next_token], dim=1)

    return tokenizer.decode(idx[0].tolist())


def save_checkpoint(path: Path, model: ResidualTransformer, optimizer: torch.optim.Optimizer, cfg: ExperimentConfig, step: int, metrics: dict[str, float]) -> None:
    ensure_dir(path.parent)
    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "config": cfg.to_dict(),
            "step": step,
            "metrics": metrics,
        },
        path,
    )


def load_checkpoint(path: Path, device: torch.device) -> tuple[ResidualTransformer, ExperimentConfig, dict]:
    payload = torch.load(path, map_location=device)
    cfg = ExperimentConfig.from_dict(payload["config"])
    model = ResidualTransformer(cfg.model).to(device)
    model.load_state_dict(payload["model_state"])
    return model, cfg, payload


def run_training(cfg: ExperimentConfig) -> Path:
    set_seed(cfg.train.seed)
    device = select_device(cfg.train.device)
    train_corpus, val_corpus, vocab_size = load_corpora(cfg.train)
    cfg.model.vocab_size = vocab_size

    run_dir = ensure_dir(cfg.train.out_dir)
    save_json(run_dir / "config.json", cfg.to_dict())

    model = ResidualTransformer(cfg.model).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.learning_rate, weight_decay=cfg.train.weight_decay)

    best_val = float("inf")
    start = timestamp()

    for step in range(1, cfg.train.train_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        running_loss = 0.0

        for _ in range(cfg.train.grad_accum_steps):
            x, y = train_corpus.get_batch(cfg.train.micro_batch_size, cfg.model.context_length, device)
            with autocast_context(device, cfg.train.dtype):
                _, loss = model(x, y)
                loss = loss / cfg.train.grad_accum_steps
            loss.backward()
            running_loss += float(loss.item())

        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip)
        lr = cosine_lr(step, cfg.train.train_steps, cfg.train.learning_rate, cfg.train.warmup_steps)
        for group in optimizer.param_groups:
            group["lr"] = lr
        optimizer.step()

        if step % cfg.train.log_interval == 0:
            append_jsonl(
                run_dir / "metrics.jsonl",
                {
                    "step": step,
                    "train_loss": running_loss * cfg.train.grad_accum_steps,
                    "lr": lr,
                    "elapsed_s": timestamp() - start,
                },
            )

        if step % cfg.train.eval_interval == 0 or step == cfg.train.train_steps:
            train_metrics = evaluate_loss(model, train_corpus, cfg, device, batches=max(2, cfg.train.eval_batches // 4))
            val_metrics = evaluate_loss(model, val_corpus, cfg, device)
            record = {
                "step": step,
                "train_eval_loss": train_metrics["loss"],
                "train_eval_ppl": train_metrics["ppl"],
                "val_loss": val_metrics["loss"],
                "val_ppl": val_metrics["ppl"],
                "params": count_parameters(model),
                "elapsed_s": timestamp() - start,
            }
            append_jsonl(run_dir / "eval.jsonl", record)
            if val_metrics["loss"] < best_val:
                best_val = val_metrics["loss"]
                save_checkpoint(run_dir / "model_best.pt", model, optimizer, cfg, step, val_metrics)

        if step % cfg.train.sample_interval == 0 or step == cfg.train.train_steps:
            sample = sample_text(model, cfg.train.sample_prompt, cfg.train.sample_tokens, device)
            (run_dir / f"sample_step_{step}.txt").write_text(sample, encoding="utf-8")

    final_metrics = evaluate_loss(model, val_corpus, cfg, device)
    save_checkpoint(run_dir / "model_final.pt", model, optimizer, cfg, cfg.train.train_steps, final_metrics)
    save_json(
        run_dir / "summary.json",
        {
            "run_name": cfg.train.run_name,
            "variant": cfg.model.variant,
            "dataset": cfg.train.dataset,
            "best_val_loss": best_val,
            "final_val_loss": final_metrics["loss"],
            "final_val_ppl": final_metrics["ppl"],
            "params": count_parameters(model),
            "device": str(device),
            "elapsed_s": timestamp() - start,
        },
    )
    return run_dir


def config_from_args(preset: str, variant: str, train_steps: int | None = None) -> ExperimentConfig:
    cfg = build_preset(preset, variant)
    if train_steps is not None:
        cfg.train.train_steps = train_steps
    return cfg
