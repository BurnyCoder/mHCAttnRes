from __future__ import annotations

import hashlib
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import tiktoken
import torch
from datasets import DatasetDict, load_dataset

from mhc_attnres.config import TrainConfig
from mhc_attnres.utils import ensure_dir

SHAKESPEARE_URL = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"


@dataclass
class TokenCorpus:
    tokens: torch.Tensor

    def get_batch(self, batch_size: int, context_length: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        upper = self.tokens.numel() - context_length - 1
        if upper <= 0:
            raise ValueError("token corpus is too small for the configured context length")
        starts = torch.randint(0, upper, (batch_size,))
        x = torch.stack([self.tokens[start : start + context_length] for start in starts])
        y = torch.stack([self.tokens[start + 1 : start + 1 + context_length] for start in starts])
        return x.to(device), y.to(device)


def get_tokenizer():
    return tiktoken.get_encoding("gpt2")


def _token_cache_path(cache_dir: Path, dataset: str, split: str, num_examples: int) -> Path:
    key = f"{dataset}:{split}:{num_examples}".encode("utf-8")
    digest = hashlib.sha1(key).hexdigest()[:12]
    return cache_dir / f"{dataset}_{split}_{num_examples}_{digest}.npy"


def _download_shakespeare(data_dir: Path) -> str:
    ensure_dir(data_dir)
    path = data_dir / "tiny_shakespeare.txt"
    if not path.exists():
        urllib.request.urlretrieve(SHAKESPEARE_URL, path)
    return path.read_text(encoding="utf-8")


def _load_tinystories() -> DatasetDict:
    dataset = load_dataset("roneneldan/TinyStories")
    if isinstance(dataset, DatasetDict):
        return dataset
    raise TypeError("unexpected TinyStories dataset shape")


def _collect_texts(dataset: str, split: str, num_examples: int, data_dir: Path) -> list[str]:
    if dataset == "shakespeare":
        text = _download_shakespeare(data_dir)
        return [text]

    data = _load_tinystories()
    if split == "validation" and "validation" not in data:
        examples = data["train"].select(range(num_examples))
        return list(examples["text"])
    records = data[split].select(range(num_examples))
    return list(records["text"])


def _encode_texts(texts: Iterable[str]) -> np.ndarray:
    tokenizer = get_tokenizer()
    tokens: list[int] = []
    eot = tokenizer.eot_token
    for text in texts:
        ids = tokenizer.encode_ordinary(text)
        tokens.extend(ids)
        tokens.append(eot)
    return np.asarray(tokens, dtype=np.uint16)


def load_corpora(train_cfg: TrainConfig, cache_root: Path = Path("cache"), data_root: Path = Path("data")) -> tuple[TokenCorpus, TokenCorpus, int]:
    ensure_dir(cache_root)
    ensure_dir(data_root)

    train_split = "train"
    val_split = "validation"
    train_cache = _token_cache_path(cache_root, train_cfg.dataset, train_split, train_cfg.train_examples)
    val_cache = _token_cache_path(cache_root, train_cfg.dataset, val_split, train_cfg.val_examples)

    if not train_cache.exists():
        texts = _collect_texts(train_cfg.dataset, train_split, train_cfg.train_examples, data_root)
        np.save(train_cache, _encode_texts(texts))
    if not val_cache.exists():
        texts = _collect_texts(train_cfg.dataset, val_split, train_cfg.val_examples, data_root)
        np.save(val_cache, _encode_texts(texts))

    train_tokens = torch.from_numpy(np.load(train_cache).astype(np.int64))
    val_tokens = torch.from_numpy(np.load(val_cache).astype(np.int64))
    vocab_size = get_tokenizer().n_vocab
    return TokenCorpus(train_tokens), TokenCorpus(val_tokens), vocab_size
