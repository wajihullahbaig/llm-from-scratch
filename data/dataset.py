"""Tokenisation and sliding-window datasets over one flat token array."""
import os
from typing import List, Tuple

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from configs.config import Config, DataConfig
from data.corpus import load_corpus


def tokenize(texts: List[str], tokenizer, batch_size: int = 1024) -> np.ndarray:
    """Encode documents and join them with <|endoftext|>, so short documents and tails are not thrown away."""
    chunks = []
    for i in range(0, len(texts), batch_size):
        for ids in tokenizer.encode_ordinary_batch(texts[i:i + batch_size]):
            chunks.append(np.array(ids + [tokenizer.eot_token], dtype=np.uint16))
    return np.concatenate(chunks)


def load_tokens(cfg: DataConfig, tokenizer) -> np.ndarray:
    """Tokenise the corpus once and cache it as .npy; later runs memory-map the cache."""
    name = cfg.source if cfg.source == "alice" else f"{cfg.source}_{cfg.subset_ratio}"
    cache = os.path.join(cfg.cache_dir, f"{name}.npy")
    if not os.path.exists(cache):
        tokens = tokenize(load_corpus(cfg.source, cfg.subset_ratio, cfg.cache_dir), tokenizer)
        os.makedirs(cfg.cache_dir, exist_ok=True)
        np.save(cache, tokens)
    return np.load(cache, mmap_mode="r")


def split_tokens(tokens: np.ndarray, val_ratio: float) -> Tuple[np.ndarray, np.ndarray]:
    cut = int(len(tokens) * (1 - val_ratio))
    return tokens[:cut], tokens[cut:]


class TokenWindows(Dataset):
    """(input, target) windows of `context_length` tokens; targets are the inputs shifted by one."""

    def __init__(self, tokens: np.ndarray, context_length: int, stride: int):
        self.tokens = tokens
        self.context_length = context_length
        self.stride = stride

    def __len__(self) -> int:
        return max(0, (len(self.tokens) - self.context_length - 1) // self.stride + 1)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        start = idx * self.stride
        window = torch.from_numpy(self.tokens[start:start + self.context_length + 1].astype(np.int64))
        return window[:-1], window[1:]


def build_loaders(tokens: np.ndarray, cfg: Config, device: torch.device) -> Tuple[DataLoader, DataLoader]:
    ctx = cfg.model.context_length
    train_tokens, val_tokens = split_tokens(tokens, cfg.data.val_ratio)
    train_set = TokenWindows(train_tokens, ctx, cfg.data.stride or ctx)
    val_set = TokenWindows(val_tokens, ctx, ctx)  # no overlap: every validation token is scored once
    if len(train_set) < cfg.training.batch_size or len(val_set) == 0:
        raise ValueError(f"Not enough tokens ({len(tokens):,}) for context_length={ctx} and "
                         f"batch_size={cfg.training.batch_size}. Use more data or lower them.")
    common = dict(batch_size=cfg.training.batch_size, num_workers=cfg.training.num_workers,
                  pin_memory=device.type == "cuda")
    return (DataLoader(train_set, shuffle=True, drop_last=True, generator=torch.Generator(), **common),
            DataLoader(val_set, shuffle=False, **common))
