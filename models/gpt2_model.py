# Implementation of the GPT model and its components
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from configs.config import ModelConfig


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention using PyTorch's fused (flash / memory-efficient) kernel."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.n_heads = cfg.n_heads
        self.drop_rate = cfg.drop_rate
        self.qkv = nn.Linear(cfg.emb_dim, 3 * cfg.emb_dim, bias=cfg.qkv_bias)  # query, key and value in one matmul
        self.out_proj = nn.Linear(cfg.emb_dim, cfg.emb_dim)

    def forward(self, x):
        b, t, d = x.shape
        # (b, t, 3*d) -> 3 x (b, n_heads, t, head_dim)
        q, k, v = self.qkv(x).view(b, t, 3, self.n_heads, d // self.n_heads).permute(2, 0, 3, 1, 4)
        dropout = self.drop_rate if self.training else 0.0
        y = F.scaled_dot_product_attention(q, k, v, dropout_p=dropout, is_causal=True)
        return self.out_proj(y.transpose(1, 2).reshape(b, t, d))


class TransformerBlock(nn.Module):
    """Pre-norm block: x + attention(norm(x)), then x + feed_forward(norm(x))."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        d = cfg.emb_dim
        self.norm1 = nn.LayerNorm(d)
        self.att = CausalSelfAttention(cfg)
        self.norm2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(approximate="tanh"), nn.Linear(4 * d, d))
        self.drop = nn.Dropout(cfg.drop_rate)

    def forward(self, x):
        x = x + self.drop(self.att(self.norm1(x)))
        return x + self.drop(self.ff(self.norm2(x)))


class GPTModel(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.grad_checkpointing = False
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.emb_dim)
        self.pos_emb = nn.Embedding(cfg.context_length, cfg.emb_dim)
        self.drop_emb = nn.Dropout(cfg.drop_rate)
        self.trf_blocks = nn.ModuleList(TransformerBlock(cfg) for _ in range(cfg.n_layers))
        self.final_norm = nn.LayerNorm(cfg.emb_dim)
        self.out_head = nn.Linear(cfg.emb_dim, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            self.out_head.weight = self.tok_emb.weight
        self._init_weights()

    def _init_weights(self):
        """GPT-2 initialisation; residual projections are scaled down by depth to keep activations stable."""
        for module in self.modules():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)
        for name, param in self.named_parameters():
            if name.endswith(("out_proj.weight", "ff.2.weight")):
                nn.init.normal_(param, std=0.02 / math.sqrt(2 * self.cfg.n_layers))

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())  # shared (tied) weights are counted once

    def forward(self, in_idx):
        seq_len = in_idx.shape[1]
        if seq_len > self.cfg.context_length:
            raise ValueError(f"Sequence length {seq_len} exceeds context_length {self.cfg.context_length}")
        x = self.tok_emb(in_idx) + self.pos_emb(torch.arange(seq_len, device=in_idx.device))
        x = self.drop_emb(x)
        for block in self.trf_blocks:
            if self.grad_checkpointing and self.training:
                x = checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)
        return self.out_head(self.final_norm(x))
