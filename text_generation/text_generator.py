"""Autoregressive decoding: greedy, temperature / top-k / top-p sampling, and beam search."""
from typing import Optional

import torch
import torch.nn.functional as F

from configs.config import GenerationConfig


def apply_repetition_penalty(logits: torch.Tensor, ids: torch.Tensor, penalty: float) -> torch.Tensor:
    """Make already-seen tokens less likely (CTRL-style: positive logits are divided, negative ones multiplied)."""
    if penalty == 1.0:
        return logits
    seen = logits.gather(1, ids)
    return logits.scatter(1, ids, torch.where(seen > 0, seen / penalty, seen * penalty))


def ban_repeated_ngrams(logits: torch.Tensor, ids: torch.Tensor, n: int) -> torch.Tensor:
    """Forbid any token that would complete an n-gram already present in the row's sequence."""
    if n <= 0:
        return logits
    for row, seq in enumerate(ids.tolist()):
        prefix = seq[len(seq) - n + 1:] if n > 1 else []
        banned = [seq[i + n - 1] for i in range(len(seq) - n + 1) if seq[i:i + n - 1] == prefix]
        logits[row, banned] = -float("inf")
    return logits


def filter_top_k_top_p(logits: torch.Tensor, top_k: Optional[int], top_p: Optional[float]) -> torch.Tensor:
    if top_k:
        kth_best = logits.topk(min(top_k, logits.shape[-1])).values[..., -1:]
        logits = logits.masked_fill(logits < kth_best, -float("inf"))
    if top_p and top_p < 1.0:
        sorted_logits, order = logits.sort(descending=True)
        probs = sorted_logits.softmax(-1)
        remove = probs.cumsum(-1) - probs > top_p  # keep the smallest set whose mass reaches top_p
        logits = logits.masked_fill(remove.scatter(-1, order, remove), -float("inf"))
    return logits


class TextGenerator:
    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate_text(self, cfg: GenerationConfig, prompt: Optional[str] = None) -> str:
        ids = self._encode(prompt or cfg.prompt)
        out = self._beam_search(ids, cfg) if cfg.num_beams > 1 else self._sample(ids, cfg)
        return self.tokenizer.decode(out.tolist())

    def _encode(self, text: str) -> torch.Tensor:
        device = next(self.model.parameters()).device
        return torch.tensor([self.tokenizer.encode(text, allowed_special={"<|endoftext|>"})], device=device)

    @torch.no_grad()
    def _next_logits(self, ids: torch.Tensor, cfg: GenerationConfig) -> torch.Tensor:
        context = ids[:, -self.model.cfg.context_length:]
        logits = self.model(context)[:, -1, :].float()
        logits = apply_repetition_penalty(logits, ids, cfg.repetition_penalty)
        return ban_repeated_ngrams(logits, ids, cfg.no_repeat_ngram_size)

    def _sample(self, ids: torch.Tensor, cfg: GenerationConfig) -> torch.Tensor:
        for _ in range(cfg.max_new_tokens):
            logits = self._next_logits(ids, cfg)
            if cfg.do_sample and cfg.temperature > 0:
                logits = filter_top_k_top_p(logits / cfg.temperature, cfg.top_k, cfg.top_p)
                next_id = torch.multinomial(logits.softmax(-1), num_samples=1)
            else:
                next_id = logits.argmax(-1, keepdim=True)
            ids = torch.cat([ids, next_id], dim=1)
            if next_id.item() == self.tokenizer.eot_token:
                break
        return ids[0]

    def _beam_search(self, ids: torch.Tensor, cfg: GenerationConfig) -> torch.Tensor:
        beams = ids.repeat(cfg.num_beams, 1)
        scores = torch.full((cfg.num_beams,), -float("inf"), device=ids.device)
        scores[0] = 0.0  # all beams start identical; expand only the first one
        for _ in range(cfg.max_new_tokens):
            log_probs = F.log_softmax(self._next_logits(beams, cfg), dim=-1)
            vocab = log_probs.shape[-1]
            scores, flat = (scores[:, None] + log_probs).view(-1).topk(cfg.num_beams)
            beams = torch.cat([beams[flat // vocab], (flat % vocab)[:, None]], dim=1)
        return beams[0]  # topk is sorted, so beam 0 has the best score
