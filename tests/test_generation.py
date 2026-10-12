import math

import torch

from configs.config import GenerationConfig
from text_generation.text_generator import (TextGenerator, apply_repetition_penalty, ban_repeated_ngrams,
                                            filter_top_k_top_p)


def test_repetition_penalty_lowers_seen_tokens_of_either_sign():
    logits = torch.tensor([[2.0, -2.0, 1.0]])
    out = apply_repetition_penalty(logits, torch.tensor([[0, 1]]), 2.0)
    assert out.tolist() == [[1.0, -4.0, 1.0]]


def test_ngram_ban_blocks_only_completions_of_current_prefix():
    # sequence "5 6 7 5 6": with n=3, the prefix (5, 6) was followed by 7 before, so only 7 is banned
    out = ban_repeated_ngrams(torch.zeros(1, 10), torch.tensor([[5, 6, 7, 5, 6]]), 3)
    assert torch.isinf(out[0]).nonzero().flatten().tolist() == [7]


def test_ngram_ban_is_per_row():
    out = ban_repeated_ngrams(torch.zeros(2, 10), torch.tensor([[1, 2, 1], [3, 4, 5]]), 2)
    assert math.isinf(out[0, 2]) and not torch.isinf(out[1]).any()


def test_top_k_and_top_p():
    logits = torch.log(torch.tensor([[0.5, 0.3, 0.15, 0.05]]))
    assert torch.isinf(filter_top_k_top_p(logits.clone(), 2, None)[0]).tolist() == [False, False, True, True]
    assert torch.isinf(filter_top_k_top_p(logits.clone(), None, 0.7)[0]).tolist() == [False, False, True, True]
    assert not torch.isinf(filter_top_k_top_p(logits.clone(), None, None)).any()


def test_greedy_is_deterministic_and_respects_length(tiny_model, tokenizer):
    gen = TextGenerator(tiny_model, tokenizer)
    cfg = GenerationConfig(prompt="Hello", max_new_tokens=5, do_sample=False)
    ids = gen._sample(gen._encode(cfg.prompt), cfg)
    assert len(ids) <= 1 + 5
    assert gen.generate_text(cfg) == gen.generate_text(cfg)


def test_prompt_longer_than_context_is_cropped(tiny_model, tokenizer):
    gen = TextGenerator(tiny_model, tokenizer)
    text = gen.generate_text(GenerationConfig(prompt="word " * 100, max_new_tokens=3))
    assert text.startswith("word ")


def test_beam_search_returns_full_length(tiny_model, tokenizer):
    gen = TextGenerator(tiny_model, tokenizer)
    cfg = GenerationConfig(prompt="Hello", max_new_tokens=4, num_beams=3, no_repeat_ngram_size=2)
    ids = gen._beam_search(gen._encode(cfg.prompt), cfg)
    assert len(ids) == 1 + 4
