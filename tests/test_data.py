import numpy as np
import pytest
import torch

from configs.config import Config, load_config
from data.corpus import clean_text
from data.dataset import TokenWindows, build_loaders, split_tokens, tokenize


def test_clean_text():
    assert clean_text("  Alice\r\nwas *very*\n\n tired ") == "Alice was very tired"


def test_tokenize_separates_documents(tokenizer):
    tokens = tokenize(["Hello world", "Second doc"], tokenizer)
    assert tokens.dtype == np.uint16
    assert list(tokens).count(tokenizer.eot_token) == 2
    assert tokenizer.decode(tokens.tolist()) == "Hello world<|endoftext|>Second doc<|endoftext|>"


@pytest.mark.parametrize("n_tokens,ctx,stride", [(100, 10, 10), (100, 10, 3), (11, 10, 5), (10, 10, 1)])
def test_windows_cover_all_valid_starts(n_tokens, ctx, stride):
    ds = TokenWindows(np.arange(n_tokens, dtype=np.uint16), ctx, stride)
    assert len(ds) == len(range(0, n_tokens - ctx, stride))
    if len(ds):
        x, y = ds[len(ds) - 1]
        assert x.shape == y.shape == (ctx,)
        assert torch.equal(x[1:], y[:-1]) and x.dtype == torch.long


def test_split_tokens():
    train, val = split_tokens(np.arange(100), 0.2)
    assert len(train) == 80 and len(val) == 20 and val[0] == 80


def test_build_loaders_rejects_tiny_corpus():
    cfg = Config()
    with pytest.raises(ValueError, match="Not enough tokens"):
        build_loaders(np.arange(50, dtype=np.uint16), cfg, torch.device("cpu"))


@pytest.mark.parametrize("name", ["gpt2_133m_config", "gpt2_162m_config", "gpt2_400m_config"])
def test_shipped_configs_load(name):
    cfg = load_config(f"configs/{name}.yaml")
    assert isinstance(cfg.training.learning_rate, float) and isinstance(cfg.training.min_lr, float)
    assert cfg.data.source in ("alice", "wikipedia")


def test_unknown_config_key_is_rejected(tmp_path):
    path = tmp_path / "bad.yaml"
    path.write_text("training:\n  batch_sise: 4\n")
    with pytest.raises(TypeError):
        load_config(str(path))
