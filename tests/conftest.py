import pytest
import tiktoken
import torch

from configs.config import ModelConfig
from models.gpt2_model import GPTModel


@pytest.fixture
def tiny_cfg():
    return ModelConfig(vocab_size=50257, context_length=32, emb_dim=32, n_heads=4, n_layers=2, drop_rate=0.0)


@pytest.fixture
def tiny_model(tiny_cfg):
    torch.manual_seed(0)
    return GPTModel(tiny_cfg).eval()


@pytest.fixture(scope="session")
def tokenizer():
    return tiktoken.get_encoding("gpt2")
