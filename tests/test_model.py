import pytest
import torch

from configs.config import ModelConfig
from models.gpt2_model import GPTModel
from utils.checkpoint import load_model, save_checkpoint


def test_forward_shape(tiny_model, tiny_cfg):
    logits = tiny_model(torch.randint(0, tiny_cfg.vocab_size, (2, 10)))
    assert logits.shape == (2, 10, tiny_cfg.vocab_size)


def test_attention_is_causal(tiny_model):
    x = torch.randint(0, 1000, (1, 16))
    changed = x.clone()
    changed[0, 10:] = 7
    with torch.no_grad():
        assert torch.allclose(tiny_model(x)[0, :10], tiny_model(changed)[0, :10], atol=1e-5)


def test_rejects_too_long_input(tiny_model, tiny_cfg):
    with pytest.raises(ValueError):
        tiny_model(torch.zeros(1, tiny_cfg.context_length + 1, dtype=torch.long))


def test_weight_tying_shares_parameters(tiny_cfg):
    tied = GPTModel(tiny_cfg)
    untied = GPTModel(ModelConfig(**{**tiny_cfg.__dict__, "tie_weights": False}))
    assert tied.out_head.weight is tied.tok_emb.weight
    assert untied.num_params() - tied.num_params() == tiny_cfg.vocab_size * tiny_cfg.emb_dim


def test_grad_checkpointing_gives_same_loss(tiny_model):
    x = torch.randint(0, 1000, (2, 16))
    tiny_model.train()
    plain = tiny_model(x).sum()
    tiny_model.grad_checkpointing = True
    assert torch.allclose(plain, tiny_model(x).sum())


def test_checkpoint_round_trip(tiny_model, tmp_path):
    path = str(tmp_path / "model.pt")
    save_checkpoint(path, tiny_model, step=3, val_loss=1.0)
    x = torch.randint(0, 1000, (1, 8))
    with torch.no_grad():
        assert torch.equal(tiny_model(x), load_model(path)(x))


def test_legacy_checkpoint_loads(tiny_model, tmp_path):
    """A checkpoint in the original layout (separate Q/K/V, scale/shift norms, untied head) loads unchanged."""
    model = GPTModel(ModelConfig(**{**tiny_model.cfg.__dict__, "tie_weights": False, "qkv_bias": True})).eval()
    old = {}
    for key, value in model.state_dict().items():
        key = key.replace(".ff.", ".ff.layers.").replace("norm1.weight", "norm1.scale").replace("norm1.bias", "norm1.shift")
        key = key.replace("norm2.weight", "norm2.scale").replace("norm2.bias", "norm2.shift")
        key = key.replace("final_norm.weight", "final_norm.scale").replace("final_norm.bias", "final_norm.shift")
        if ".qkv." in key:
            prefix, kind = key.split(".qkv.")
            for name, part in zip(("query", "key", "value"), value.chunk(3)):
                old[f"{prefix}.W_{name}.{kind}"] = part
        else:
            old[key] = value
    legacy_config = {**model.cfg.__dict__, "model_type": "GPT_CONFIG_TINY"}
    del legacy_config["tie_weights"]
    path = str(tmp_path / "legacy.pt")
    torch.save({"model_state_dict": old, "optimizer_state_dict": {}, "config": legacy_config, "epochs": 1}, path)

    x = torch.randint(0, 1000, (1, 8))
    with torch.no_grad():
        assert torch.allclose(model(x), load_model(path)(x), atol=1e-6)
