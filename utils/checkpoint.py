"""Save and load model checkpoints, including those written by the original version of this project."""
import os
from dataclasses import asdict, fields
from typing import Optional

import torch

from configs.config import ModelConfig
from models.gpt2_model import GPTModel


def save_checkpoint(path: str, model: GPTModel, step: int, val_loss: float, train_state: Optional[dict] = None) -> None:
    """Save weights and model config; `train_state` (optimizer etc.) is included when training should be resumable."""
    ckpt = {"model": model.state_dict(), "config": asdict(model.cfg), "step": step, "val_loss": val_loss}
    if train_state is not None:
        ckpt["train_state"] = train_state
    torch.save(ckpt, path + ".tmp")
    os.replace(path + ".tmp", path)  # a crash mid-save never corrupts the previous checkpoint


def load_model(path: str, device="cpu") -> GPTModel:
    ckpt = torch.load(path, map_location=device, weights_only=True)
    if "model_state_dict" in ckpt:
        ckpt = convert_legacy_checkpoint(ckpt)
    model = GPTModel(ModelConfig(**ckpt["config"]))
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval()


def convert_legacy_checkpoint(ckpt: dict) -> dict:
    """Map the old layout (separate W_query/W_key/W_value, custom LayerNorm scale/shift) to the current one."""
    known = {f.name for f in fields(ModelConfig)}
    config = {k: v for k, v in ckpt["config"].items() if k in known}
    config["tie_weights"] = False  # the old model had an untied output head
    old = ckpt["model_state_dict"]

    new = {}
    for key, value in old.items():
        if key.endswith(".mask") or ".W_" in key:
            continue
        new[key.replace(".ff.layers.", ".ff.").replace(".scale", ".weight").replace(".shift", ".bias")] = value
    for prefix in {k.split(".W_")[0] for k in old if ".W_" in k}:
        for kind in ("weight", "bias"):
            if f"{prefix}.W_query.{kind}" in old:
                parts = [old[f"{prefix}.W_{name}.{kind}"] for name in ("query", "key", "value")]
                new[f"{prefix}.qkv.{kind}"] = torch.cat(parts)
    return {"model": new, "config": config}
