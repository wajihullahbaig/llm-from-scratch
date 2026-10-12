import csv
import logging
import math

import numpy as np
import pytest
import torch

from configs.config import Config, TrainingConfig
from data.dataset import build_loaders
from models.gpt2_model import GPTModel
from training.trainer import Trainer, build_optimizer, lr_at
from utils.checkpoint import load_model


def test_lr_schedule_warms_up_then_decays_to_min():
    lrs = [lr_at(s, 100, 10, 1.0, 0.1) for s in range(100)]
    assert lrs[0] == pytest.approx(0.1) and lrs[9] == pytest.approx(1.0)
    assert all(a >= b for a, b in zip(lrs[9:], lrs[10:]))
    assert lr_at(100, 100, 10, 1.0, 0.1) == pytest.approx(0.1)
    assert lr_at(0, 5, 0, 1.0, 0.1) == pytest.approx(1.0)  # no warmup does not divide by zero


def test_no_weight_decay_on_biases_and_norms(tiny_model):
    decay, no_decay = build_optimizer(tiny_model, TrainingConfig(), torch.device("cpu")).param_groups
    assert all(p.dim() >= 2 for p in decay["params"]) and all(p.dim() < 2 for p in no_decay["params"])
    assert no_decay["weight_decay"] == 0.0


def make_trainer(tiny_cfg, tmp_path, **training):
    tmp_path.mkdir(parents=True, exist_ok=True)
    cfg = Config(model=tiny_cfg, training=TrainingConfig(batch_size=4, **training))
    tokens = np.tile(np.arange(64, dtype=np.uint16), 40)  # a short repeating pattern the model can learn
    train_loader, val_loader = build_loaders(tokens, cfg, torch.device("cpu"))
    torch.manual_seed(0)
    model = GPTModel(tiny_cfg)
    return Trainer(model, train_loader, val_loader, cfg.training, torch.device("cpu"), str(tmp_path),
                   logging.getLogger("test"), sampler=lambda: "sample")


def test_training_reduces_loss_and_writes_outputs(tiny_cfg, tmp_path):
    trainer = make_trainer(tiny_cfg, tmp_path, num_epochs=8, learning_rate=3.0e-3, eval_interval=10,
                           sample_interval=20, grad_accum_steps=2)
    history = trainer.fit()
    assert history[-1]["val_loss"] < history[0]["val_loss"] < math.log(tiny_cfg.vocab_size) + 1
    assert history[-1]["step"] == trainer.total_steps
    with open(tmp_path / "metrics.csv") as f:
        assert len(list(csv.DictReader(f))) == len(history)
    assert load_model(str(tmp_path / "best.pt")).cfg == tiny_cfg
    assert (tmp_path / "last.pt").exists() and (tmp_path / "training.png").exists()
    with open(tmp_path / "samples.csv", encoding="utf-8") as f:
        assert [row["text"] for row in csv.DictReader(f)] == ["sample"] * (trainer.total_steps // 20)
    assert history[-1]["eta_min"] == 0 and history[-1]["elapsed_min"] > 0


def test_resume_matches_uninterrupted_run(tiny_cfg, tmp_path):
    settings = dict(num_epochs=4, learning_rate=3.0e-3, eval_interval=10, grad_accum_steps=2)
    full = make_trainer(tiny_cfg, tmp_path / "full", **settings).fit()

    interrupted = make_trainer(tiny_cfg, tmp_path / "run", **settings)
    step_once = interrupted._optimizer_step

    def step_then_interrupt():  # simulate Ctrl+C mid-epoch, right after step 13
        step_once()
        if interrupted.step == 13:
            raise KeyboardInterrupt

    interrupted._optimizer_step = step_then_interrupt
    interrupted.fit()

    resumed = make_trainer(tiny_cfg, tmp_path / "run", **settings)
    resumed.resume(str(tmp_path / "run" / "last.pt"))
    assert resumed.step == 13
    history = resumed.fit()
    assert [r["step"] for r in history] == [r["step"] for r in full]
    assert history[-1]["val_loss"] == pytest.approx(full[-1]["val_loss"], rel=1e-4)


def test_best_checkpoint_cannot_be_resumed(tiny_cfg, tmp_path):
    trainer = make_trainer(tiny_cfg, tmp_path, num_epochs=1, eval_interval=5)
    trainer.fit()
    with pytest.raises(ValueError, match="no training state"):
        make_trainer(tiny_cfg, tmp_path).resume(str(tmp_path / "best.pt"))


def test_early_stopping(tiny_cfg, tmp_path):
    # learning_rate 0 means validation loss never improves after the first evaluation
    trainer = make_trainer(tiny_cfg, tmp_path, num_epochs=50, learning_rate=0.0, min_lr=0.0,
                           eval_interval=2, early_stopping_patience=2)
    history = trainer.fit()
    assert len(history) == 3 and trainer.step < trainer.total_steps
