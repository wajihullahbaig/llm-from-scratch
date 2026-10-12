"""Training loop: mixed precision, gradient accumulation, warmup + cosine LR, evaluation, checkpoints, early stopping."""
import contextlib
import csv
import itertools
import math
import os
import time
from typing import Callable, Dict, List, Optional

import torch
import torch.nn.functional as F

from configs.config import TrainingConfig
from utils.checkpoint import save_checkpoint
from utils.plot import plot_dashboard


def lr_at(step: int, total_steps: int, warmup_steps: int, peak_lr: float, min_lr: float) -> float:
    """Linear warmup to peak_lr, then cosine decay to min_lr."""
    if step < warmup_steps:
        return peak_lr * (step + 1) / warmup_steps
    progress = min(1.0, (step - warmup_steps) / max(1, total_steps - warmup_steps))
    return min_lr + 0.5 * (peak_lr - min_lr) * (1 + math.cos(math.pi * progress))


def autocast_dtype(precision: str, device: torch.device) -> Optional[torch.dtype]:
    """bf16 on Ampere+ GPUs, fp16 on older GPUs, fp32 elsewhere (None means no autocast)."""
    if precision == "auto":
        if device.type != "cuda":
            return None
        return torch.bfloat16 if torch.cuda.get_device_capability(device)[0] >= 8 else torch.float16
    return {"fp32": None, "fp16": torch.float16, "bf16": torch.bfloat16}[precision]


def build_optimizer(model: torch.nn.Module, cfg: TrainingConfig, device: torch.device) -> torch.optim.AdamW:
    """AdamW with weight decay on weight matrices only (not on biases or LayerNorm parameters)."""
    params = [p for p in model.parameters() if p.requires_grad]
    groups = [{"params": [p for p in params if p.dim() >= 2], "weight_decay": cfg.weight_decay},
              {"params": [p for p in params if p.dim() < 2], "weight_decay": 0.0}]
    return torch.optim.AdamW(groups, lr=cfg.learning_rate, betas=(0.9, 0.95), fused=device.type == "cuda")


class Trainer:
    def __init__(self, model, train_loader, val_loader, cfg: TrainingConfig, device: torch.device,
                 out_dir: str, logger, sampler: Optional[Callable[[], str]] = None, seed: int = 42):
        self.model, self.train_loader, self.val_loader, self.seed = model, train_loader, val_loader, seed
        self.cfg, self.device, self.out_dir, self.logger, self.sampler = cfg, device, out_dir, logger, sampler
        self.optimizer = build_optimizer(model, cfg, device)
        self.dtype = autocast_dtype(cfg.precision, device)
        self.scaler = torch.amp.GradScaler("cuda", enabled=self.dtype == torch.float16)
        self.forward = torch.compile(model) if cfg.compile else model
        model.grad_checkpointing = cfg.grad_checkpointing
        self.total_steps = cfg.num_epochs * len(train_loader) // cfg.grad_accum_steps
        if self.total_steps == 0:
            raise ValueError("No optimizer steps: lower batch_size / grad_accum_steps or use more data.")
        self.warmup_steps = int(cfg.warmup_ratio * self.total_steps)
        self.history: List[Dict] = []
        self.step, self.tokens = 0, 0
        self.best_val, self.evals_without_improvement = math.inf, 0

    def state_dict(self) -> Dict:
        """Everything besides the model weights needed to continue training."""
        return {"optimizer": self.optimizer.state_dict(), "scaler": self.scaler.state_dict(), "tokens": self.tokens,
                "best_val": self.best_val, "evals_without_improvement": self.evals_without_improvement,
                "history": self.history}

    def resume(self, path: str) -> None:
        ckpt = torch.load(path, map_location=self.device, weights_only=True)
        if "train_state" not in ckpt:
            raise ValueError(f"{path} has no training state; resume from a last.pt checkpoint.")
        self.model.load_state_dict(ckpt["model"])
        state = ckpt["train_state"]
        self.optimizer.load_state_dict(state["optimizer"])
        self.scaler.load_state_dict(state["scaler"])
        self.step, self.tokens, self.history = ckpt["step"], state["tokens"], state["history"]
        self.best_val, self.evals_without_improvement = state["best_val"], state["evals_without_improvement"]

    def fit(self) -> List[Dict]:
        self.session_start, self.session_start_step = time.time(), self.step
        self.minutes_before = self.history[-1]["elapsed_min"] if self.history else 0.0  # from runs being resumed
        self._reset_window()
        self.model.train()
        try:
            self._run()
        except KeyboardInterrupt:
            self.logger.warning("Interrupted by user, saving the last checkpoint.")
        self._save_last()
        return self.history

    def _save_last(self) -> None:
        save_checkpoint(os.path.join(self.out_dir, "last.pt"), self.model, self.step, self.best_val, self.state_dict())

    def _run(self) -> None:
        accum = self.cfg.grad_accum_steps
        done = self.step * accum  # micro-batches already trained on (non-zero when resuming)
        for micro_step, (epoch, x, y) in enumerate(self._micro_batches(skip=done), start=done + 1):
            loss = self._loss(x, y) / accum
            self.scaler.scale(loss).backward()
            self.window_loss += loss.detach()
            self.tokens += x.numel()
            if micro_step % accum:
                continue
            self._optimizer_step()
            if self.step % self.cfg.eval_interval == 0 or self.step == self.total_steps:
                if self._evaluate_and_log(epoch):
                    self.logger.info(f"Early stopping: no improvement in {self.cfg.early_stopping_patience} evaluations.")
                    return
            if self.sampler and self.cfg.sample_interval and self.step % self.cfg.sample_interval == 0:
                self._log_sample()
            if self.cfg.save_interval and self.step % self.cfg.save_interval == 0:
                self._save_last()

    def _micro_batches(self, skip: int = 0):
        """Yield (epoch, x, y) after skipping `skip` micro-batches. Each epoch's shuffle is seeded, so a resumed
        run continues with exactly the batches the uninterrupted run would have seen."""
        epochs_done, skip_in_epoch = divmod(skip, len(self.train_loader))
        for epoch in range(epochs_done + 1, self.cfg.num_epochs + 1):
            self.train_loader.generator.manual_seed(self.seed + epoch)
            for x, y in itertools.islice(self.train_loader, skip_in_epoch, None):
                yield epoch, x, y
            skip_in_epoch = 0

    def _autocast(self):
        if self.dtype is None:
            return contextlib.nullcontext()
        return torch.autocast(self.device.type, dtype=self.dtype)

    def _loss(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device, non_blocking=True)
        y = y.to(self.device, non_blocking=True)
        with self._autocast():
            logits = self.forward(x)
            return F.cross_entropy(logits.flatten(0, 1), y.flatten())

    def _optimizer_step(self) -> None:
        self.lr = lr_at(self.step, self.total_steps, self.warmup_steps, self.cfg.learning_rate, self.cfg.min_lr)
        for group in self.optimizer.param_groups:
            group["lr"] = self.lr
        self.scaler.unscale_(self.optimizer)
        self.grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip or math.inf)
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.optimizer.zero_grad(set_to_none=True)
        self.step += 1
        self.window_steps += 1

    @torch.no_grad()
    def evaluate(self) -> float:
        """Mean loss over the first `eval_batches` validation batches (the same batches every time)."""
        self.model.eval()
        batches = itertools.islice(self.val_loader, self.cfg.eval_batches or None)
        losses = [self._loss(x, y) for x, y in batches]
        self.model.train()
        return torch.stack(losses).mean().item()

    def _evaluate_and_log(self, epoch: int) -> bool:
        """Evaluate, record metrics, keep the best checkpoint. Returns True when training should stop early."""
        train_time = time.time() - self.window_start
        record = self._record(epoch, self.evaluate(), train_time)
        self.history.append(record)
        self._append_csv("metrics.csv", record)
        plot_dashboard(self.history, self.total_steps, os.path.join(self.out_dir, "training.png"))
        self.logger.info(f"Epoch {epoch} step {self.step}/{self.total_steps} | train {record['train_loss']:.3f} | "
                         f"val {record['val_loss']:.3f} (ppl {record['val_ppl']:.1f}) | lr {self.lr:.2e} | "
                         f"grad norm {record['grad_norm']:.2f} | {record['tokens_per_sec']:,.0f} tok/s | "
                         f"elapsed {record['elapsed_min']:.1f} min, ETA {record['eta_min']:.1f} min")
        self._reset_window()
        return self._track_best(record["val_loss"])

    def _record(self, epoch: int, val_loss: float, train_time: float) -> Dict:
        session_minutes = (time.time() - self.session_start) / 60
        minutes_per_step = session_minutes / max(1, self.step - self.session_start_step)  # includes eval overhead
        return {"step": self.step, "epoch": epoch, "tokens": self.tokens,
                "train_loss": (self.window_loss / self.window_steps).item(), "val_loss": val_loss,
                "val_ppl": math.exp(min(val_loss, 20)), "lr": self.lr, "grad_norm": self.grad_norm.item(),
                "tokens_per_sec": (self.tokens - self.window_tokens) / train_time,
                "elapsed_min": self.minutes_before + session_minutes,
                "eta_min": minutes_per_step * (self.total_steps - self.step)}

    def _track_best(self, val_loss: float) -> bool:
        if val_loss < self.best_val:
            self.best_val, self.evals_without_improvement = val_loss, 0
            save_checkpoint(os.path.join(self.out_dir, "best.pt"), self.model, self.step, val_loss)
            return False
        self.evals_without_improvement += 1
        patience = self.cfg.early_stopping_patience
        return patience > 0 and self.evals_without_improvement >= patience

    def _reset_window(self) -> None:
        """Start a new measurement window for train loss and throughput (evaluation time is excluded)."""
        self.window_loss = torch.zeros((), device=self.device)
        self.window_steps, self.window_tokens, self.window_start = 0, self.tokens, time.time()

    def _append_csv(self, name: str, row: Dict) -> None:
        path = os.path.join(self.out_dir, name)
        is_new = not os.path.exists(path)
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(row))
            if is_new:
                writer.writeheader()
            writer.writerow(row)

    def _log_sample(self) -> None:
        start = time.time()
        self.model.eval()
        text = self.sampler()
        self.model.train()
        self.logger.info(f"Sample: {text}")
        self._append_csv("samples.csv", {"step": self.step, "tokens": self.tokens, "text": text})
        self.window_start += time.time() - start  # keep sampling time out of the throughput metric
