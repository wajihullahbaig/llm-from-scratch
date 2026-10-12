"""Train a GPT model.

Usage: python train.py --config configs/gpt2_162m_config.yaml [--data alice|wikipedia]
Resume: python train.py --resume outputs/<run>/last.pt [--config new_settings.yaml]
"""
import argparse
import math
import os
from datetime import datetime

import tiktoken
import torch

from configs.config import Config, load_config, save_config
from data.dataset import build_loaders, load_tokens
from models.gpt2_model import GPTModel
from text_generation.text_generator import TextGenerator
from training.trainer import Trainer
from utils.misc import pick_device, set_seed, setup_logger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", help="Path to a YAML config (when resuming, defaults to the run's config.yaml)")
    parser.add_argument("--data", choices=["alice", "wikipedia"], help="Override data.source from the config")
    parser.add_argument("--resume", help="Continue training from a last.pt checkpoint, in the same run directory")
    args = parser.parse_args()
    if not args.config and not args.resume:
        parser.error("--config is required unless --resume is given")
    return args


def make_run_dir(cfg: Config, config_path: str) -> str:
    name = os.path.splitext(os.path.basename(config_path))[0]
    run_dir = os.path.join(cfg.run.output_dir, f"{name}_{cfg.data.source}_{datetime.now():%Y%m%d_%H%M%S}")
    os.makedirs(run_dir, exist_ok=True)
    return run_dir


def main() -> None:
    args = parse_args()
    resume_dir = os.path.dirname(os.path.abspath(args.resume)) if args.resume else None
    cfg = load_config(args.config or os.path.join(resume_dir, "config.yaml"))
    cfg.data.source = args.data or cfg.data.source
    run_dir = resume_dir or make_run_dir(cfg, args.config)
    save_config(cfg, os.path.join(run_dir, "config.yaml"))
    logger = setup_logger("gpt", os.path.join(run_dir, "train.log"))
    set_seed(cfg.run.seed)
    device = pick_device(cfg.run.device)
    torch.backends.cuda.matmul.allow_tf32 = True  # faster fp32 matmuls on Ampere+ GPUs
    logger.info(f"Run directory: {run_dir} | device: {device} | torch {torch.__version__}")

    tokenizer = tiktoken.get_encoding("gpt2")
    tokens = load_tokens(cfg.data, tokenizer)
    train_loader, val_loader = build_loaders(tokens, cfg, device)
    logger.info(f"Data: {cfg.data.source}, {len(tokens):,} tokens | {len(train_loader.dataset):,} train windows, "
                f"{len(val_loader.dataset):,} val windows of {cfg.model.context_length} tokens")

    model = GPTModel(cfg.model).to(device)
    logger.info(f"Model: {model.num_params() / 1e6:.1f}M parameters, {cfg.model.n_layers} layers, "
                f"{cfg.model.n_heads} heads, {cfg.model.emb_dim} dims, context {cfg.model.context_length}")

    generator = TextGenerator(model, tokenizer)
    trainer = Trainer(model, train_loader, val_loader, cfg.training, device, run_dir, logger,
                      sampler=lambda: generator.generate_text(cfg.generation), seed=cfg.run.seed)
    if args.resume:
        trainer.resume(args.resume)
        logger.info(f"Resumed from {args.resume} at step {trainer.step:,} (best val loss {trainer.best_val:.3f})")
    logger.info(f"Training for {trainer.total_steps:,} optimizer steps of "
                f"{cfg.training.batch_size * cfg.training.grad_accum_steps * cfg.model.context_length:,} tokens")
    trainer.fit()
    logger.info(f"Best val loss {trainer.best_val:.3f} (ppl {math.exp(min(trainer.best_val, 20)):.1f}). "
                f"Checkpoints, metrics.csv, samples.csv and training.png are in {run_dir}")
    logger.info(f"Final sample: {generator.generate_text(cfg.generation)}")


if __name__ == "__main__":
    main()
