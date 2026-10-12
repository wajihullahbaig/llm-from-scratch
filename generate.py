"""Generate text with a trained checkpoint.

Usage: python generate.py --checkpoint outputs/<run>/best.pt [--config configs/x.yaml] [--prompt "Once upon a time"]
Decoding settings come from the config's `generation` section (defaults if no config is given).
"""
import argparse

import tiktoken

from configs.config import Config, load_config
from text_generation.text_generator import TextGenerator
from utils.checkpoint import load_model
from utils.misc import pick_device, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--checkpoint", required=True, help="Path to a .pt checkpoint")
    parser.add_argument("--config", help="YAML config whose `generation` section is used")
    parser.add_argument("--prompt", help="Override the configured prompt")
    parser.add_argument("--max-new-tokens", type=int, help="Override the configured number of new tokens")
    parser.add_argument("--device", help="cpu | mps | cuda (default: best available)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config) if args.config else Config()
    cfg.generation.max_new_tokens = args.max_new_tokens or cfg.generation.max_new_tokens
    set_seed(cfg.run.seed)
    model = load_model(args.checkpoint, pick_device(args.device or cfg.run.device))
    generator = TextGenerator(model, tiktoken.get_encoding("gpt2"))
    print(generator.generate_text(cfg.generation, prompt=args.prompt))


if __name__ == "__main__":
    main()
