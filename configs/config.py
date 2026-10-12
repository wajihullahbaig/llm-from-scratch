"""Typed run configuration loaded from YAML. Unknown keys raise an error instead of being silently ignored."""
from dataclasses import asdict, dataclass, field, fields
from typing import Optional

import yaml


@dataclass
class RunConfig:
    output_dir: str = "outputs"
    device: Optional[str] = None  # cpu | mps | cuda; None picks the best available
    seed: int = 42


@dataclass
class DataConfig:
    source: str = "alice"  # alice | wikipedia
    subset_ratio: float = 1.0  # fraction of Wikipedia articles to use
    val_ratio: float = 0.1
    stride: Optional[int] = None  # None = context_length (no overlapping training windows)
    cache_dir: str = "corpuses"


@dataclass
class TrainingConfig:
    num_epochs: int = 1
    batch_size: int = 8
    grad_accum_steps: int = 1  # effective batch = batch_size * grad_accum_steps
    learning_rate: float = 6.0e-4  # peak learning rate
    min_lr: float = 6.0e-5
    warmup_ratio: float = 0.05
    weight_decay: float = 0.1
    grad_clip: float = 1.0  # 0 disables clipping
    precision: str = "auto"  # auto | fp32 | fp16 | bf16
    compile: bool = False  # torch.compile (needs Triton, i.e. Linux + CUDA)
    grad_checkpointing: bool = False  # trade compute for memory on long contexts
    eval_interval: int = 100  # optimizer steps between evaluations
    eval_batches: int = 20  # validation batches per evaluation (0 = full set)
    early_stopping_patience: int = 0  # stop after this many evaluations without val improvement (0 = off)
    sample_interval: int = 500  # optimizer steps between text samples (0 = off)
    save_interval: int = 500  # optimizer steps between resumable last.pt saves (0 = only at the end)
    num_workers: int = 0


@dataclass
class GenerationConfig:
    prompt: str = "Water is essential for all forms of life because"
    max_new_tokens: int = 50
    temperature: float = 1.0
    top_k: Optional[int] = 50
    top_p: Optional[float] = None
    repetition_penalty: float = 1.0
    no_repeat_ngram_size: int = 0
    num_beams: int = 1  # > 1 enables beam search (ignores sampling settings)
    do_sample: bool = True  # False = greedy decoding


@dataclass
class ModelConfig:
    vocab_size: int = 50257
    context_length: int = 1024
    emb_dim: int = 768
    n_heads: int = 12
    n_layers: int = 12
    drop_rate: float = 0.1
    qkv_bias: bool = False
    tie_weights: bool = True  # share token embedding and output head weights, as GPT-2 does

    def __post_init__(self):
        if self.emb_dim % self.n_heads:
            raise ValueError("emb_dim must be divisible by n_heads")


@dataclass
class Config:
    run: RunConfig = field(default_factory=RunConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    generation: GenerationConfig = field(default_factory=GenerationConfig)
    model: ModelConfig = field(default_factory=ModelConfig)


def load_config(path: str) -> Config:
    with open(path, encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    sections = {f.name: f.default_factory for f in fields(Config)}
    unknown = set(raw) - set(sections)
    if unknown:
        raise ValueError(f"Unknown config sections in {path}: {sorted(unknown)}")
    return Config(**{name: cls(**(raw.get(name) or {})) for name, cls in sections.items()})


def save_config(cfg: Config, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(asdict(cfg), f, sort_keys=False)
