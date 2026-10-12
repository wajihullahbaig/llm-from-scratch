# Miscellaneous helper functions
import logging
import random
from typing import Optional

import numpy as np
import torch


def setup_logger(name: str, log_file: Optional[str] = None) -> logging.Logger:
    """Log to the console and, optionally, to a file."""
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    handlers = [logging.StreamHandler()]
    if log_file:
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    for handler in handlers:
        handler.setFormatter(logging.Formatter("%(asctime)s - %(levelname)s - %(message)s"))
        logger.addHandler(handler)
    return logger


def pick_device(name: Optional[str] = None) -> torch.device:
    """Return the requested device if available, otherwise the best one (cuda > mps > cpu)."""
    available = {"cuda": torch.cuda.is_available(), "mps": torch.backends.mps.is_available(), "cpu": True}
    if name and available.get(name):
        return torch.device(name)
    if name:
        logging.getLogger(__name__).warning("Device %r not available, picking the best available one", name)
    return torch.device(next(device for device, ok in available.items() if ok))


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # also seeds all CUDA devices
