"""Raw text corpora: Alice in Wonderland (Project Gutenberg) and English Wikipedia (Hugging Face)."""
import os
import re
import urllib.request
from typing import List

ALICE_URL = "https://www.gutenberg.org/cache/epub/11/pg11.txt"
ALICE_START = "Alice was beginning to get very tired of sitting by her sister on the"
ALICE_END = "*** END OF THE PROJECT GUTENBERG EBOOK ALICE'S ADVENTURES IN WONDERLAND ***"


def clean_text(text: str) -> str:
    """Collapse newlines, asterisks and repeated whitespace into single spaces."""
    return re.sub(r"[\s*]+", " ", text).strip()


def load_alice(cache_dir: str) -> List[str]:
    path = os.path.join(cache_dir, "alice-wonderland.txt")
    if not os.path.exists(path):
        os.makedirs(cache_dir, exist_ok=True)
        with urllib.request.urlopen(ALICE_URL) as response:
            raw = response.read().decode("utf-8")
        with open(path, "w", encoding="utf-8") as f:
            f.write(clean_text(raw[raw.find(ALICE_START):raw.find(ALICE_END)]))
    with open(path, encoding="utf-8") as f:
        return [f.read()]


def load_wikipedia(subset_ratio: float) -> List[str]:
    from datasets import load_dataset  # heavy import, only needed for Wikipedia

    ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train")
    n_articles = max(1, int(len(ds) * subset_ratio))
    return [clean_text(text) for text in ds.select(range(n_articles))["text"]]


def load_corpus(source: str, subset_ratio: float, cache_dir: str) -> List[str]:
    """Return the corpus as a list of documents."""
    if source == "alice":
        return load_alice(cache_dir)
    if source == "wikipedia":
        return load_wikipedia(subset_ratio)
    raise ValueError(f"Unknown data source: {source!r} (expected 'alice' or 'wikipedia')")
