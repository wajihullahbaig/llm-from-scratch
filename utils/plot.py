# Training dashboard, redrawn after every evaluation
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")  # render to file only; never block on a window
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import StrMethodFormatter  # noqa: E402

SURFACE, INK, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#e1e0d9", "#c3c2b7"
TRAIN, VAL = "#2a78d6", "#eb6834"  # categorical slots 1 and 2 (validated colour-blind safe)


def _style(ax, title: str, ylabel: str) -> None:
    ax.set_facecolor(SURFACE)
    ax.set_title(title, loc="left", fontsize=10, color=INK)
    ax.set_xlabel("Optimizer step", fontsize=8, color=MUTED)
    ax.set_ylabel(ylabel, fontsize=8, color=MUTED)
    ax.grid(color=GRID, linewidth=0.6)
    ax.tick_params(colors=MUTED, labelsize=8, length=0)
    for side, spine in ax.spines.items():
        spine.set_visible(side in ("left", "bottom"))
        spine.set_color(AXIS)


def _plot_loss(ax, history: List[Dict], steps: List[int]) -> None:
    ax.plot(steps, [r["train_loss"] for r in history], color=TRAIN, linewidth=2, label="Train")
    ax.plot(steps, [r["val_loss"] for r in history], color=VAL, linewidth=2, label="Validation")
    best = min(history, key=lambda r: r["val_loss"])
    ax.scatter([best["step"]], [best["val_loss"]], s=40, color=VAL, edgecolors=SURFACE, linewidths=2, zorder=3)
    ax.annotate(f"best {best['val_loss']:.3f}", (best["step"], best["val_loss"]), textcoords="offset points",
                xytext=(0, 8), ha="center", fontsize=8, color=INK)
    ax.legend(frameon=False, fontsize=8, labelcolor=INK)
    _style(ax, "Loss", "Cross-entropy")


def _plot_single(ax, steps: List[int], values: List[float], title: str, ylabel: str) -> None:
    ax.plot(steps, values, color=TRAIN, linewidth=2)
    _style(ax, title, ylabel)


def _plot_summary(ax, history: List[Dict], total_steps: int) -> None:
    """Stat tile: the headline numbers of the run so far."""
    last, best = history[-1], min(history, key=lambda r: r["val_loss"])
    mean_speed = sum(r["tokens_per_sec"] for r in history) / len(history)
    rows = [("Step", f"{last['step']:,} / {total_steps:,}"),
            ("Tokens seen", f"{last['tokens'] / 1e6:,.2f}M"),
            ("Best val loss", f"{best['val_loss']:.3f} (step {best['step']:,})"),
            ("Best val perplexity", f"{best['val_ppl']:,.1f}"),
            ("Throughput (now / mean)", f"{last['tokens_per_sec']:,.0f} / {mean_speed:,.0f} tok/s"),
            ("Elapsed", f"{last['elapsed_min']:.1f} min"),
            ("ETA", f"{last['eta_min']:.1f} min")]
    ax.axis("off")
    ax.set_title("Run summary", loc="left", fontsize=10, color=INK)
    for i, (label, value) in enumerate(rows):
        y = 0.92 - i * 0.14
        ax.text(0.0, y, label, fontsize=9, color=MUTED, transform=ax.transAxes)
        ax.text(1.0, y, value, fontsize=9, color=INK, ha="right", transform=ax.transAxes)


def plot_dashboard(history: List[Dict], total_steps: int, path: str) -> None:
    """Save a dashboard of loss, perplexity, throughput, learning rate, gradient norm and a run summary."""
    if not history:
        return
    steps = [r["step"] for r in history]
    fig, axes = plt.subplots(2, 3, figsize=(15, 7.5), facecolor=SURFACE)
    _plot_loss(axes[0, 0], history, steps)
    _plot_single(axes[0, 1], steps, [r["val_ppl"] for r in history], "Validation perplexity", "Perplexity")
    _plot_single(axes[0, 2], steps, [r["tokens_per_sec"] for r in history], "Training throughput", "Tokens / sec")
    _plot_single(axes[1, 0], steps, [r["lr"] for r in history], "Learning rate", "LR")
    _plot_single(axes[1, 1], steps, [r["grad_norm"] for r in history], "Gradient norm (before clipping)", "L2 norm")
    _plot_summary(axes[1, 2], history, total_steps)
    for ax in list(axes.flat)[:5]:
        ax.set_xlim(0, total_steps * 1.03)  # small margin so end-of-run markers are not clipped
    axes[0, 2].set_ylim(bottom=0)
    axes[0, 2].yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    axes[1, 0].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    fig.tight_layout()
    fig.savefig(path, dpi=110, facecolor=SURFACE)
    plt.close(fig)
