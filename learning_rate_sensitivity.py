"""
Learning-rate sensitivity for stale backprop vs local-forward fixed backprop.

For each learning rate, both methods start from the same initial weights and the
same rotated-letter-A dataset.  The plot reports the loss after exactly 1, 3,
and 10 training updates.
"""

from __future__ import annotations

import os
import warnings

import matplotlib.pyplot as plt
import numpy as np

from rotation_backprop_comparison import (
    forward_pass,
    initial_weights,
    loss,
    rotation_problem,
    step_fixed,
    step_stale,
)


STEP_COUNTS = (1, 3, 10)
LR_MIN = 0.0
LR_MAX = 20.0
NUM_LRS = 401
UNSTABLE_LOSS = 1e6


def loss_after_steps(
    step_fn,
    W0: np.ndarray,
    X: np.ndarray,
    target: np.ndarray,
    lr: float,
    steps: int,
):
    Ws = W0.copy()

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with np.errstate(all="ignore"):
            for _ in range(steps):
                Ws = step_fn(Ws, X, target, lr)
                if not np.all(np.isfinite(Ws)):
                    return np.nan, True

    Y = forward_pass(X, Ws)[-1]
    if not np.all(np.isfinite(Y)):
        return np.nan, True

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with np.errstate(all="ignore"):
            final_loss = loss(Y, target)
    if not np.isfinite(final_loss) or final_loss > UNSTABLE_LOSS:
        return np.nan, True

    return final_loss, False


def sweep_method(
    step_fn,
    W0: np.ndarray,
    X: np.ndarray,
    target: np.ndarray,
    lrs: np.ndarray,
    steps: int,
):
    losses = np.empty_like(lrs, dtype=float)
    unstable = np.zeros_like(lrs, dtype=bool)

    for i, lr in enumerate(lrs):
        losses[i], unstable[i] = loss_after_steps(step_fn, W0, X, target, float(lr), steps)

    return losses, unstable


def best_finite(lrs: np.ndarray, losses: np.ndarray):
    idx = int(np.nanargmin(losses))
    return float(lrs[idx]), float(losses[idx])


def plot_sensitivity(
    lrs: np.ndarray,
    sweeps: dict[int, dict[str, tuple[np.ndarray, np.ndarray]]],
    out_path: str = "learning_rate_sensitivity.png",
):
    fig, axes = plt.subplots(1, len(STEP_COUNTS), figsize=(16.2, 5.6), dpi=150)
    fig.patch.set_facecolor("white")

    styles = {
        "stale": ("classic backprop", "#c44e52"),
        "fixed": ("fixed backprop", "#2374ab"),
    }

    for ax, steps in zip(axes, STEP_COUNTS):
        finite_losses = []
        for method_results in sweeps[steps].values():
            losses, _ = method_results
            finite_losses.append(losses[np.isfinite(losses)])
        finite_losses = np.concatenate([v for v in finite_losses if len(v)])
        y_min = max(float(np.min(finite_losses)) * 0.2, 1e-21)
        y_max = max(float(np.max(finite_losses)) * 5.0, 1.0)
        top_marker = y_max / 1.35

        for key, (label, color) in styles.items():
            losses, unstable = sweeps[steps][key]
            best_lr, best_loss = best_finite(lrs, losses)
            ax.plot(
                lrs,
                losses,
                lw=2.0,
                color=color,
                label=label,
            )
            ax.scatter([best_lr], [best_loss], color=color, s=42, zorder=4)
            if np.any(unstable):
                ax.scatter(
                    lrs[unstable],
                    np.full(np.count_nonzero(unstable), top_marker),
                    marker="x",
                    color=color,
                    s=12,
                    alpha=0.35,
                )

        ax.set_yscale("log")
        ax.set_xlim(LR_MIN, LR_MAX)
        ax.set_ylim(y_min, y_max)
        ax.set_xlabel("learning rate")
        ax.set_title(f"loss after {steps} step{'s' if steps != 1 else ''}", fontsize=12, fontweight="bold")
        ax.grid(True, which="both", color="#dddddd", lw=0.75)
        ax.legend(loc="lower left", frameon=True, facecolor="white", framealpha=0.95, fontsize=8.5)

    axes[0].set_ylabel("loss")
    fig.suptitle("Learning-rate sensitivity on the rotation task", y=1.02, fontsize=15, fontweight="bold")
    fig.text(
        0.995,
        0.98,
        "x at top = nonfinite or loss > 1e6",
        ha="right",
        va="top",
        fontsize=9,
        color="#444444",
        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#cccccc"),
    )
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return os.path.abspath(out_path)


def main():
    X, target = rotation_problem()
    W0 = initial_weights()
    lrs = np.linspace(LR_MIN, LR_MAX, NUM_LRS)

    sweeps: dict[int, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    for steps in STEP_COUNTS:
        sweeps[steps] = {
            "stale": sweep_method(step_stale, W0, X, target, lrs, steps),
            "fixed": sweep_method(step_fixed, W0, X, target, lrs, steps),
        }

    print(f"steps per run: {STEP_COUNTS}")
    print(f"learning-rate range: [{LR_MIN}, {LR_MAX}] with {NUM_LRS} points")
    print(f"shared initial loss: {loss(forward_pass(X, W0)[-1], target):.6e}")
    for steps in STEP_COUNTS:
        print(f"\nafter {steps} step{'s' if steps != 1 else ''}:")
        for key, label in (("stale", "stale"), ("fixed", "fixed")):
            losses, unstable = sweeps[steps][key]
            best_lr, best_loss = best_finite(lrs, losses)
            print(
                f"  {label} best lr: {best_lr:.3f}, "
                f"loss: {best_loss:.6e}, unstable/capped points: {np.count_nonzero(unstable)}"
            )

    out_path = plot_sensitivity(lrs, sweeps)
    print(f"Saved figure to {out_path}")


if __name__ == "__main__":
    main()
