"""
Rotation-learning comparison for stale backprop vs locally corrected backprop.

This uses the same letter-A point cloud as beyond_backprop.py, but the network is
three simple linear 2D layers:

    A2 = A1 @ W1.T
    A3 = A2 @ W2.T
    Y  = A3 @ W3.T

The left implementation computes all backward values from the old forward pass
and applies all weight updates simultaneously.  The right implementation updates
top-to-bottom and recomputes only the local forward value needed to refresh the
residual before propagating it lower:

    B3_new = B3 + (forward(A3, W3_new) - Y)
    B2_new = B2 + (forward(A2, W2_new) - A3)
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np

from beyond_backprop import D, generate_letter_a, orthogonalize


DEPTH = 3
SEED = 89
STEPS = 20
FIXED_LR = 10.0
SNAPSHOT_STEPS = (0, 1, 2, 3, 5, 10, 20)
STALE_LR_CANDIDATES = np.linspace(0.5, 7.0, 262)


def forward(A: np.ndarray, W: np.ndarray) -> np.ndarray:
    return A @ W.T


def loss(Y: np.ndarray, target: np.ndarray) -> float:
    return 0.5 * float(np.mean(np.sum((Y - target) ** 2, axis=1)))


def forward_pass(X: np.ndarray, Ws: np.ndarray) -> list[np.ndarray]:
    As = [X]
    for W in Ws:
        As.append(forward(As[-1], W))
    return As


def initial_weights() -> np.ndarray:
    rng = np.random.default_rng(SEED)
    return np.array([
        orthogonalize(np.eye(D) + 0.4 * orthogonalize(rng.normal(size=(D, D))))
        for _ in range(DEPTH)
    ])


def rotation_problem() -> tuple[np.ndarray, np.ndarray]:
    X = generate_letter_a()
    target_mat = np.array([[0.0, -1.0], [1.0, 0.0]])
    target = X @ target_mat.T
    return X, target


def step_stale(Ws: np.ndarray, X: np.ndarray, target: np.ndarray, lr: float) -> np.ndarray:
    """Traditional backprop: compute every gradient from stale weights."""
    As = forward_pass(X, Ws)
    batch_size = X.shape[0]

    Bs = [None] * (DEPTH + 1)
    Bs[DEPTH] = As[-1] - target
    Gs = [None] * DEPTH

    for i in range(DEPTH - 1, -1, -1):
        Gs[i] = Bs[i + 1].T @ As[i] / batch_size
        Bs[i] = Bs[i + 1] @ Ws[i]

    return Ws - lr * np.array(Gs)


def step_fixed(Ws: np.ndarray, X: np.ndarray, target: np.ndarray, lr: float) -> np.ndarray:
    """Top-to-bottom update with local-forward residual refreshes."""
    A1, A2, A3, Y = forward_pass(X, Ws)
    batch_size = X.shape[0]

    B3 = Y - target
    G3 = B3.T @ A3 / batch_size
    W3_new = Ws[2] - lr * G3

    B3_new = B3 + (forward(A3, W3_new) - Y)
    B2 = B3_new @ W3_new
    G2 = B2.T @ A2 / batch_size
    W2_new = Ws[1] - lr * G2

    B2_new = B2 + (forward(A2, W2_new) - A3)
    B1 = B2_new @ W2_new
    G1 = B1.T @ A1 / batch_size
    W1_new = Ws[0] - lr * G1

    return np.array([W1_new, W2_new, W3_new])


@dataclass
class History:
    name: str
    lr: float
    losses: list[float]
    snapshots: dict[int, np.ndarray]


def train(name: str, step_fn, X: np.ndarray, target: np.ndarray, lr: float) -> History:
    Ws = initial_weights()
    losses: list[float] = []
    snapshots: dict[int, np.ndarray] = {}

    for step in range(STEPS + 1):
        Y = forward_pass(X, Ws)[-1]
        losses.append(loss(Y, target))
        if step in SNAPSHOT_STEPS:
            snapshots[step] = Y.copy()
        if step < STEPS:
            Ws = step_fn(Ws, X, target, lr)

    return History(name, lr, losses, snapshots)


def tune_lr(step_fn, X: np.ndarray, target: np.ndarray, candidates: np.ndarray) -> float:
    best_lr = float(candidates[0])
    best_loss = float("inf")

    for lr in candidates:
        history = train("candidate", step_fn, X, target, float(lr))
        final_loss = history.losses[-1]
        if np.isfinite(final_loss) and final_loss < best_loss:
            best_lr = float(lr)
            best_loss = final_loss

    return best_lr


def plot_histories(
    stale: History,
    fixed: History,
    target: np.ndarray,
    out_path: str = "rotation_backprop_comparison.png",
) -> str:
    methods = [stale, fixed]
    n_rows = len(SNAPSHOT_STEPS) + 1
    fig, axs = plt.subplots(
        n_rows,
        2,
        figsize=(10.5, 12.4),
        dpi=150,
        gridspec_kw={"height_ratios": [1.25] + [1.0] * len(SNAPSHOT_STEPS)},
    )
    fig.patch.set_facecolor("white")

    all_points = [target]
    for history in methods:
        all_points.extend(history.snapshots.values())
    stacked = np.vstack(all_points)
    lo = float(np.min(stacked)) - 0.08
    hi = float(np.max(stacked)) + 0.08

    colors = ["#c44e52", "#2374ab"]
    for col, (history, color) in enumerate(zip(methods, colors)):
        ax = axs[0, col]
        ax.plot(history.losses, color=color, lw=2.0)
        ax.scatter(range(len(history.losses)), history.losses, color=color, s=10)
        ax.set_yscale("log")
        ax.grid(True, color="#dddddd", lw=0.7)
        ax.set_title(f"{history.name} (lr={history.lr:.3g})", fontsize=12, fontweight="bold")
        ax.set_xlabel("training step")
        ax.set_ylabel("loss")
        ax.text(
            0.98,
            0.93,
            f"final loss={history.losses[-1]:.2e}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=9,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#cccccc"),
        )

        for row, step in enumerate(SNAPSHOT_STEPS, start=1):
            ax = axs[row, col]
            ax.scatter(
                target[:, 0],
                target[:, 1],
                s=12,
                color="#bbbbbb",
                alpha=0.7,
                edgecolors="none",
                label="target",
            )
            ax.scatter(
                history.snapshots[step][:, 0],
                history.snapshots[step][:, 1],
                s=12,
                color=color,
                alpha=0.9,
                edgecolors="none",
                label="output",
            )
            ax.set_xlim(lo, hi)
            ax.set_ylim(lo, hi)
            ax.set_aspect("equal")
            ax.axis("off")
            ax.text(
                0.02,
                0.94,
                f"step {step}\nloss={history.losses[step]:.2e}",
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=8.5,
                bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="none", alpha=0.82),
            )

    fig.suptitle(
        "Rotation learning with 3 linear 2D layers",
        y=0.995,
        fontsize=15,
        fontweight="bold",
    )
    fig.text(
        0.5,
        0.973,
        "gray = target rotated A; colored = network output; shared initialization; stale LR tuned on this run",
        ha="center",
        va="top",
        fontsize=10,
        color="#444444",
    )
    fig.tight_layout(rect=(0.02, 0.01, 0.98, 0.955))
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return os.path.abspath(out_path)


def main():
    X, target = rotation_problem()
    stale_lr = tune_lr(step_stale, X, target, STALE_LR_CANDIDATES)
    stale = train("stale/simultaneous backprop", step_stale, X, target, stale_lr)
    fixed = train("local-forward fixed backprop", step_fixed, X, target, FIXED_LR)

    print(f"stale tuned learning rate: {stale.lr:.6f}")
    print(f"fixed learning rate: {fixed.lr:.6f}")
    print(f"steps: {STEPS}")
    print(f"stale final loss: {stale.losses[-1]:.6e}")
    print(f"fixed final loss: {fixed.losses[-1]:.6e}")

    assert np.isfinite(stale.losses[-1])
    assert fixed.losses[-1] < 1e-12

    out_path = plot_histories(stale, fixed, target)
    print(f"Saved figure to {out_path}")


if __name__ == "__main__":
    main()
