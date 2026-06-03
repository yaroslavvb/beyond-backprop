"""
Self-contained learning-rate sensitivity plot for classic vs fixed backprop.

Task:
  Learn a 90-degree rotation of a 2D letter-A point cloud with three linear
  2D layers.

Methods:
  classic backprop:
    Compute all gradients from the original forward pass, then update all
    layers simultaneously.

  fixed backprop:
    Update top-to-bottom. After updating an upper layer, recompute only the
    local forward value needed to refresh the residual before propagating it to
    the next lower layer.

For every learning rate and method, the run starts from the exact same initial
weights W0. The figure shows loss after 1, 3, and 10 update steps.
"""

from __future__ import annotations

import os
import warnings

import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # pragma: no cover - used only if Pillow is unavailable.
    Image = ImageDraw = ImageFont = None


D = 2
DEPTH = 3
SEED = 89
STEP_COUNTS = (1, 3, 10)
LR_MIN = 0.0
LR_MAX = 20.0
NUM_LRS = 401
UNSTABLE_LOSS = 1e6
OUT_PATH = "classic_vs_fixed.png"


def simple_letter_a_points(n: int = 70) -> np.ndarray:
    """Fallback point cloud if Pillow is unavailable."""
    t = np.linspace(0.0, 1.0, n)
    left = np.column_stack((-0.42 + 0.42 * t, -0.5 + t))
    right = np.column_stack((0.42 - 0.42 * t, -0.5 + t))
    bar = np.column_stack((np.linspace(-0.24, 0.24, n), np.full(n, -0.05)))
    return np.vstack((left, right, bar))


def generate_letter_a(size: int = 38, img_size: int = 50) -> np.ndarray:
    """Rasterize an A, normalize to [-0.5, 0.5], then rotate clockwise."""
    if Image is None:
        return simple_letter_a_points()

    font = None
    for path in (
        "/System/Library/Fonts/Times.ttc",
        "/System/Library/Fonts/Supplemental/Times New Roman.ttf",
        "/Library/Fonts/Times New Roman.ttf",
    ):
        if os.path.exists(path):
            try:
                font = ImageFont.truetype(path, size)
                break
            except Exception:
                pass

    if font is None:
        try:
            fp = font_manager.findfont(font_manager.FontProperties(family="serif"))
            font = ImageFont.truetype(fp, size)
        except Exception:
            font = None

    image = Image.new("L", (img_size, img_size), 255)
    draw = ImageDraw.Draw(image)
    if font is not None:
        bbox = draw.textbbox((0, 0), "A", font=font)
        w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        draw.text(
            ((img_size - w) / 2 - bbox[0], (img_size - h) / 2 - bbox[1]),
            "A",
            fill=0,
            font=font,
        )
    else:
        draw.line([(15, 40), (25, 10)], fill=0, width=4)
        draw.line([(25, 10), (35, 40)], fill=0, width=4)
        draw.line([(18, 30), (32, 30)], fill=0, width=3)

    pixels = np.array(image)
    ys, xs = np.where(pixels < 128)
    X = np.column_stack((xs, img_size - ys)).astype(float)
    scaled = (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0)) - 0.5

    clockwise = np.array([[0.0, 1.0], [-1.0, 0.0]])
    return scaled @ clockwise.T


def orthogonalize(m: np.ndarray) -> np.ndarray:
    """Gram-Schmidt orthonormalization of matrix rows."""
    q = np.zeros_like(m, dtype=float)
    for i in range(m.shape[0]):
        v = m[i].astype(float).copy()
        for j in range(i):
            v -= np.dot(q[j], m[i]) * q[j]
        n = np.linalg.norm(v)
        q[i] = v / n if n > 1e-12 else v
    return q


def initial_weights() -> np.ndarray:
    rng = np.random.default_rng(SEED)
    return np.array([
        orthogonalize(np.eye(D) + 0.4 * orthogonalize(rng.normal(size=(D, D))))
        for _ in range(DEPTH)
    ])


def rotation_problem() -> tuple[np.ndarray, np.ndarray]:
    X = generate_letter_a()
    rotate_counterclockwise = np.array([[0.0, -1.0], [1.0, 0.0]])
    target = X @ rotate_counterclockwise.T
    return X, target


def forward(A: np.ndarray, W: np.ndarray) -> np.ndarray:
    return A @ W.T


def forward_pass(X: np.ndarray, Ws: np.ndarray) -> list[np.ndarray]:
    As = [X]
    for W in Ws:
        As.append(forward(As[-1], W))
    return As


def loss(Y: np.ndarray, target: np.ndarray) -> float:
    return 0.5 * float(np.mean(np.sum((Y - target) ** 2, axis=1)))


def step_classic(Ws: np.ndarray, X: np.ndarray, target: np.ndarray, lr: float) -> np.ndarray:
    """Classic backprop: gradients all use the stale pre-update weights."""
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
    """Fixed backprop with top-to-bottom local-forward residual refreshes."""
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


def loss_after_steps(
    step_fn,
    W0: np.ndarray,
    X: np.ndarray,
    target: np.ndarray,
    lr: float,
    steps: int,
) -> tuple[float, bool]:
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
) -> tuple[np.ndarray, np.ndarray]:
    losses = np.empty_like(lrs, dtype=float)
    unstable = np.zeros_like(lrs, dtype=bool)

    for i, lr in enumerate(lrs):
        losses[i], unstable[i] = loss_after_steps(step_fn, W0, X, target, float(lr), steps)

    return losses, unstable


def best_finite(lrs: np.ndarray, losses: np.ndarray) -> tuple[float, float]:
    idx = int(np.nanargmin(losses))
    return float(lrs[idx]), float(losses[idx])


def plot_sensitivity(
    lrs: np.ndarray,
    sweeps: dict[int, dict[str, tuple[np.ndarray, np.ndarray]]],
    out_path: str = OUT_PATH,
) -> str:
    fig, axes = plt.subplots(1, len(STEP_COUNTS), figsize=(16.2, 5.6), dpi=150)
    fig.patch.set_facecolor("white")

    styles = {
        "classic": ("classic backprop", "#c44e52"),
        "fixed": ("fixed backprop", "#2374ab"),
    }

    for ax, steps in zip(axes, STEP_COUNTS):
        finite_chunks = []
        for losses, _ in sweeps[steps].values():
            finite = losses[np.isfinite(losses)]
            if len(finite):
                finite_chunks.append(finite)
        finite_losses = np.concatenate(finite_chunks)
        y_min = max(float(np.min(finite_losses)) * 0.2, 1e-21)
        y_max = max(float(np.max(finite_losses)) * 5.0, 1.0)
        top_marker = y_max / 1.35

        for key, (label, color) in styles.items():
            losses, unstable = sweeps[steps][key]
            best_lr, best_loss = best_finite(lrs, losses)
            ax.plot(lrs, losses, lw=2.0, color=color, label=label)
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
        suffix = "s" if steps != 1 else ""
        ax.set_title(f"loss after {steps} step{suffix}", fontsize=12, fontweight="bold")
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


def main() -> None:
    X, target = rotation_problem()
    W0 = initial_weights()
    lrs = np.linspace(LR_MIN, LR_MAX, NUM_LRS)

    sweeps: dict[int, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    for steps in STEP_COUNTS:
        sweeps[steps] = {
            "classic": sweep_method(step_classic, W0, X, target, lrs, steps),
            "fixed": sweep_method(step_fixed, W0, X, target, lrs, steps),
        }

    print(f"steps per run: {STEP_COUNTS}")
    print(f"learning-rate range: [{LR_MIN}, {LR_MAX}] with {NUM_LRS} points")
    print(f"shared initial loss: {loss(forward_pass(X, W0)[-1], target):.6e}")
    for steps in STEP_COUNTS:
        suffix = "s" if steps != 1 else ""
        print(f"\nafter {steps} step{suffix}:")
        for key, label in (("classic", "classic backprop"), ("fixed", "fixed backprop")):
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
