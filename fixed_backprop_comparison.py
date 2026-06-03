"""
Compare stale/simultaneous backprop updates with an interleaved "fixed" update.

All layers are linear and use row-batch convention:

    forward(A, W)  = A @ W.T
    backward(B, W) = B @ W
    grad(B, A)     = B.T @ A

The example has a batch of two 2-feature examples and three 2x2 layers.  W3 is
crafted so that one unit learning-rate update makes the local output
forward(A3, W3_new) exactly equal to Yhat.  The fixed method then recomputes
only the needed local forward values before propagating each lower signal:

    B3_new = B3 + (forward(A3, W3_new) - Y0)
    B2_new = B2 + (forward(A2, W2_new) - A3)

In this crafted case those corrected residuals make the W2 and W1 updates exact
no-ops.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np


LR = 1.0


def forward(A: np.ndarray, W: np.ndarray) -> np.ndarray:
    return A @ W.T


def backward(B: np.ndarray, W: np.ndarray) -> np.ndarray:
    return B @ W


def grad(B: np.ndarray, A: np.ndarray) -> np.ndarray:
    return B.T @ A


def loss(Y: np.ndarray, Yhat: np.ndarray) -> float:
    return 0.5 * float(np.sum((Y - Yhat) ** 2))


def predict(Y: np.ndarray) -> np.ndarray:
    return np.argmax(Y, axis=1)


def fmt_matrix(M: np.ndarray) -> str:
    rows = []
    for row in M:
        rows.append("[" + " ".join(f"{x:5.0f}" for x in row) + "]")
    return "\n".join(rows)


@dataclass
class RunResult:
    name: str
    A1: np.ndarray
    A2: np.ndarray
    A3: np.ndarray
    Y0: np.ndarray
    B3: np.ndarray
    B3_new: np.ndarray
    G3: np.ndarray
    B2: np.ndarray
    B2_new: np.ndarray
    G2: np.ndarray
    B1: np.ndarray
    G1: np.ndarray
    W1_new: np.ndarray
    W2_new: np.ndarray
    W3_new: np.ndarray
    Y_new: np.ndarray


def example():
    X = np.eye(2)
    Yhat = np.array([
        [2.0, 1.0],
        [-2.0, -1.0],
    ])

    W1 = np.eye(2)
    W2 = np.eye(2)
    W3 = np.array([
        [1.0, -1.0],
        [3.0, -3.0],
    ])
    return X, Yhat, W1, W2, W3


def forward_pass(X, W1, W2, W3):
    A1 = X
    A2 = forward(A1, W1)
    A3 = forward(A2, W2)
    Y = forward(A3, W3)
    return A1, A2, A3, Y


def run_stale_left(X, Yhat, W1, W2, W3):
    A1, A2, A3, Y0 = forward_pass(X, W1, W2, W3)

    B3 = Y0 - Yhat
    G3 = grad(B3, A3)
    B3_new = B3
    B2 = backward(B3, W3)
    G2 = grad(B2, A2)
    B2_new = B2
    B1 = backward(B2, W2)
    G1 = grad(B1, A1)

    W3_new = W3 - LR * G3
    W2_new = W2 - LR * G2
    W1_new = W1 - LR * G1
    _, _, _, Y_new = forward_pass(X, W1_new, W2_new, W3_new)

    return RunResult(
        "left: stale/simultaneous",
        A1,
        A2,
        A3,
        Y0,
        B3,
        B3_new,
        G3,
        B2,
        B2_new,
        G2,
        B1,
        G1,
        W1_new,
        W2_new,
        W3_new,
        Y_new,
    )


def run_fixed_right(X, Yhat, W1, W2, W3):
    A1, A2, A3, Y0 = forward_pass(X, W1, W2, W3)

    B3 = Y0 - Yhat
    G3 = grad(B3, A3)
    W3_new = W3 - LR * G3

    Y_after_W3_update = forward(A3, W3_new)
    B3_new = B3 + (Y_after_W3_update - Y0)
    B2 = backward(B3_new, W3_new)
    G2 = grad(B2, A2)
    W2_new = W2 - LR * G2

    A3_after_W2_update = forward(A2, W2_new)
    B2_new = B2 + (A3_after_W2_update - A3)
    B1 = backward(B2_new, W2_new)
    G1 = grad(B1, A1)
    W1_new = W1 - LR * G1

    _, _, _, Y_new = forward_pass(X, W1_new, W2_new, W3_new)

    return RunResult(
        "right: interleaved/fixed",
        A1,
        A2,
        A3,
        Y0,
        B3,
        B3_new,
        G3,
        B2,
        B2_new,
        G2,
        B1,
        G1,
        W1_new,
        W2_new,
        W3_new,
        Y_new,
    )


def print_run(result: RunResult, Yhat: np.ndarray):
    print(f"\n{result.name}")
    print("-" * len(result.name))
    print(f"initial loss: {loss(result.Y0, Yhat):.0f}")
    print(f"final loss:   {loss(result.Y_new, Yhat):.0f}")
    print(f"initial pred: {predict(result.Y0).tolist()}")
    print(f"target pred:  {predict(Yhat).tolist()}")
    print(f"final pred:   {predict(result.Y_new).tolist()}")
    for label in ("B3", "B3_new", "G3", "B2", "B2_new", "G2", "B1", "G1"):
        print(f"\n{label} =\n{fmt_matrix(getattr(result, label))}")
    for label in ("W3_new", "W2_new", "W1_new", "Y_new"):
        print(f"\n{label} =\n{fmt_matrix(getattr(result, label))}")


def save_figure(left: RunResult, right: RunResult, Yhat: np.ndarray, out_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 8.2), dpi=150)
    fig.patch.set_facecolor("white")

    def block(result: RunResult) -> str:
        fixed = result.name.startswith("right:")
        if fixed:
            b3_new_line = "B3_new = B3 + (A3 @ W3_new.T - Y0) ="
            b2_line = "B2 = B3_new @ W3_new ="
            b2_new_line = "B2_new = B2 + (A2 @ W2_new.T - A3) ="
            b1_line = "B1 = B2_new @ W2_new ="
        else:
            b3_new_line = "B3_new = B3  # no local correction"
            b2_line = "B2 = B3 @ W3 ="
            b2_new_line = "B2_new = B2  # no local correction"
            b1_line = "B1 = B2 @ W2 ="

        return "\n".join(
            [
                f"{result.name}",
                "",
                f"Y0 =\n{fmt_matrix(result.Y0)}",
                f"Yhat =\n{fmt_matrix(Yhat)}",
                f"loss: {loss(result.Y0, Yhat):.0f} -> {loss(result.Y_new, Yhat):.0f}",
                f"pred: {predict(result.Y0).tolist()} -> {predict(result.Y_new).tolist()}",
                "",
                f"B3 = Y0 - Yhat =\n{fmt_matrix(result.B3)}",
                f"G3 = B3.T @ A3 =\n{fmt_matrix(result.G3)}",
                f"W3 -= G3 ->\n{fmt_matrix(result.W3_new)}",
                f"{b3_new_line}\n{fmt_matrix(result.B3_new)}",
                "",
                f"{b2_line}\n{fmt_matrix(result.B2)}",
                f"G2 =\n{fmt_matrix(result.G2)}",
                f"W2 -= G2 ->\n{fmt_matrix(result.W2_new)}",
                f"{b2_new_line}\n{fmt_matrix(result.B2_new)}",
                "",
                f"{b1_line}\n{fmt_matrix(result.B1)}",
                f"G1 =\n{fmt_matrix(result.G1)}",
                f"W1 -= G1 ->\n{fmt_matrix(result.W1_new)}",
                "",
                f"Y after all updates =\n{fmt_matrix(result.Y_new)}",
            ]
        )

    for ax, result in zip(axes, (left, right)):
        ax.axis("off")
        ax.text(
            0.03,
            0.98,
            block(result),
            transform=ax.transAxes,
            ha="left",
            va="top",
            family="monospace",
            fontsize=9.4,
            color="#111111",
            linespacing=1.18,
        )

    axes[1].text(
        0.62,
        0.58,
        "G2 = 0\nG1 = 0",
        transform=axes[1].transAxes,
        ha="center",
        va="center",
        family="monospace",
        fontsize=15,
        bbox=dict(boxstyle="round,pad=0.45", fc="#ffd98e", ec="none"),
    )
    axes[1].text(
        0.62,
        0.16,
        "Y = Yhat",
        transform=axes[1].transAxes,
        ha="center",
        va="center",
        family="monospace",
        fontsize=15,
        bbox=dict(boxstyle="round,pad=0.45", fc="#bde7c1", ec="none"),
    )

    fig.suptitle(
        "3 linear 2-feature layers: stale backprop vs interleaved fixed update",
        x=0.5,
        y=0.99,
        fontsize=14,
        fontweight="bold",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    X, Yhat, W1, W2, W3 = example()

    print("Example")
    print("-------")
    print(f"X = A1 = A2 = A3 =\n{fmt_matrix(X)}")
    print(f"\nYhat =\n{fmt_matrix(Yhat)}")
    print(f"\nW1 = W2 =\n{fmt_matrix(W1)}")
    print(f"\nW3 initial =\n{fmt_matrix(W3)}")

    left = run_stale_left(X, Yhat, W1, W2, W3)
    right = run_fixed_right(X, Yhat, W1, W2, W3)

    print_run(left, Yhat)
    print_run(right, Yhat)

    assert np.allclose(right.Y_new, Yhat)
    assert np.allclose(right.G2, 0)
    assert np.allclose(right.G1, 0)

    out_path = os.path.abspath("fixed_backprop_comparison.png")
    save_figure(left, right, Yhat, out_path)
    print(f"\nSaved figure to {out_path}")


if __name__ == "__main__":
    main()
