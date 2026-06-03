"""
Self-contained Colab-ready learning-rate sensitivity plot for classic vs fixed
backprop on the attention reconstruction task.

Network:
  Three self-attention layers, dim=8, sequence length=6, batch size=64.

Task:
  A frozen teacher network maps a fixed orthogonal-row input batch to a target.
  A student with the same architecture is updated for 1, 3, or 10 steps.

The plot compares loss after 1, 3, and 10 update steps over a range of learning
rates.  Every method and every learning rate starts from the same student
initialization and the same input batch.
"""

from __future__ import annotations

import copy
import os
import warnings

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


SEED = 42
DIM = 8
NUM_ROWS = 6
BATCH_SIZE = 64
STEP_COUNTS = (1, 3, 10)
LR_MIN = 0.0
LR_MAX = 250.0
NUM_LRS = 251
UNSTABLE_LOSS = 1e6
OUT_PATH = "classic_vs_fixed_attention.png"
SMALL_ROTATION_ANGLE = np.pi / 100


def generate_orthogonal_matrix(dim: int) -> torch.Tensor:
    """Generate a random orthogonal matrix using QR decomposition."""
    H = torch.randn(dim, dim)
    Q, R = torch.linalg.qr(H)
    signs = torch.diag(R).sign()
    return Q * signs


def enforce_rotation(mat: torch.Tensor) -> torch.Tensor:
    """Flip the first row if needed so the orthogonal matrix has det +1."""
    return mat if torch.linalg.det(mat) > 0 else torch.cat((-mat[:1], mat[1:]), dim=0)


def sample_so(dim: int) -> torch.Tensor:
    return enforce_rotation(generate_orthogonal_matrix(dim))


def generalized_rotation(dim: int, theta: float = SMALL_ROTATION_ANGLE) -> torch.Tensor:
    """Random conjugate of block-diagonal 2D rotations by theta."""
    if dim % 2 != 0:
        raise ValueError("generalized_rotation expects an even dimension")

    c = torch.cos(torch.tensor(theta, dtype=torch.float32))
    s = torch.sin(torch.tensor(theta, dtype=torch.float32))
    block = torch.tensor([[c, -s], [s, c]], dtype=torch.float32)
    real_jordan = torch.block_diag(*[block for _ in range(dim // 2)])
    p = sample_so(dim)
    return p.T @ real_jordan @ p


def generate_orthogonal_rows_batch(batch_size: int, num_rows: int, dim: int) -> torch.Tensor:
    """Generate a batch of row-orthogonal inputs with shape (B, N, D)."""
    examples = []
    for _ in range(batch_size):
        Q = generate_orthogonal_matrix(dim)
        examples.append(Q[:num_rows, :])
    return torch.stack(examples)


class SelfAttentionLayer(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.W_q = nn.Parameter(torch.empty(dim, dim))
        self.W_k = nn.Parameter(torch.empty(dim, dim))
        self.W_v = nn.Parameter(torch.empty(dim, dim))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        with torch.no_grad():
            self.W_q.copy_(generalized_rotation(self.dim))
            self.W_k.copy_(generalized_rotation(self.dim))
            self.W_v.copy_(generalized_rotation(self.dim))

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        Q = X @ self.W_q
        K = X @ self.W_k
        V = X @ self.W_v
        Q = F.softplus(Q)
        K = F.softplus(K)

        # Normalized causal linear self-attention:
        #   S_i = sum_{j<=i} K_j outer V_j
        #   z_i = sum_{j<=i} K_j
        #   Y_i = (Q_i^T S_i) / (Q_i^T z_i)
        kv_state = torch.cumsum(torch.einsum("bnd,bne->bnde", K, V), dim=1)
        k_state = torch.cumsum(K, dim=1)
        numerator = torch.einsum("bnd,bnde->bne", Q, kv_state)
        denominator = torch.einsum("bnd,bnd->bn", Q, k_state).unsqueeze(-1)
        return numerator / denominator.clamp_min(1e-6)


class ThreeLayerSelfAttention(nn.Module):
    def __init__(self, dim: int):
        super().__init__()
        self.layer1 = SelfAttentionLayer(dim)
        self.layer2 = SelfAttentionLayer(dim)
        self.layer3 = SelfAttentionLayer(dim)

    def forward(self, X: torch.Tensor) -> torch.Tensor:
        x = self.layer1(X)
        x = self.layer2(x)
        x = self.layer3(x)
        return x


def layers(model: ThreeLayerSelfAttention):
    return [model.layer1, model.layer2, model.layer3]


def layer_params(layer):
    return [layer.W_q, layer.W_k, layer.W_v]


def reconstruction_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.mean((pred - target) ** 2)


def mse_grad(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return (2.0 / pred.numel()) * (pred - target)


def mse_hessian_scale(tensor: torch.Tensor) -> float:
    return 2.0 / tensor.numel()


def forward_activations(model: ThreeLayerSelfAttention, X: torch.Tensor) -> list[torch.Tensor]:
    activations = [X]
    for layer in layers(model):
        activations.append(layer(activations[-1]))
    return activations


def make_problem():
    torch.manual_seed(SEED)
    np.random.seed(SEED)

    teacher = ThreeLayerSelfAttention(DIM)
    for p in teacher.parameters():
        p.requires_grad_(False)

    student = ThreeLayerSelfAttention(DIM)
    student_state = copy.deepcopy(student.state_dict())

    X = generate_orthogonal_rows_batch(BATCH_SIZE, NUM_ROWS, DIM)
    with torch.no_grad():
        target = teacher(X).detach()

    return X, target, student_state


def make_student(student_state) -> ThreeLayerSelfAttention:
    model = ThreeLayerSelfAttention(DIM)
    model.load_state_dict(copy.deepcopy(student_state))
    return model


def apply_grads(params, grads, lr: float) -> None:
    with torch.no_grad():
        for p, g in zip(params, grads):
            p -= lr * g


def step_classic(
    model: ThreeLayerSelfAttention,
    X: torch.Tensor,
    target: torch.Tensor,
    lr: float,
) -> None:
    pred = model(X)
    loss = reconstruction_loss(pred, target)
    params = list(model.parameters())
    grads = torch.autograd.grad(loss, params)
    apply_grads(params, grads, lr)


def local_param_grads(layer, A: torch.Tensor, grad_out: torch.Tensor):
    out = layer(A.detach())
    return torch.autograd.grad(out, layer_params(layer), grad_outputs=grad_out.detach())


def local_input_vjp(layer, A: torch.Tensor, grad_out: torch.Tensor) -> torch.Tensor:
    A_req = A.detach().requires_grad_(True)
    out = layer(A_req)
    return torch.autograd.grad(out, A_req, grad_outputs=grad_out.detach())[0].detach()


def step_fixed(
    model: ThreeLayerSelfAttention,
    X: torch.Tensor,
    target: torch.Tensor,
    lr: float,
) -> None:
    """
    Top-to-bottom fixed backprop.

    Backward values are represented in mean-MSE gradient units.  Therefore the
    local-forward residual refresh B += A_new - A from the unnormalized squared
    loss becomes B += (2 / numel) * (A_new - A).
    """
    layer1, layer2, layer3 = layers(model)
    with torch.no_grad():
        A0, A1, A2, Y = [a.detach() for a in forward_activations(model, X)]

    B3 = mse_grad(Y, target)
    grads3 = local_param_grads(layer3, A2, B3)
    apply_grads(layer_params(layer3), grads3, lr)

    with torch.no_grad():
        Y_after_W3 = layer3(A2).detach()
    B3_new = B3 + mse_hessian_scale(Y) * (Y_after_W3 - Y)
    B2 = local_input_vjp(layer3, A2, B3_new)

    grads2 = local_param_grads(layer2, A1, B2)
    apply_grads(layer_params(layer2), grads2, lr)

    with torch.no_grad():
        A2_after_W2 = layer2(A1).detach()
    B2_new = B2 + mse_hessian_scale(A2) * (A2_after_W2 - A2)
    B1 = local_input_vjp(layer2, A1, B2_new)

    grads1 = local_param_grads(layer1, A0, B1)
    apply_grads(layer_params(layer1), grads1, lr)


def loss_after_steps(
    step_fn,
    student_state,
    X: torch.Tensor,
    target: torch.Tensor,
    lr: float,
    steps: int,
) -> tuple[float, bool]:
    model = make_student(student_state)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with np.errstate(all="ignore"):
            for _ in range(steps):
                step_fn(model, X, target, lr)
                if not all(torch.isfinite(p).all() for p in model.parameters()):
                    return np.nan, True

    with torch.no_grad():
        pred = model(X)
        if not torch.isfinite(pred).all():
            return np.nan, True
        loss = reconstruction_loss(pred, target).item()

    if not np.isfinite(loss) or loss > UNSTABLE_LOSS:
        return np.nan, True

    return loss, False


def sweep_method(step_fn, student_state, X: torch.Tensor, target: torch.Tensor, lrs: np.ndarray, steps: int):
    losses = np.empty_like(lrs, dtype=float)
    unstable = np.zeros_like(lrs, dtype=bool)

    for i, lr in enumerate(lrs):
        losses[i], unstable[i] = loss_after_steps(step_fn, student_state, X, target, float(lr), steps)

    return losses, unstable


def best_finite(lrs: np.ndarray, losses: np.ndarray) -> tuple[float, float]:
    idx = int(np.nanargmin(losses))
    return float(lrs[idx]), float(losses[idx])


def plot_sensitivity(lrs: np.ndarray, sweeps: dict[int, dict[str, tuple[np.ndarray, np.ndarray]]]):
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
        y_min = max(float(np.min(finite_losses)) * 0.2, 1e-8)
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

    axes[0].set_ylabel("MSE reconstruction loss")
    fig.suptitle("Learning-rate sensitivity on the attention reconstruction task", y=1.02, fontsize=15, fontweight="bold")
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
    fig.savefig(OUT_PATH, bbox_inches="tight", facecolor="white")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plt.show(block=False)
    plt.close(fig)
    return os.path.abspath(OUT_PATH)


def main() -> None:
    X, target, student_state = make_problem()
    lrs = np.linspace(LR_MIN, LR_MAX, NUM_LRS)

    sweeps: dict[int, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    for steps in STEP_COUNTS:
        sweeps[steps] = {
            "classic": sweep_method(step_classic, student_state, X, target, lrs, steps),
            "fixed": sweep_method(step_fixed, student_state, X, target, lrs, steps),
        }

    initial_model = make_student(student_state)
    with torch.no_grad():
        initial_loss = reconstruction_loss(initial_model(X), target).item()

    print(f"attention network: dim={DIM}, seq_len={NUM_ROWS}, batch_size={BATCH_SIZE}")
    print(f"steps per run: {STEP_COUNTS}")
    print(f"learning-rate range: [{LR_MIN}, {LR_MAX}] with {NUM_LRS} points")
    print(f"shared initial loss: {initial_loss:.6e}")
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
