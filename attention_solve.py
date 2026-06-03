# /// script
# dependencies = [
#   "torch",
# ]
# ///

import time
import math
import copy
import torch
import torch.nn as nn

torch.manual_seed(42)

NUM_LAYERS = 3
LOSS_REDUCTION_FACTOR = 100.0

def generate_orthogonal_matrix(dim):
    """Generates a random orthogonal matrix using QR decomposition."""
    H = torch.randn(dim, dim)
    Q, R = torch.linalg.qr(H)
    return Q * torch.diagonal(R).sign()

def sample_so(dim):
    """Samples a random special orthogonal matrix (det +1)."""
    Q = generate_orthogonal_matrix(dim)
    return Q if torch.linalg.det(Q) > 0 else torch.cat((-Q[:1], Q[1:]), dim=0)

def generalized_rotation(dim, theta=math.pi/10):
    """Random conjugate of block-diagonal 2D rotations by theta."""
    if dim % 2 != 0: raise ValueError("generalized_rotation expects an even dimension")
    c, s = math.cos(theta), math.sin(theta)
    block = torch.tensor([[c, -s], [s, c]], dtype=torch.float32)
    real_jordan = torch.block_diag(*([block] * (dim // 2)))
    p = sample_so(dim)
    return p.T @ real_jordan @ p

def generate_orthogonal_rows_batch(batch_size, num_rows, dim):
    """Generates a batch of row-orthogonal inputs with shape (B, N, D)."""
    return torch.stack([generate_orthogonal_matrix(dim)[:num_rows] for _ in range(batch_size)])

class LinearSelfAttention(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.W_q = nn.Parameter(generalized_rotation(dim))
        self.W_k = nn.Parameter(generalized_rotation(dim))
        self.W_v = nn.Parameter(generalized_rotation(dim))

    def forward(self, X):
        Q, K, V = X @ self.W_q, X @ self.W_k, X @ self.W_v
        # Pure Polynomial Causal Mixing 
        kv_state = torch.cumsum(torch.einsum("bnd,bne->bnde", K, V), dim=1)
        return torch.einsum("bnd,bnde->bne", Q, kv_state)

class MultiLayerSelfAttention(nn.Sequential):
    def __init__(self, dim, num_layers=NUM_LAYERS):
        super().__init__(*[LinearSelfAttention(dim) for _ in range(num_layers)])

def apply_grads(params, grads, lr):
    with torch.no_grad():
        for p, g in zip(params, grads): p -= lr * g

def step_classic(model, X, target, lr):
    """Standard backpropagation update step."""
    loss = torch.mean((model(X) - target) ** 2)
    params = list(model.parameters())
    grads = torch.autograd.grad(loss, params)
    apply_grads(params, grads, lr)

def step_fixed(model, X, target, lr):
    """Alternative backpropagation update utilizing explicitly computed layer-wise targets."""
    layers = list(model)
    with torch.no_grad():
        acts = [X]
        for layer in layers:
            acts.append(layer(acts[-1]).detach())

    B = (2.0 / acts[-1].numel()) * (acts[-1] - target)
    
    for i in reversed(range(len(layers))):
        layer, A_in, A_out = layers[i], acts[i], acts[i+1]
        
        # Local parameter gradients
        params = list(layer.parameters())
        grads = torch.autograd.grad(layer(A_in.detach()), params, grad_outputs=B.detach())
        apply_grads(params, grads, lr)
        
        if i > 0:
            with torch.no_grad():
                A_out_after = layer(A_in).detach()
            # MSE Hessian scaled adjustment
            B_new = B + (2.0 / A_out.numel()) * (A_out_after - A_out)
            
            # Local input vector-Jacobian product
            A_req = A_in.detach().requires_grad_(True)
            B = torch.autograd.grad(layer(A_req), A_req, grad_outputs=B_new.detach())[0].detach()

def run_step_with_lr_tuning(model, step_fn, X, Y, lr, eval_x=None, eval_y=None, revert_on_fail=False):
    """Performs a model update step, evaluates loss, and dynamically adjusts learning rate with backtracking."""
    with torch.no_grad():
        loss_before = torch.mean((model(X) - Y) ** 2).item()
    state_backup = copy.deepcopy(model.state_dict()) if revert_on_fail else None

    while True:
        try:
            step_fn(model, X, Y, lr)
            with torch.no_grad():
                loss_after = torch.mean((model(X) - Y) ** 2).item()
            failed = not math.isfinite(loss_after) or (loss_after > loss_before if revert_on_fail else loss_after >= loss_before)
        except Exception:
            failed = True

        if failed:
            if revert_on_fail: model.load_state_dict(state_backup)
            lr *= 0.8
            if not revert_on_fail or lr < 1e-15: break
        else:
            lr *= 1.1
            break

    eval_loss = torch.mean((model(eval_x) - eval_y) ** 2).item() if eval_x is not None else None
    return lr, eval_loss

def train():
    start_time = time.time()
    dim, num_rows, batch_size, max_steps = 32, 8, 32, 1000

    print(f"Initializing Teacher Model (layers={NUM_LAYERS}, dim={dim}, seq_len={num_rows})...")
    teacher = MultiLayerSelfAttention(dim).requires_grad_(False)

    print("Initializing Student Models...")
    student_init = MultiLayerSelfAttention(dim)
    with torch.no_grad():
        for p in student_init.parameters(): p.copy_(torch.eye(dim))

    model_c, model_f = copy.deepcopy(student_init), copy.deepcopy(student_init)

    print("Tuning initial learning rates (via backtracking search over 100 steps)...")
    lr_c = lr_f = 1.0
    torch.manual_seed(999) 
    for _ in range(100):
        X = generate_orthogonal_rows_batch(batch_size, num_rows, dim)
        with torch.no_grad(): Y = teacher(X)
        lr_c, _ = run_step_with_lr_tuning(model_c, step_classic, X, Y, lr_c)
        lr_f, _ = run_step_with_lr_tuning(model_f, step_fixed, X, Y, lr_f)

    print(f"-> LR Tuning Phase Completed | backprop LR: {lr_c:.3f} | altprop LR: {lr_f:.3f}")

    # Restore to initialized state, maintaining the stable learning rates tuned from step-tests above
    model_c.load_state_dict(student_init.state_dict())
    model_f.load_state_dict(student_init.state_dict())

    torch.manual_seed(100)
    eval_x = generate_orthogonal_rows_batch(batch_size, num_rows, dim)
    with torch.no_grad():
        eval_y = teacher(eval_x)
        start_eval_loss = torch.mean((model_c(eval_x) - eval_y) ** 2).item()
        
    target_eval_loss = start_eval_loss / LOSS_REDUCTION_FACTOR
    classic_reached = fixed_reached = None
    print(f"\nStopping criterion: both methods reach <= {target_eval_loss:.2e} "
          f"({LOSS_REDUCTION_FACTOR:.0f}x below start {start_eval_loss:.2e}).")

    torch.manual_seed(42)
    for step in range(1, max_steps + 1):
        X = generate_orthogonal_rows_batch(batch_size, num_rows, dim)
        with torch.no_grad(): Y = teacher(X)

        lr_c, loss_c = run_step_with_lr_tuning(model_c, step_classic, X, Y, lr_c, eval_x, eval_y, revert_on_fail=True)
        lr_f, loss_f = run_step_with_lr_tuning(model_f, step_fixed, X, Y, lr_f, eval_x, eval_y, revert_on_fail=True)

        if classic_reached is None and loss_c <= target_eval_loss: classic_reached = step
        if fixed_reached is None and loss_f <= target_eval_loss: fixed_reached = step

        if step % 10 == 0 or step == 1:
            print(f"Step {step:3d} | backprop eval loss: {loss_c:.2e} (LR: {lr_c:.1f}) | altprop eval loss: {loss_f:.2e} (LR: {lr_f:.1f})")

        if classic_reached and fixed_reached:
            print(f"\nReached {LOSS_REDUCTION_FACTOR:.0f}x reduction: backprop at step {classic_reached}, "
                  f"altprop at step {fixed_reached}; stopping at step {step}.")
            break
    else:
        print(f"\nReached max_steps={max_steps} before both hit target. backprop={classic_reached}, altprop={fixed_reached}")

    print(f"End-to-end execution time: {time.time() - start_time:.3f} seconds")

if __name__ == '__main__':
    train()
