# /// script
# dependencies = [
#   "torch",
#   "matplotlib",
#   "numpy",
# ]
# ///

import time
import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from dashboard import generate_html_dashboard

# Set random seed for reproducibility
torch.manual_seed(42)
np.random.seed(42)

SMALL_ROTATION_ANGLE = math.pi / 10
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

def generalized_rotation(dim, theta=SMALL_ROTATION_ANGLE):
    """Random conjugate of block-diagonal 2D rotations by theta."""
    if dim % 2 != 0:
        raise ValueError("generalized_rotation expects an even dimension")

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
        self.dim = dim
        self.W_q = nn.Parameter(generalized_rotation(dim))
        self.W_k = nn.Parameter(generalized_rotation(dim))
        self.W_v = nn.Parameter(generalized_rotation(dim))

    def forward(self, X):
        Q = X @ self.W_q
        K = X @ self.W_k
        V = X @ self.W_v

        # 1. Pure Polynomial Causal Mixing 
        kv_state = torch.cumsum(torch.einsum("bnd,bne->bnde", K, V), dim=1)
        return torch.einsum("bnd,bnde->bne", Q, kv_state)

NUM_LAYERS = 1

class MultiLayerSelfAttention(nn.Sequential):
    def __init__(self, dim, num_layers=NUM_LAYERS):
        super().__init__(*[LinearSelfAttention(dim) for _ in range(num_layers)])

def mse_hessian_scale(tensor):
    return 2.0 / tensor.numel()

def reconstruction_loss(pred, target):
    return torch.mean((pred - target) ** 2)

def reconstruction_loss_grad(pred, target):
    return (2.0 / pred.numel()) * (pred - target)

def compute_average_angle(pred, target):
    """Computes the average angle in radians normalized by pi (ranges from 0 to 1) between predicted and target vectors."""
    cos_sim = F.cosine_similarity(pred, target, dim=-1)
    cos_sim = torch.clamp(cos_sim, -1.0, 1.0)
    angles = torch.acos(cos_sim)
    angles_normalized = angles / math.pi
    return torch.mean(angles_normalized).item()

def forward_activations(model, X):
    activations = [X]
    for layer in model:
        activations.append(layer(activations[-1]))
    return activations

def apply_grads(params, grads, lr):
    with torch.no_grad():
        for p, g in zip(params, grads):
            p -= lr * g

def local_param_grads(layer, A, grad_out):
    out = layer(A.detach())
    return torch.autograd.grad(out, list(layer.parameters()), grad_outputs=grad_out.detach())

def local_input_vjp(layer, A, grad_out):
    A_req = A.detach().requires_grad_(True)
    out = layer(A_req)
    return torch.autograd.grad(out, A_req, grad_outputs=grad_out.detach())[0].detach()

def step_classic(model, X, target, lr):
    loss = reconstruction_loss(model(X), target)
    grads = torch.autograd.grad(loss, list(model.parameters()))
    apply_grads(list(model.parameters()), grads, lr)

def step_fixed(model, X, target, lr):
    layers = list(model)
    with torch.no_grad():
        activations = [a.detach() for a in forward_activations(model, X)]

    B = reconstruction_loss_grad(activations[-1], target)
    
    # Generic loop replacing explicitly unrolled reverse-layer logic
    for i in reversed(range(len(layers))):
        layer = layers[i]
        A_in = activations[i]
        A_out = activations[i+1]
        
        grads = local_param_grads(layer, A_in, B)
        apply_grads(list(layer.parameters()), grads, lr)
        
        if i > 0:
            with torch.no_grad():
                A_out_after = layer(A_in).detach()
            B_new = B + mse_hessian_scale(A_out) * (A_out_after - A_out)
            B = local_input_vjp(layer, A_in, B_new)

def run_step_with_lr_tuning(model, step_fn, X, Y, lr, eval_x=None, eval_y=None, revert_on_fail=False):
    """Performs a model update step, evaluates loss, and dynamically adjusts learning rate with backtracking."""
    with torch.no_grad():
        loss_before = reconstruction_loss(model(X), Y).item()
        
    state_backup = copy.deepcopy(model.state_dict()) if revert_on_fail else None
    
    if revert_on_fail:
        while True:
            try:
                step_fn(model, X, Y, lr)
                with torch.no_grad():
                    loss_after = reconstruction_loss(model(X), Y).item()
                
                if not math.isfinite(loss_after) or loss_after > loss_before:
                    model.load_state_dict(state_backup)
                    lr *= 0.9
                    if lr < 1e-15:
                        loss_after = loss_before
                        break
                else:
                    lr *= 1.1
                    break
            except Exception:
                model.load_state_dict(state_backup)
                lr *= 0.9
                if lr < 1e-15:
                    loss_after = loss_before
                    break
    else:
        try:
            step_fn(model, X, Y, lr)
            with torch.no_grad():
                loss_after = reconstruction_loss(model(X), Y).item()
            
            if not math.isfinite(loss_after) or loss_after >= loss_before:
                lr *= 0.9
            else:
                lr *= 1.1
        except Exception:
            loss_after = float('inf')
            lr *= 0.9

    eval_loss = None
    if eval_x is not None and eval_y is not None:
        with torch.no_grad():
            eval_loss = reconstruction_loss(model(eval_x), eval_y).item()
            
    return lr, loss_after, eval_loss

def tune_lr_backtracking_100steps(student_initial_state, teacher, dim, batch_size, num_rows, start_lr=1.0):
    """Fine-grained learning rate tuning leveraging dynamic tuning helper method."""
    num_layers = len(list(teacher))
    model_c = MultiLayerSelfAttention(dim, num_layers=num_layers)
    model_c.load_state_dict(copy.deepcopy(student_initial_state))
    model_f = MultiLayerSelfAttention(dim, num_layers=num_layers)
    model_f.load_state_dict(copy.deepcopy(student_initial_state))

    classic_lr = fixed_lr = start_lr

    torch.manual_seed(999) # Separate fixed seed for LR tuning batches
    for step in range(1, 101):
        batch_x = generate_orthogonal_rows_batch(batch_size, num_rows, dim)
        with torch.no_grad():
            batch_y = teacher(batch_x)

        classic_lr, loss_c, _ = run_step_with_lr_tuning(model_c, step_classic, batch_x, batch_y, classic_lr)
        fixed_lr, loss_f, _ = run_step_with_lr_tuning(model_f, step_fixed, batch_x, batch_y, fixed_lr)

        if step >= 60:
            print(f"  Tune Step {step:3d} | Classic LR: {classic_lr:.1f} (loss: {loss_c:.2e}) | Fixed LR: {fixed_lr:.1f} (loss: {loss_f:.2e})")

    print(f"-> LR Tuning Phase Completed (100 steps) | Classic LR: {classic_lr:.3f} | Fixed LR: {fixed_lr:.3f}")
    return classic_lr, fixed_lr

def run_initial_lr_search(student_initial_state, teacher, dim, batch_size, num_rows):
    """
    Computes and plots the learning rate sensitivity (1-step and 10-step losses)
    over a grid of learning rates.
    """
    print("\nRunning Initial Learning Rate Search (1-step and 10-step sensitivity)...")
    student_init = MultiLayerSelfAttention(dim, num_layers=len(list(teacher)))
    student_init.load_state_dict(copy.deepcopy(student_initial_state))
    
    # Define two fixed batches
    torch.manual_seed(123)
    batch_x_a = generate_orthogonal_rows_batch(batch_size, num_rows, dim)
    with torch.no_grad():
        batch_y_a = teacher(batch_x_a)

    batch_x_b = generate_orthogonal_rows_batch(batch_size, num_rows, dim)
    with torch.no_grad():
        batch_y_b = teacher(batch_x_b)

    # Compute initial loss (step 0)
    with torch.no_grad():
        init_loss_a = reconstruction_loss(student_init(batch_x_a), batch_y_a).item()

    # Range of learning rates (log space)
    lrs = np.logspace(-2, 3, 200) # 200 points from 10^-2 (0.01) to 10^3 (1000)
    steps_list = [1, 10]

    # Pre-allocate dictionary for results
    results = {
        s: {"same": [], "diff": []} for s in steps_list
    }

    for lr in lrs:
        for steps in steps_list:
            # Load starting model state
            model = MultiLayerSelfAttention(dim, num_layers=len(list(teacher)))
            model.load_state_dict(copy.deepcopy(student_initial_state))
            
            # Take multiple gradient steps on Batch A
            for _ in range(steps):
                step_classic(model, batch_x_a, batch_y_a, lr)
                
            # Compute loss after steps
            with torch.no_grad():
                loss_same = reconstruction_loss(model(batch_x_a), batch_y_a).item()
                loss_diff = reconstruction_loss(model(batch_x_b), batch_y_b).item()
                
            results[steps]["same"].append(loss_same)
            results[steps]["diff"].append(loss_diff)

    # Find the maximum learning rate where same-batch loss is still below the initial loss (across any of the steps)
    stable_lrs = []
    for s in steps_list:
        for lr, loss in zip(lrs, results[s]["same"]):
            if loss < init_loss_a:
                stable_lrs.append(lr)
    max_stable_lr = max(stable_lrs) if stable_lrs else lrs[-1]

    # Plotting the results
    fig, ax = plt.subplots(figsize=(8, 5.5), dpi=150)
    plt.style.use('seaborn-v0_8-whitegrid')

    colors = {1: '#1f77b4', 10: '#2ca02c'}

    # Plot Step 0 initial reference line
    ax.axhline(init_loss_a, color='#7f7f7f', linestyle=':', linewidth=1.5, label='Initial Loss (Step 0)')

    for s in steps_list:
        # Same batch (solid lines)
        ax.plot(lrs, results[s]["same"], label=f'{s} Step(s) - Same Batch (A)', color=colors[s], linewidth=2.0)
        # Different batch (dashed lines)
        ax.plot(lrs, results[s]["diff"], label=f'{s} Step(s) - Different Batch (B)', color=colors[s], linestyle='--', linewidth=2.0)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(lrs[0], max_stable_lr)
    # Set y limit to only show values below initial loss (plus 5% margin for visual clearance)
    ax.set_ylim(None, init_loss_a * 1.05)

    ax.set_xlabel('Learning Rate (log scale)', fontsize=11)
    ax.set_ylabel('Mean Squared Error (MSE) Loss (log scale)', fontsize=11)
    ax.set_title('Initial Learning Rate Search: 1-step and 10-step Loss Sensitivity', fontsize=12, fontweight='bold')
    ax.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=9)

    plt.tight_layout()
    plot_path = "initial_lr_search.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Saved initial learning rate search plot to {plot_path}")



def train():
    start_time = time.time()
    dim, num_rows, batch_size, max_steps = 32, 8, 32, 1000

    print(f"Initializing Teacher Model (layers={NUM_LAYERS}, dim={dim}, seq_len={num_rows})...")
    teacher = MultiLayerSelfAttention(dim).requires_grad_(False)

    print("Initializing Student Initialization State...")
    student_init = MultiLayerSelfAttention(dim)
    with torch.no_grad():
        for p in student_init.parameters():
            p.copy_(torch.eye(dim))
    student_initial_state = copy.deepcopy(student_init.state_dict())

    # Run initial learning rate search (1-step and 10-step sweep)
    run_initial_lr_search(student_initial_state, teacher, dim, batch_size, num_rows)

    print("Tuning initial learning rates (via backtracking search over 100 steps)...")
    classic_lr, fixed_lr = tune_lr_backtracking_100steps(
        student_initial_state, teacher, dim, batch_size, num_rows, start_lr=1.0
    )

    print("\nRunning training with dynamic learning rate tuning...")
    student_classic = MultiLayerSelfAttention(dim)
    student_classic.load_state_dict(copy.deepcopy(student_initial_state))
    student_fixed = MultiLayerSelfAttention(dim)
    student_fixed.load_state_dict(copy.deepcopy(student_initial_state))

    torch.manual_seed(100)
    eval_x = generate_orthogonal_rows_batch(batch_size, num_rows, dim)
    with torch.no_grad():
        eval_y = teacher(eval_x)
        start_eval_loss = reconstruction_loss(student_classic(eval_x), eval_y).item()
        
    target_eval_loss = start_eval_loss / LOSS_REDUCTION_FACTOR
    classic_reached_step = fixed_reached_step = None
    print(f"Stopping criterion: both methods reach <= {target_eval_loss:.2e} "
          f"on fixed eval batch ({LOSS_REDUCTION_FACTOR:.0f}x below start {start_eval_loss:.2e}).")

    class_loss_ind = [start_eval_loss]
    fixed_loss_ind = [start_eval_loss]
    classic_lr_history = [classic_lr]
    fixed_lr_history = [fixed_lr]

    with torch.no_grad():
        init_angle_c = compute_average_angle(student_classic(eval_x), eval_y)
        init_angle_f = compute_average_angle(student_fixed(eval_x), eval_y)
    class_angles_history = [init_angle_c]
    fixed_angles_history = [init_angle_f]

    # Track weight changes for student_classic
    classic_init_params = {name: p.clone().detach() for name, p in student_classic.named_parameters()}
    classic_prev_params = {name: p.clone().detach() for name, p in student_classic.named_parameters()}
    classic_dist_to_start = {name: [0.0] for name in classic_init_params}
    classic_path_length = {name: [0.0] for name in classic_init_params}

    # Track weight changes for student_fixed
    fixed_init_params = {name: p.clone().detach() for name, p in student_fixed.named_parameters()}
    fixed_prev_params = {name: p.clone().detach() for name, p in student_fixed.named_parameters()}
    fixed_dist_to_start = {name: [0.0] for name in fixed_init_params}
    fixed_path_length = {name: [0.0] for name in fixed_init_params}

    torch.manual_seed(42)
    final_step = max_steps
    
    for step in range(1, max_steps + 1):
        batch_x = generate_orthogonal_rows_batch(batch_size, num_rows, dim)
        with torch.no_grad():
            batch_y = teacher(batch_x)

        # CLASSIC MODEL step evaluation
        classic_lr_used = classic_lr
        classic_lr, _, loss_class_eval = run_step_with_lr_tuning(
            student_classic, step_classic, batch_x, batch_y, classic_lr, eval_x, eval_y, revert_on_fail=True
        )
        class_loss_ind.append(loss_class_eval)
        classic_lr_history.append(classic_lr_used)

        # FIXED MODEL step evaluation
        fixed_lr_used = fixed_lr
        fixed_lr, _, loss_fixed_eval = run_step_with_lr_tuning(
            student_fixed, step_fixed, batch_x, batch_y, fixed_lr, eval_x, eval_y, revert_on_fail=True
        )
        fixed_loss_ind.append(loss_fixed_eval)
        fixed_lr_history.append(fixed_lr_used)

        with torch.no_grad():
            angle_class_eval = compute_average_angle(student_classic(eval_x), eval_y)
            angle_fixed_eval = compute_average_angle(student_fixed(eval_x), eval_y)
        class_angles_history.append(angle_class_eval)
        fixed_angles_history.append(angle_fixed_eval)

        # Track weight updates
        with torch.no_grad():
            for name, p in student_classic.named_parameters():
                dist_start = torch.norm(p - classic_init_params[name], p='fro').item()
                classic_dist_to_start[name].append(dist_start)
                
                step_dist = torch.norm(p - classic_prev_params[name], p='fro').item()
                new_path = classic_path_length[name][-1] + step_dist
                classic_path_length[name].append(new_path)
                
                classic_prev_params[name].copy_(p)

            for name, p in student_fixed.named_parameters():
                dist_start = torch.norm(p - fixed_init_params[name], p='fro').item()
                fixed_dist_to_start[name].append(dist_start)
                
                step_dist = torch.norm(p - fixed_prev_params[name], p='fro').item()
                new_path = fixed_path_length[name][-1] + step_dist
                fixed_path_length[name].append(new_path)
                
                fixed_prev_params[name].copy_(p)

        if classic_reached_step is None and loss_class_eval <= target_eval_loss:
            classic_reached_step = step
        if fixed_reached_step is None and loss_fixed_eval <= target_eval_loss:
            fixed_reached_step = step

        if step % 10 == 0 or step == 1:
            print(f"Step {step:3d} | Classic eval loss: {loss_class_eval:.2e} (LR: {classic_lr_used:.1f}) | "
                  f"Fixed eval loss: {loss_fixed_eval:.2e} (LR: {fixed_lr_used:.1f})")

        if classic_reached_step is not None and fixed_reached_step is not None:
            final_step = step
            print(f"Reached 100x reduction: classic at step {classic_reached_step}, "
                  f"fixed at step {fixed_reached_step}; stopping at step {step}.")
            break
    else:
        print(f"Reached max_steps={max_steps} before both methods hit the 100x target. "
              f"classic_reached={classic_reached_step}, fixed_reached={fixed_reached_step}")

    # Plot 1: Reconstruction Error
    fig1, ax1 = plt.subplots(figsize=(8, 5.5), dpi=150)
    plt.style.use('seaborn-v0_8-whitegrid')
    ax1.plot(class_loss_ind, label='Classic - Fixed Eval Batch', color='#1f77b4', linewidth=3.0)
    ax1.plot(fixed_loss_ind, label='Fixed - Fixed Eval Batch', color='#ff7f0e', linewidth=1.5)
    ax1.axhline(target_eval_loss, color='#333333', linewidth=1.0, linestyle='-.',
                label=f'100x target ({target_eval_loss:.1e})')
    ax1.set_yscale('log')
    ax1.set_title(
        f'Reconstruction Loss (MSE) on New Batch over Steps\n'
        f'Stopped at step {final_step} (Classic: {classic_reached_step or "N/A"}, Fixed: {fixed_reached_step or "N/A"})',
        fontsize=12, fontweight='bold'
    )
    ax1.set_xlabel('Step', fontsize=11)
    ax1.set_ylabel('Mean Squared Error (MSE) Loss', fontsize=11)
    ax1.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=9)
    plt.tight_layout()
    plot_path_error = 'reconstruction_error.png'
    plt.savefig(plot_path_error)
    plt.close()
    print(f"Saved reconstruction error plot to {plot_path_error}")

    # Plot 2: Dynamic Learning Rate
    fig2, ax2 = plt.subplots(figsize=(8, 5.5), dpi=150)
    ax2.plot(classic_lr_history, label='Classic Learning Rate', color='#1f77b4', linewidth=3.0)
    ax2.plot(fixed_lr_history, label='Fixed Learning Rate', color='#ff7f0e', linewidth=1.5, linestyle='--')
    ax2.set_title(
        f'Dynamic Learning Rate Schedule over Steps\n'
        f'Final Classic LR={classic_lr:.1f} | Final Fixed LR={fixed_lr:.1f}',
        fontsize=12, fontweight='bold'
    )
    ax2.set_xlabel('Step', fontsize=11)
    ax2.set_ylabel('Learning Rate (LR)', fontsize=11)
    ax2.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=9)
    plt.tight_layout()
    plot_path_lr = 'dynamic_lr_over_time.png'
    plt.savefig(plot_path_lr)
    plt.close()
    print(f"Saved dynamic learning rate plot to {plot_path_lr}")

    # Plot 3: Target Angles over Time
    fig3, ax3 = plt.subplots(figsize=(8, 5.5), dpi=150)
    ax3.plot(class_angles_history, label='Classic - Fixed Eval Batch', color='#1f77b4', linewidth=3.0)
    ax3.plot(fixed_angles_history, label='Fixed - Fixed Eval Batch', color='#ff7f0e', linewidth=1.5)
    ax3.set_title(
        f'Average Angle Between Produced and Desired Targets\n'
        f'Stopped at step {final_step} (Classic: {classic_reached_step or "N/A"}, Fixed: {fixed_reached_step or "N/A"})',
        fontsize=12, fontweight='bold'
    )
    ax3.set_xlabel('Step', fontsize=11)
    ax3.set_ylabel(r'Average Angle (normalized by $\pi$ rad)', fontsize=11)
    ax3.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=9)
    plt.tight_layout()
    plot_path_angles = 'target_angles_over_time.png'
    plt.savefig(plot_path_angles)
    plt.close()
    print(f"Saved target angles plot to {plot_path_angles}")

    # Plot 4: Parameter Changes over Time (Frobenius Norms)
    fig4, (ax_q, ax_k, ax_v) = plt.subplots(3, 1, figsize=(8, 12), sharex=True, dpi=150)
    plt.style.use('seaborn-v0_8-whitegrid')
    
    # Find parameter keys robustly
    name_q = next((k for k in classic_init_params if 'W_q' in k), None)
    name_k = next((k for k in classic_init_params if 'W_k' in k), None)
    name_v = next((k for k in classic_init_params if 'W_v' in k), None)

    # Plot W_q
    if name_q:
        ax_q.plot(classic_dist_to_start[name_q], label='Classic - Dist to Start', color='#1f77b4', linestyle='-', linewidth=2.5)
        ax_q.plot(classic_path_length[name_q], label='Classic - Path Length', color='#1f77b4', linestyle='--', linewidth=2.0)
        ax_q.plot(fixed_dist_to_start[name_q], label='Fixed - Dist to Start', color='#aec7e8', linestyle=':', linewidth=2.0)
        ax_q.plot(fixed_path_length[name_q], label='Fixed - Path Length', color='#aec7e8', linestyle='-.', linewidth=1.5)
        ax_q.set_title('Query Weight Matrix ($W_q$) Changes', fontsize=11, fontweight='bold')
        ax_q.set_ylabel('Frobenius Norm', fontsize=10)
        ax_q.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=8, ncol=2)

    # Plot W_k
    if name_k:
        ax_k.plot(classic_dist_to_start[name_k], label='Classic - Dist to Start', color='#2ca02c', linestyle='-', linewidth=2.5)
        ax_k.plot(classic_path_length[name_k], label='Classic - Path Length', color='#2ca02c', linestyle='--', linewidth=2.0)
        ax_k.plot(fixed_dist_to_start[name_k], label='Fixed - Dist to Start', color='#98df8a', linestyle=':', linewidth=2.0)
        ax_k.plot(fixed_path_length[name_k], label='Fixed - Path Length', color='#98df8a', linestyle='-.', linewidth=1.5)
        ax_k.set_title('Key Weight Matrix ($W_k$) Changes', fontsize=11, fontweight='bold')
        ax_k.set_ylabel('Frobenius Norm', fontsize=10)
        ax_k.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=8, ncol=2)

    # Plot W_v
    if name_v:
        ax_v.plot(classic_dist_to_start[name_v], label='Classic - Dist to Start', color='#d62728', linestyle='-', linewidth=2.5)
        ax_v.plot(classic_path_length[name_v], label='Classic - Path Length', color='#d62728', linestyle='--', linewidth=2.0)
        ax_v.plot(fixed_dist_to_start[name_v], label='Fixed - Dist to Start', color='#ff9896', linestyle=':', linewidth=2.0)
        ax_v.plot(fixed_path_length[name_v], label='Fixed - Path Length', color='#ff9896', linestyle='-.', linewidth=1.5)
        ax_v.set_title('Value Weight Matrix ($W_v$) Changes', fontsize=11, fontweight='bold')
        ax_v.set_xlabel('Step', fontsize=10)
        ax_v.set_ylabel('Frobenius Norm', fontsize=10)
        ax_v.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=8, ncol=2)

    fig4.suptitle('Weight Matrix Changes over Steps (Frobenius Norm)', fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout()
    plot_path_params = 'weight_changes_over_time.png'
    plt.savefig(plot_path_params)
    plt.close()
    print(f"Saved weight changes plot to {plot_path_params}")
    for name in classic_dist_to_start:
        print(f"  Classic {name} | Dist to Start: {classic_dist_to_start[name][-1]:.6f} | Path Length: {classic_path_length[name][-1]:.6f}")
    for name in fixed_dist_to_start:
        print(f"  Fixed {name}   | Dist to Start: {fixed_dist_to_start[name][-1]:.6f} | Path Length: {fixed_path_length[name][-1]:.6f}")

    # Generate test inputs with varying norms to plot norm transformations


    execution_time = time.time() - start_time
    print(f"\nEnd-to-end execution time: {execution_time:.3f} seconds")

    classic_steps_str = f"{classic_reached_step} steps" if classic_reached_step is not None else "Failed"
    fixed_steps_str = f"{fixed_reached_step} steps" if fixed_reached_step is not None else "Failed"
    generate_html_dashboard(
        num_layers=NUM_LAYERS,
        dim=dim,
        seq_len=num_rows,
        classic_steps=classic_steps_str,
        fixed_steps=fixed_steps_str,
        final_classic_lr=classic_lr,
        final_fixed_lr=fixed_lr,
        execution_time=execution_time
    )

if __name__ == '__main__':
    train()
