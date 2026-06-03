# /// script
# dependencies = [
#   "torch",
#   "numpy",
#   "matplotlib",
# ]
# ///

import torch
import numpy as np
import matplotlib.pyplot as plt
import copy
import sys
from attention_solve import MultiLayerSelfAttention, generate_orthogonal_rows_batch, step_classic, reconstruction_loss

# Seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

dim, num_rows, batch_size = 32, 8, 32

# Instantiate teacher
teacher = MultiLayerSelfAttention(dim).requires_grad_(False)

# Instantiate student init state (identity matrices)
student_init = MultiLayerSelfAttention(dim)
with torch.no_grad():
    for p in student_init.parameters():
        p.copy_(torch.eye(dim))
student_initial_state = copy.deepcopy(student_init.state_dict())

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
    init_loss_b = reconstruction_loss(student_init(batch_x_b), batch_y_b).item()

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
        model = MultiLayerSelfAttention(dim)
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
fig, ax = plt.subplots(figsize=(10, 7), dpi=150)
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

ax.set_xlabel('Learning Rate (log scale)', fontsize=12)
ax.set_ylabel('Mean Squared Error (MSE) Loss (log scale)', fontsize=12)
ax.set_title('Learning Rate Sensitivity: Loss vs Learning Rate\n(After 1 and 10 steps starting from Identity Initialization)', fontsize=13, fontweight='bold')
ax.legend(frameon=True, facecolor='white', framealpha=0.9, fontsize=10)

plt.tight_layout()
plot_path = "lr_sensitivity_plot.png"
plt.savefig(plot_path)
print(f"Saved learning rate graph to {plot_path}")
