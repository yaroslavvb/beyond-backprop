"""
Target-propagation training of a tanh network, ported from Wolfram Language.

The task: learn a stack of `depth` 2x2 tanh layers that rotate a point cloud
shaped like the letter "A" by 90 degrees. Instead of gradient descent, each
layer is fit by *inverting* the desired output through a pseudo-inverse
(a "beyond backprop" / target-propagation scheme):

    forward(A, W)    = tanh(A @ W^T)
    backwardA(B,A,W) = A - arctanh(clip(forward(A,W) - B)) @ pinv(W)^T   (target for the input)
    backwardW(B,A,W) = W - arctanh(clip(forward(A,W) - B))^T @ pinv(A^T) (weight correction)

Each outer iteration runs a full forward/backward sweep, then updates exactly
ONE layer (layer `layerIdx`), sequentially layers 1..depth. The 6x5 grid shows,
for iteration c (column), the activations A[1..6] (rows) of the network *before*
that iteration's update -- matching the loss printed in the column header.

Note on reproducibility: the original uses Mathematica's SeedRandom[1]. NumPy's
RNG and normal-variate algorithm differ, so the random initial weights -- and
thus the exact loss values -- cannot be byte-for-byte identical. The algorithm,
structure, and qualitative result are faithful; `SEED` below is chosen to land
the losses close to the reference figure (26.100, 7.190, 2.630, 1.190, 0.715).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import font_manager
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Problem constants (mirrors the Wolfram script)
# ---------------------------------------------------------------------------
SEED = 89               # stand-in for Mathematica's SeedRandom[1]
D = 2                   # d : point / hidden dimension
DEPTH = 5               # depth : number of tanh layers
LEARNING_RATE = 0.4     # learningRate
FONT_SIZE = 38          # rasterization size for the letter "A"
DOMAIN = (-0.999, 0.999)
PLOT_RANGE = 0.5        # pr = {{-1/2,1/2},{-1/2,1/2}}


# ---------------------------------------------------------------------------
# Letter-"A" point cloud  (generateLetterA)
# ---------------------------------------------------------------------------
def generate_letter_a(size=FONT_SIZE, img_size=50):
    """Rasterize an "A", take black-pixel positions, normalize each axis to
    [-0.5, 0.5], then rotate by -pi/2 (clockwise)."""
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
    if font is None:  # cross-platform fallback: any serif face matplotlib knows
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
        draw.text(((img_size - w) / 2 - bbox[0], (img_size - h) / 2 - bbox[1]),
                  "A", fill=0, font=font)
    else:  # last-resort hand-drawn "A"
        draw.line([(15, 40), (25, 10)], fill=0, width=4)
        draw.line([(25, 10), (35, 40)], fill=0, width=4)
        draw.line([(18, 30), (32, 30)], fill=0, width=3)

    pixels = np.array(image)
    ys, xs = np.where(pixels < 128)
    # flip y so it grows upward, matching Mathematica image coordinates
    X = np.column_stack((xs, img_size - ys)).astype(float)

    min_xy = X.min(axis=0)
    max_xy = X.max(axis=0)
    scaled = (X - min_xy) / (max_xy - min_xy) - 0.5      # each axis -> [-0.5, 0.5]

    R = np.array([[0.0, 1.0], [-1.0, 0.0]])             # RotationMatrix[-Pi/2]
    return scaled @ R.T                                 # apply R to every row


# ---------------------------------------------------------------------------
# Core linear-algebra primitives
# ---------------------------------------------------------------------------
def orthogonalize(m):
    """Gram-Schmidt orthonormalization of the ROWS of m (= Wolfram Orthogonalize)."""
    q = np.zeros_like(m, dtype=float)
    for i in range(m.shape[0]):
        v = m[i].astype(float).copy()
        for j in range(i):
            v -= np.dot(q[j], m[i]) * q[j]
        n = np.linalg.norm(v)
        q[i] = v / n if n > 1e-12 else v
    return q


def forward(A, W):
    return np.tanh(A @ W.T)


def clip(mat):
    return np.clip(mat, DOMAIN[0], DOMAIN[1])


def backwardA(B, A, W):
    target = forward(A, W) - B
    Asol = np.arctanh(clip(target)) @ np.linalg.pinv(W).T
    return A - Asol


def backwardW(B, A, W):
    target = forward(A, W) - B
    Wsol = np.arctanh(clip(target)).T @ np.linalg.pinv(A.T)
    return W - Wsol


def norm2(A):
    return float(np.sum(A * A))


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def train():
    np.random.seed(SEED)

    X = generate_letter_a()
    target_mat = np.array([[0.0, -1.0], [1.0, 0.0]])    # RotationMatrix[Pi/2]
    target = X @ target_mat.T                           # targetFunc[X]

    # Ws = Orthogonalize[I + 0.4 Orthogonalize[randn[d,d]]], one per layer
    Ws = np.array([
        orthogonalize(np.eye(D) + 0.4 * orthogonalize(np.random.normal(size=(D, D))))
        for _ in range(DEPTH)
    ])

    def update(Ws_curr, dWs_curr, dims):
        out = Ws_curr.copy()
        for j in dims:
            out[j] = Ws_curr[j] - LEARNING_RATE * dWs_curr[j]
        return out

    losses, cols = [], []
    for layer_idx in range(DEPTH):
        # --- forwardBackward ---
        As = [X]                                        # FoldList[forward, X, Ws]
        for W in Ws:
            As.append(forward(As[-1], W))

        Bs = [As[-1] - target]                          # output error
        for i in range(DEPTH - 1, -1, -1):              # propagate targets backward
            Bs.append(backwardA(Bs[-1], As[i], Ws[i]))

        dWs = np.array([
            backwardW(Bs[DEPTH - 1 - i], As[i], Ws[i])  # Bs[[depth+1-i]] (1-indexed)
            for i in range(DEPTH)
        ])

        # loss and column are recorded with the CURRENT (pre-update) weights.
        # In the source, update[Ws,dWs,{i}] runs with i==0 -> a no-op, so the
        # plotted column is exactly the current activations `As`.
        losses.append(norm2(As[-1] - target))
        cols.append(As)

        # update exactly one layer, then move on
        Ws = update(Ws, dWs, [layer_idx])

    return cols, losses


# ---------------------------------------------------------------------------
# Visualization  (TableForm-style 6x5 grid)
# ---------------------------------------------------------------------------
def plot(cols, losses, out_path="simulation_results.png"):
    n_rows, n_cols = DEPTH + 1, DEPTH          # A[1..6] rows, 5 loss columns
    fig, axs = plt.subplots(n_rows, n_cols, figsize=(13, 9.5), dpi=150)
    fig.patch.set_facecolor("white")

    blue = "#3f9fd6"
    for c in range(n_cols):
        for r in range(n_rows):
            ax = axs[r, c]
            pts = cols[c][r]
            ax.scatter(pts[:, 0], pts[:, 1], color=blue, s=8, alpha=0.9,
                       edgecolors="none")
            ax.set_xlim(-PLOT_RANGE - 0.05, PLOT_RANGE + 0.05)
            ax.set_ylim(-PLOT_RANGE - 0.05, PLOT_RANGE + 0.05)
            ax.set_aspect("equal")
            ax.axis("off")

    plt.subplots_adjust(left=0.10, right=0.97, top=0.88, bottom=0.04,
                        hspace=0.08, wspace=0.08)
    fig.canvas.draw()

    tl, br = axs[0, 0].get_position(), axs[n_rows - 1, n_cols - 1].get_position()
    y_top, x_left = tl.y1 + 0.02, tl.x0 - 0.045
    x_right, y_bottom = br.x1 + 0.01, br.y0 - 0.02

    # TableForm rules: one horizontal line under the header, one vertical line
    fig.add_artist(plt.Line2D([x_left, x_right], [y_top, y_top],
                              color="#333333", lw=1.2, transform=fig.transFigure))
    fig.add_artist(plt.Line2D([x_left, x_left], [y_bottom, y_top],
                              color="#333333", lw=1.2, transform=fig.transFigure))

    for c in range(n_cols):                    # column headers: loss values
        bb = axs[0, c].get_position()
        fig.text((bb.x0 + bb.x1) / 2, y_top + 0.012, f"loss={losses[c]:.3f}",
                 ha="center", va="bottom", fontsize=12, family="monospace",
                 color="#111111")
    for r in range(n_rows):                    # row headers: A[1]..A[6]
        bb = axs[r, 0].get_position()
        fig.text(x_left - 0.015, (bb.y0 + bb.y1) / 2, f"A[{r + 1}]",
                 ha="right", va="center", fontsize=12, family="monospace",
                 color="#111111")

    fig.savefig(out_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return out_path


def main():
    cols, losses = train()
    print("Losses (pre-update, per iteration):")
    for i, l in enumerate(losses, 1):
        print(f"  iter {i}: loss = {l:.3f}")
    path = plot(cols, losses)
    print(f"Saved figure to {os.path.abspath(path)}")


if __name__ == "__main__":
    main()
