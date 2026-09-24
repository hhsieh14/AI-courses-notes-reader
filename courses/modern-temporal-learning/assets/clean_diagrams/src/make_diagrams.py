"""Redraw the LSTM-cell and transformer-overview diagrams.

Run from this folder:  python make_diagrams.py
Writes ../lstm_memory_gates.png and ../transformer_overview.png
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Ellipse, FancyArrowPatch

INK, MUTED = "#1f2937", "#4b5563"
C = dict(red="#dc2626", green="#16a34a", orange="#ea580c", blue="#2563eb",
         purple="#7c3aed", gray="#64748b")
FILL = dict(red="#fee2e2", green="#dcfce7", orange="#ffedd5", blue="#dbeafe",
            purple="#ede9fe", gray="#f1f5f9", yellow="#fef9c3")


def box(ax, x, y, w, h, text, color="gray", fill=None, size=11, weight="bold", sub=None):
    ax.add_patch(FancyBboxPatch((x - w / 2, y - h / 2), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=FILL[fill or color], ec=C.get(color, color), lw=1.8))
    if sub:
        ax.text(x, y + 0.13, text, ha="center", va="center", fontsize=size, weight=weight, color=INK)
        ax.text(x, y - 0.17, sub, ha="center", va="center", fontsize=size - 2, color=MUTED)
    else:
        ax.text(x, y, text, ha="center", va="center", fontsize=size, weight=weight, color=INK)


def node(ax, x, y, text, color, r=0.26):
    ax.add_patch(Ellipse((x, y), 2 * r, 2 * r, fc="white", ec=C[color], lw=2.2, zorder=3))
    ax.text(x, y, text, ha="center", va="center", fontsize=13, weight="bold", color=INK, zorder=4)


def arrow(ax, p, q, color=INK, lw=1.8, rad=0.0, style="-|>", ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=14, color=color, lw=lw,
                                 connectionstyle=f"arc3,rad={rad}", linestyle=ls, zorder=2,
                                 shrinkA=0, shrinkB=0))


def lstm():
    fig, ax = plt.subplots(figsize=(12, 6.6))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6.6); ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)
    ax.text(6, 6.3, "LSTM cell: an additive memory line with three gates", ha="center",
            fontsize=16, weight="bold", color=INK)
    ax.text(6, 5.93, r"$c_t=f_t\odot c_{t-1}+i_t\odot\tilde c_t \qquad h_t=o_t\odot\tanh(c_t)$",
            ha="center", fontsize=13, color=INK)

    Y = 5.0                                     # cell-state line
    ax.plot([0.7, 11.1], [Y, Y], color=C["purple"], lw=4, zorder=1)
    arrow(ax, (10.9, Y), (11.4, Y), C["purple"], lw=4)
    ax.text(0.65, Y + 0.28, r"$c_{t-1}$", fontsize=14, color=C["purple"])
    ax.text(11.15, Y + 0.28, r"$c_t$", fontsize=14, color=C["purple"])
    node(ax, 2.8, Y, "×", "red")
    node(ax, 5.8, Y, "+", "green")

    # gates
    G = 2.2
    box(ax, 2.8, G, 1.6, 0.8, "forget gate", "red", sub=r"$f_t=\sigma(\cdot)$")
    box(ax, 4.8, G, 1.6, 0.8, "input gate", "green", sub=r"$i_t=\sigma(\cdot)$")
    box(ax, 6.8, G, 1.6, 0.8, "candidate", "orange", sub=r"$\tilde c_t=\tanh(\cdot)$")
    box(ax, 8.8, G, 1.6, 0.8, "output gate", "blue", sub=r"$o_t=\sigma(\cdot)$")

    arrow(ax, (2.8, G + 0.4), (2.8, Y - 0.26), C["red"])
    node(ax, 5.8, 3.55, "×", "green", r=0.22)
    arrow(ax, (4.8, G + 0.4), (5.62, 3.42), C["green"])
    arrow(ax, (6.8, G + 0.4), (5.98, 3.42), C["orange"])
    arrow(ax, (5.8, 3.77), (5.8, Y - 0.26), C["green"])
    ax.text(5.95, 4.3, r"$i_t\odot\tilde c_t$", fontsize=12, color=C["green"])

    # output branch: tanh only on the way to h_t
    ax.plot([10.1, 10.1], [Y, 4.45], color=C["purple"], lw=2)
    node(ax, 10.1, 4.2, "", "blue", r=0.25)
    ax.text(10.1, 4.2, "tanh", ha="center", va="center", fontsize=9.5, weight="bold", color=INK, zorder=5)
    arrow(ax, (10.1, 3.95), (10.1, 3.42), C["blue"])
    node(ax, 10.1, 3.2, "×", "blue", r=0.22)
    arrow(ax, (9.6, G + 0.4), (9.93, 3.03), C["blue"])
    arrow(ax, (10.32, 3.2), (11.4, 3.2), C["blue"], lw=2.2)
    ax.text(11.1, 3.38, r"$h_t$", fontsize=14, color=C["blue"])

    # input bus
    box(ax, 1.1, 0.75, 1.7, 0.6, r"$[\,x_t;\;h_{t-1}\,]$", "gray", weight="normal", size=13)
    ax.plot([1.95, 8.8], [0.75, 0.75], color=MUTED, lw=1.8)
    for x in (2.8, 4.8, 6.8, 8.8):
        arrow(ax, (x, 0.75), (x, G - 0.4), MUTED)
    ax.text(6, 0.2, "Along the top line the old memory is only scaled and added to: no weight matrix, no squashing.",
            ha="center", fontsize=10.5, color=MUTED)
    fig.savefig("../lstm_memory_gates.png", dpi=150, bbox_inches="tight", facecolor="white")


def transformer():
    fig, ax = plt.subplots(figsize=(12, 8.2))
    ax.set_xlim(0, 12); ax.set_ylim(0, 8.2); ax.axis("off")
    fig.subplots_adjust(0, 0, 1, 1)
    ax.text(6, 7.95, "Transformer encoder–decoder", ha="center", fontsize=16, weight="bold", color=INK)
    ax.text(6, 7.62, "Attention mixes information across positions; the feed-forward layer transforms each position on its own.",
            ha="center", fontsize=10.5, color=MUTED)

    def stack(x, layers, y0, title, n_label, side=-1):
        top = y0 + 0.62 * (len(layers) - 1)
        ax.add_patch(FancyBboxPatch((x - 1.75, y0 - 0.45), 3.5, top - y0 + 0.9,
                                    boxstyle="round,pad=0.02,rounding_size=0.15",
                                    fc="#f8fafc", ec="#cbd5e1", lw=1.5, zorder=0))
        ax.text(x + side * 2.0, (y0 + top) / 2, n_label, ha="right" if side < 0 else "left", va="center", fontsize=13,
                weight="bold", color=MUTED)
        ax.text(x - 1.7, top + 0.58, title, ha="left", fontsize=12.5, weight="bold", color=INK)
        ys = []
        for i, (label, color) in enumerate(layers):
            y = y0 + 0.62 * i
            box(ax, x, y, 3.0, 0.44, label, color, size=10.5)
            if i:
                arrow(ax, (x, ys[-1] + 0.22), (x, y - 0.22), MUTED, lw=1.4)
            ys.append(y)
        return ys

    # encoder
    ex = 3.0
    box(ax, ex, 1.0, 3.0, 0.5, "input embedding\n+ positional encoding", "blue", size=9.5)
    enc = stack(ex, [("multi-head self-attention", "purple"), ("add & norm", "gray"),
                     ("feed-forward (per position)", "green"), ("add & norm", "gray")],
                2.1, "Encoder stack", "× N")
    arrow(ax, (ex, 1.25), (ex, enc[0] - 0.22), MUTED, lw=1.4)
    for a, b in ((0, 1), (2, 3)):                  # residual paths
        arrow(ax, (ex + 1.5, enc[a] - 0.3), (ex + 1.5, enc[b]), C["red"], lw=1.3, rad=-0.6, ls="--")
    box(ax, ex, 5.35, 3.0, 0.5, "encoder output (memory)", "orange", fill="yellow", size=10.5)
    arrow(ax, (ex, enc[-1] + 0.22), (ex, 5.1), MUTED, lw=1.4)

    # decoder
    dx = 9.0
    box(ax, dx, 1.0, 3.0, 0.5, "output embedding (shifted right)\n+ positional encoding", "blue", size=9.5)
    dec = stack(dx, [("masked self-attention", "purple"), ("add & norm", "gray"),
                     ("cross-attention", "purple"), ("add & norm", "gray"),
                     ("feed-forward (per position)", "green"), ("add & norm", "gray")],
                2.1, "Decoder stack", "× N", side=1)
    arrow(ax, (dx, 1.25), (dx, dec[0] - 0.22), MUTED, lw=1.4)
    for a, b in ((0, 1), (2, 3), (4, 5)):
        arrow(ax, (dx + 1.5, dec[a] - 0.3), (dx + 1.5, dec[b]), C["red"], lw=1.3, rad=-0.6, ls="--")
    box(ax, dx, 6.6, 3.0, 0.5, "linear + softmax (or regression head)", "orange", size=9.5)
    arrow(ax, (dx, dec[-1] + 0.22), (dx, 6.35), MUTED, lw=1.4)

    # memory → cross-attention (K, V)
    arrow(ax, (ex + 1.5, 5.35), (dx - 1.5, dec[2]), C["orange"], lw=2.2, rad=0.25)
    ax.text(6.0, 5.75, "keys & values from the encoder,\nqueries from the decoder", ha="center",
            fontsize=10, color=C["orange"], weight="bold")
    ax.text(6.0, 0.35, "Dashed red arrows: residual connections.  Training: all positions in parallel (causal mask).  "
            "Inference: one output at a time.", ha="center", fontsize=9.5, color=MUTED)
    fig.savefig("../transformer_overview.png", dpi=150, bbox_inches="tight", facecolor="white")


if __name__ == "__main__":
    lstm()
    transformer()
