"""Regenerate the CSE 575 figures that are drawn from data.

Run from this folder:  python make_diagrams.py
Outputs go to the parent folder (docs/assets/diagrams/).
Every figure is computed, not sketched, so the geometry is exact.
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC

OUT = Path(__file__).resolve().parent.parent
BLUE, RED, INK, GREY, PURPLE = "#2b6cb0", "#c0392b", "#1b2230", "#8a94a6", "#6b46c1"
plt.rcParams.update({"font.size": 10, "axes.edgecolor": GREY, "axes.labelcolor": INK,
                     "xtick.color": GREY, "ytick.color": GREY, "svg.fonttype": "none"})


def _clean(ax):
    ax.set_xticks([]); ax.set_yticks([])
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def svm_figure():
    rng = np.random.default_rng(3)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))

    # 1. hard margin: separable data, exact solution from a large-C linear SVM
    Xp = rng.normal([2.2, 2.2], 0.45, (12, 2)); Xn = rng.normal([0.3, 0.4], 0.45, (12, 2))
    X = np.vstack([Xp, Xn]); y = np.r_[np.ones(12), -np.ones(12)]
    clf = SVC(kernel="linear", C=1e6).fit(X, y)
    w, b = clf.coef_[0], clf.intercept_[0]
    ax = axes[0]
    xs = np.linspace(-1, 3.6, 50)
    for off, ls in ((0, "-"), (1, "--"), (-1, "--")):
        ax.plot(xs, (off - b - w[0] * xs) / w[1], ls, color=PURPLE, lw=2 if off == 0 else 1.2)
    ax.scatter(*Xp.T, c=RED, s=28, zorder=3); ax.scatter(*Xn.T, c=BLUE, s=28, zorder=3)
    sv = clf.support_vectors_
    ax.scatter(*sv.T, s=160, facecolors="none", edgecolors=INK, lw=1.4, zorder=4)
    # margin arrow: between the two margin lines, along the normal
    n = w / np.linalg.norm(w); p0 = np.array([1.2, (-b - w[0] * 1.2) / w[1]])
    m = 1 / np.linalg.norm(w)
    ax.annotate("", p0 + m * n, p0 - m * n, arrowprops=dict(arrowstyle="<->", color="#067647", lw=1.6))
    ax.text(*(p0 - [0.95, -0.15]), r"$2/\|w\|$", color="#067647", fontsize=11)
    ax.set_xlim(-1, 3.6); ax.set_ylim(-1, 3.6); ax.set_aspect("equal")
    ax.set_title("1. Hard margin", loc="left", color=INK)
    ax.text(-0.9, -0.9, "circled = support vectors,\nexactly on the dashed lines", fontsize=8.5, color=GREY)
    _clean(ax)

    # 2. soft margin: overlapping classes, moderate C
    Xp = rng.normal([2.0, 2.0], 0.7, (18, 2)); Xn = rng.normal([0.6, 0.6], 0.7, (18, 2))
    X = np.vstack([Xp, Xn]); y = np.r_[np.ones(18), -np.ones(18)]
    clf = SVC(kernel="linear", C=1.0).fit(X, y)
    w, b = clf.coef_[0], clf.intercept_[0]
    ax = axes[1]
    xs = np.linspace(-1.2, 4, 50)
    for off, ls in ((0, "-"), (1, "--"), (-1, "--")):
        ax.plot(xs, (off - b - w[0] * xs) / w[1], ls, color=PURPLE, lw=2 if off == 0 else 1.2)
    ax.scatter(*Xp.T, c=RED, s=28, zorder=3); ax.scatter(*Xn.T, c=BLUE, s=28, zorder=3)
    margins = y * (X @ w + b); xi = np.maximum(0, 1 - margins)
    viol = xi > 1
    ax.scatter(*X[viol].T, s=170, facecolors="none", edgecolors="#b54708", lw=1.6, zorder=4)
    ax.set_xlim(-1.2, 4); ax.set_ylim(-1.2, 4); ax.set_aspect("equal")
    ax.set_title("2. Soft margin (C = 1)", loc="left", color=INK)
    ax.text(-1.1, -1.1, r"orange rings: $\xi_i>1$ (misclassified)", fontsize=8.5, color="#b54708")
    _clean(ax)

    # 3. feature map x -> (x, x^2)
    ax = axes[2]
    x_in = np.r_[rng.uniform(-0.8, 0.8, 8)]; x_out = np.r_[rng.uniform(1.3, 2.2, 4), -rng.uniform(1.3, 2.2, 4)]
    ax.scatter(x_in, np.zeros_like(x_in) - 1.6, c=BLUE, s=28); ax.scatter(x_out, np.zeros_like(x_out) - 1.6, c=RED, s=28)
    ax.scatter(x_in, x_in ** 2, c=BLUE, s=28); ax.scatter(x_out, x_out ** 2, c=RED, s=28)
    ax.axhline(1.1, color=PURPLE, lw=2)
    ax.axhline(-1.6, color=GREY, lw=0.8)
    ax.text(-2.4, -1.4, "input: not separable on a line", fontsize=8.5, color=GREY)
    ax.text(-2.4, 1.25, r"after $\phi(x)=(x,x^2)$: a line separates", fontsize=8.5, color=PURPLE)
    ax.set_xlim(-2.5, 2.5); ax.set_ylim(-2, 5)
    ax.set_title("3. Feature map / kernel", loc="left", color=INK)
    _clean(ax)

    fig.tight_layout()
    fig.savefig(OUT / "08_svm_margin_slack_kernels.svg", bbox_inches="tight")
    plt.close(fig)


def pca_figure():
    rng = np.random.default_rng(0)
    n = 200
    X = rng.multivariate_normal([0, 0], [[3.0, 1.6], [1.6, 1.3]], n)
    X -= X.mean(0)
    evals, evecs = np.linalg.eigh(np.cov(X.T, bias=True))
    order = evals.argsort()[::-1]; evals, evecs = evals[order], evecs[:, order]
    if evecs[0, 0] < 0: evecs[:, 0] *= -1          # eigenvector sign is arbitrary; point u1 up-right
    if evecs[1, 1] < 0: evecs[:, 1] *= -1
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    ax = axes[0]
    ax.scatter(*X.T, s=8, c=GREY, alpha=0.6)
    for j, (c, name) in enumerate(((RED, "u_1"), (BLUE, "u_2"))):
        v = evecs[:, j] * 2 * np.sqrt(evals[j])
        ax.annotate("", v, (0, 0), arrowprops=dict(arrowstyle="-|>", color=c, lw=2))
        ax.text(*(v + ([0.3, -0.6] if j == 0 else [-0.2, 0.35])), rf"${name}$, $\lambda_{j+1}={evals[j]:.2f}$", color=c, ha="left" if j == 0 else "right",
                bbox=dict(fc="white", ec="none", alpha=0.85, pad=1))
    ax.set_aspect("equal"); ax.axhline(0, color=GREY, lw=0.5); ax.axvline(0, color=GREY, lw=0.5)
    ax.set_title("Centered data and principal directions", loc="left", color=INK)
    ax.set_xlabel(r"$x_1$"); ax.set_ylabel(r"$x_2$")
    # cumulative explained variance for a 10-D example with decaying spectrum
    lam = np.array([5.0, 2.5, 1.2, 0.6, 0.3, 0.15, 0.1, 0.07, 0.05, 0.03])
    R = np.cumsum(lam) / lam.sum(); k = int(np.argmax(R >= 0.95)) + 1
    ax = axes[1]
    ax.plot(range(1, 11), R, "o-", color=PURPLE)
    ax.axhline(0.95, ls="--", color=RED, lw=1); ax.axvline(k, ls=":", color=RED, lw=1)
    ax.text(k + 0.2, 0.55, f"k = {k}\n(first $R_k \\geq 0.95$)", color=RED)
    ax.set_ylim(0.4, 1.02); ax.set_xticks(range(1, 11))
    ax.set_xlabel("number of components k"); ax.set_ylabel(r"cumulative explained variance $R_k$")
    ax.set_title("Choosing k", loc="left", color=INK)
    for a in axes:
        a.spines["top"].set_visible(False); a.spines["right"].set_visible(False)
    fig.tight_layout(); fig.savefig(OUT / "11_pca_projection.svg", bbox_inches="tight"); plt.close(fig)


def nn_figure():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [1, 1.15]})
    ax = axes[0]; ax.axis("off"); ax.set_xlim(0, 10); ax.set_ylim(0, 6)
    ins = ["$x_1$", "$x_2$", "$x_3$", "1"]
    for i, lab in enumerate(ins):
        yy = 5 - i * 1.3
        ax.text(0.5, yy, lab, ha="center", va="center", fontsize=12)
        ax.annotate("", (5.1, 3.0), (0.9, yy), arrowprops=dict(arrowstyle="-|>", color=GREY))
        ax.text(2.7, yy - 0.05 + (3.0 - yy) * 0.45, ["$w_1$", "$w_2$", "$w_3$", "$b$"][i], color=BLUE)
    circ = plt.Circle((6.3, 3.0), 1.2, fc="#fff7e6", ec="#b54708", lw=1.5); ax.add_patch(circ)
    ax.text(6.3, 3.35, r"$z=b+\sum_i w_ix_i$", ha="center", fontsize=9.5)
    ax.text(6.3, 2.55, r"$a=\sigma(z)$", ha="center", fontsize=10, color="#b54708")
    ax.annotate("", (9.6, 3.0), (7.5, 3.0), arrowprops=dict(arrowstyle="-|>", color=INK))
    ax.text(9.7, 3.0, "$a$", va="center", fontsize=12)
    ax.set_title("One neuron", loc="left", color=INK)

    ax = axes[1]
    z = np.linspace(-5, 5, 400)
    ax.plot(z, 1 / (1 + np.exp(-z)), color=BLUE, lw=2, label=r"sigmoid, $\sigma(0)=0.5$")
    ax.plot(z, np.tanh(z), "--", color=PURPLE, lw=2, label="tanh")
    ax.plot(z, np.maximum(0, z), color="#b54708", lw=2, label="ReLU")
    ax.plot(z, np.where(z > 0, z, 0.1 * z), ":", color=RED, lw=2, label=r"leaky ReLU, $\alpha=0.1$")
    ax.axhline(0, color=GREY, lw=0.6); ax.axvline(0, color=GREY, lw=0.6)
    ax.scatter([0], [0.5], color=BLUE, zorder=5, s=20)
    ax.set_ylim(-1.3, 3); ax.set_xlabel("z"); ax.set_ylabel("activation")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.set_title("Activation functions", loc="left", color=INK)
    fig.tight_layout(); fig.savefig(OUT / "12_nn_neuron_activations.svg", bbox_inches="tight"); plt.close(fig)


def elbo_figure():
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.bar([0], [3.0], width=0.5, color="#2f855a", label="ELBO")
    ax.bar([1.4], [4.6], width=0.5, color=BLUE)
    ax.plot([0.25, 1.15], [3.0, 3.0], ls="--", color="#2f855a", lw=1)
    ax.annotate("", (0.7, 4.6), (0.7, 3.0), arrowprops=dict(arrowstyle="<->", color=RED, lw=1.6))
    ax.plot([0.55, 1.15], [4.6, 4.6], ls="--", color=BLUE, lw=1)
    ax.text(0.62, 3.8, r"gap $= D_{KL}(Q(z)\,\|\,p(z\mid x;\theta))\geq 0$", color=RED, va="center", ha="right")
    ax.text(0, 3.15, r"ELBO$(Q,\theta)$", ha="center", color="#2f855a")
    ax.text(1.4, 4.75, r"$\log p(x;\theta)$", ha="center", color=BLUE)
    ax.text(0.7, -0.75, "E-step: set $Q=p(z\\mid x;\\theta)$, the gap closes.   M-step: maximize the ELBO over $\\theta$.",
            ha="center", fontsize=9.5, color=INK)
    ax.set_xlim(-0.6, 2.2); ax.set_ylim(-1, 5.4); ax.axis("off")
    fig.tight_layout(); fig.savefig(OUT / "10_elbo_interpretation.svg", bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    svm_figure(); pca_figure(); nn_figure(); elbo_figure()
    print("wrote figures to", OUT)
