"""Regenerate computed figures for the Statistical Learning Theory notes.

Run from this folder:  python make_diagrams.py
"""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVC

OUT = Path(__file__).resolve().parent.parent
BLUE, RED, INK, GREY, PURPLE = "#2b6cb0", "#c0392b", "#1b2230", "#8a94a6", "#6b46c1"
plt.rcParams.update({"font.size": 10, "svg.fonttype": "none"})


def hard_margin():
    rng = np.random.default_rng(7)
    Xp = rng.normal([2.4, 2.2], 0.5, (10, 2)); Xn = rng.normal([0.2, 0.4], 0.5, (10, 2))
    X = np.vstack([Xp, Xn]); y = np.r_[np.ones(10), -np.ones(10)]
    clf = SVC(kernel="linear", C=1e6).fit(X, y)
    w, b = clf.coef_[0], clf.intercept_[0]
    fig, ax = plt.subplots(figsize=(6.5, 4.6))
    xs = np.linspace(-1.2, 4, 50)
    for off, ls, lab in ((0, "-", r"$w^\top x+b=0$"), (1, "--", r"$w^\top x+b=+1$"), (-1, "--", r"$w^\top x+b=-1$")):
        ax.plot(xs, (off - b - w[0] * xs) / w[1], ls, color=PURPLE, lw=2 if off == 0 else 1.2, label=lab)
    ax.scatter(*Xp.T, c=RED, s=30, zorder=3, label="$y=+1$"); ax.scatter(*Xn.T, c=BLUE, s=30, zorder=3, label="$y=-1$")
    ax.scatter(*clf.support_vectors_.T, s=170, facecolors="none", edgecolors=INK, lw=1.4, zorder=4, label="support vectors")
    n = w / np.linalg.norm(w); m = 1 / np.linalg.norm(w)
    p0 = np.array([1.0, (-b - w[0] * 1.0) / w[1]])
    ax.annotate("", p0 + m * n, p0 - m * n, arrowprops=dict(arrowstyle="<->", color="#067647", lw=1.6))
    ax.text(*(p0 - [1.0, -0.1]), r"margin $2/\|w\|$", color="#067647")
    ax.set_xlim(-1.2, 4); ax.set_ylim(-1.2, 4); ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    ax.legend(frameon=True, fontsize=8, loc="lower right", framealpha=0.9)
    ax.set_title("Hard-margin SVM (exact solution)", loc="left")
    fig.tight_layout(); fig.savefig(OUT / "hard_margin_svm.svg", bbox_inches="tight"); plt.close(fig)


if __name__ == "__main__":
    hard_margin()
    print("wrote figures to", OUT)
