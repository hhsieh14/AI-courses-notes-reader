# 8. VC Dimension

Rademacher complexity measures a class by how well it can correlate with random noise. The Vapnik–Chervonenkis (VC) dimension measures it combinatorially: how many points can the class label in *every* possible way? This chapter defines shattering and VC dimension, computes it for intervals and half-spaces, and then connects it back to Rademacher complexity.

## 1. Classes of sets

Let $\mathcal{C}\subseteq 2^{\mathcal{Z}}$ be a collection of subsets of $\mathcal Z$. A binary classifier $f$ is identified with its positive region $C_f=\{z:f(z)=1\}$; for example, a linear classifier corresponds to a half-space. Label patterns a class can produce on a sample $S$ are then the traces $S\cap C$.

## 2. Shattering

### Definition 1

A finite set $S\subseteq\mathcal Z$ is **shattered** by $\mathcal C$ if every subset of $S$ is cut out by some member:

$$\left\lbrace S\cap C:C\in\mathcal{C}\right\rbrace=2^{S}.$$

For $|S|=m$ that is all $2^m$ labelings.

> [!TIP]
> **What shattering doesn't mean**
>
> A different $C$ may be used for each target subset. No single set has to realize all patterns.

## 3. Example: intervals on the line

$\mathcal{C}=\{[s,t]:s\leq t\}$. Two points $A<B$: all four subsets are achievable. Use an interval missing both, one around only $A$, one around only $B$, or one covering both. So every 2-point set is shattered.

## 4. VC dimension

### Definition 2

$$\mathrm{VC}(\mathcal{C})=\sup\left\lbrace \lvert S\rvert:S\text{ is shattered by }\mathcal{C}\right\rbrace .$$

It is monotone: $\mathcal C_1\subseteq\mathcal C_2\Rightarrow\mathrm{VC}(\mathcal C_1)\le\mathrm{VC}(\mathcal C_2)$, since anything $\mathcal C_1$ shatters, $\mathcal C_2$ shatters too.

To prove $\mathrm{VC}=d$ you need two things: **some** set of size $d$ that is shattered, and **no** set of size $d+1$ that is shattered.

## 5. Intervals have VC dimension 2

**Lower bound:** §3 shatters two points.

**Upper bound:** take any $A<B<C$. The subset $\{A,C\}$ can't be produced, because any interval containing $A$ and $C$ also contains $B$. So no 3-point set is shattered. $\square$

## 6. Half-spaces in the plane

$$\mathcal{C}=\left\lbrace \{(x_{1},x_{2}):w_{1}x_{1}+w_{2}x_{2}+b\geq 0\}:w_{1},w_{2},b\in\mathbb{R}\right\rbrace .$$

**Lower bound, VC ≥ 3.** Take three points **not on a line**. The empty set and the full set are easy. Any single vertex can be cut off from the other two by a line, and taking the complementary half-space selects any pair. So all 8 subsets are achievable.

The non-collinear condition is essential: for three collinear points, no half-space picks the two outer points without the middle one. The VC lower bound only needs *one* shattered set, so this is fine.

**Upper bound, VC ≤ 3.** One bad 4-point configuration isn't enough; we must show *no* 4 points can be shattered. Any 4 points in the plane fall into one of two cases:

- **One point lies inside the triangle of the other three** (or on a segment between two of them). Any half-space containing the three outer points is convex, so it contains the inner one too. Labeling the outer three positive and the inner one negative is impossible.
- **The four points are in convex position** (a convex quadrilateral). Then the two diagonals cross. A half-space containing both endpoints of one diagonal contains the whole diagonal, including the crossing point; the complementary half-space would have to contain the other diagonal and hence the same crossing point. Labeling one diagonal pair positive and the other negative is impossible.

(Both cases are instances of **Radon's theorem**: any $d+2$ points in $\mathbb R^d$ can be split into two groups whose convex hulls intersect, and a half-space and its complement can't separate groups with intersecting hulls.) $\square$

$$\mathrm{VC}(\text{half-spaces in }\mathbb R^2)=3 .$$

## 7. Half-spaces in $\mathbb R^d$

$$\mathrm{VC}(\mathcal{C}_{d})=d+1 .$$

**Lower bound:** the origin together with the $d$ standard basis vectors can be shattered. For any labeling $y\in\{\pm1\}^{d+1}$, take $b=y_0/2$ and $w_i=y_i$; then $\mathrm{sign}(w^\top e_i+b)=\mathrm{sign}(y_i+y_0/2)=y_i$ and $\mathrm{sign}(b)=y_0$.

**Upper bound:** Radon's theorem in $\mathbb R^d$, exactly as in the 2-D argument.

The VC dimension of affine classifiers equals their number of parameters, $d+1$. That is a coincidence of this class, not a rule. The class $\{\mathrm{sign}(\sin(\omega x))\}$ has a single parameter and infinite VC dimension.

## 8. From VC dimension to Rademacher complexity

> [!NOTE]
> **Beyond the lecture: the quantitative link**
>
> Let $\Pi_{\mathcal C}(n)=\max_{|S|=n}|\{S\cap C\}|$ be the number of distinct labelings on $n$ points (the growth function).
>
> 1. **Sauer–Shelah lemma:** if $\mathrm{VC}(\mathcal C)=d$, then $\Pi_{\mathcal C}(n)\le\sum_{i=0}^{d}\binom ni\le\left(\frac{en}{d}\right)^d$ for $n\ge d$. The count grows polynomially, not like $2^n$.
> 2. **Massart's finite-class lemma:** for a class with $N$ labelings on the sample, $\widehat{\mathfrak R}_n\le\sqrt{2\log N/n}$.
>
> Combining them, for $\pm1$-valued classifiers,
>
> $$\mathfrak R_n(\mathcal F)\le\sqrt{\frac{2d\log(en/d)}{n}},$$
>
> so the bound of Chapter 7 gives $R(f)\le\widehat R_n(f)+O\!\left(\sqrt{d\log n/n}\right)$ uniformly over the class. The VC dimension plays the role that $B\sqrt{\sum K(x_i,x_i)}$ played for kernel SVMs.

| | Rademacher complexity | VC dimension |
|---|---|---|
| Measures | ability to fit random signs on *this* distribution | worst-case number of realizable labelings |
| Depends on the data distribution | yes | no |
| Handles real-valued / margin classes | yes (e.g. RKHS balls) | binary classes only |
| Typical bound | $B/\sqrt n$ for norm-bounded kernel classes | $\sqrt{d\log n/n}$ |

The difference matters for SVMs. With an RBF kernel the VC dimension of the hypothesis space is infinite, yet the Rademacher bound through $\lVert w\rVert\le B$ is finite. **Margin/norm control, not dimension, explains why kernel SVMs generalize.**

## Summary

- A class shatters a set if it realizes every labeling of it; VC dimension is the largest shattered size.
- Intervals: 2. Half-spaces in $\mathbb R^d$: $d+1$ (upper bound via Radon's theorem).
- Sauer–Shelah plus Massart give $\mathfrak R_n=O(\sqrt{d\log n/n})$.
- For kernel methods, norm-based Rademacher bounds are the useful ones.

## Questions to test yourself

<details>
<summary>Why does shattering require all 2^|S| subsets?</summary>

VC dimension measures the largest sample on which the class can fit *any* labeling, including adversarial ones. Realizing only many patterns doesn't mean it can fit arbitrary noise.
</details>

<details>
<summary>Why doesn't one non-shattered 4-point set prove VC ≤ 3?</summary>

VC ≤ 3 is a claim about *every* 4-point set. The proof has to cover all configurations, which the two-case (Radon) argument does.
</details>

<details>
<summary>An RBF-kernel SVM has infinite VC dimension. Why can it still generalize?</summary>

The learned function lies in a norm-bounded RKHS ball, whose Rademacher complexity is $O(B/\sqrt n)$ regardless of the feature-space dimension.
</details>

---

[← Previous: Generalization and Rademacher Complexity](07_generalization_and_rademacher.md) · [Course map](../course_map.md) · [Next: Smooth Convex Optimization →](09_smooth_convex_optimization.md)
