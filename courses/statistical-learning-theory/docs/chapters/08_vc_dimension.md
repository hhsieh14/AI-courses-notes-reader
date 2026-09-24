---
course: "Statistical Learning Theory"
chapter: "08"
title: "VC Dimension"
source_pages: "598SLT.pdf, pp. 55-56"
status: "consolidated v1.0"
---

# VC Dimension

**Source:** 598SLT.pdf, pp. 55-56.

## 1. A combinatorial measure of function-class complexity

The previous chapter measures the richness of a function class through Rademacher complexity. The source now introduces another complexity measure: the **Vapnik-Chervonenkis dimension**, usually abbreviated as the **VC dimension**.

The course first formulates the definition for a class of subsets. Let

$$\mathcal{C}\subseteq 2^{\mathcal{Z}}$$

be a collection of subsets of a space $\mathcal{Z}$.

When $\mathcal{C}$ comes from binary prediction functions, each predictor can be identified with the region on which it predicts the positive class. For example, a linear binary classifier determines one half-space, so a class of linear classifiers can be studied as a class of subsets.

> [!TIP]
> **Why switch from functions to sets?**
>
> For a binary-valued function $f$, consider its positive region $C_{f}=\{z\in\mathcal{Z}:f(z)=1\}$. Questions about the label patterns produced by the function class become questions about which intersections $S\cap C_{f}$ can be formed on a finite sample $S$.

## 2. Shattering

### Definition 1: A finite set is shattered

Let $S\subseteq\mathcal{Z}$ be finite. The set $S$ is **shattered** by $\mathcal{C}$ if, for every subset $S'\subseteq S$, there exists a set $C\in\mathcal{C}$ such that

$$S'=S\cap C.$$

Equivalently, the collection of traces of $\mathcal{C}$ on $S$ is the complete power set of $S$:

$$\left\lbrace S\cap C:C\in\mathcal{C}\right\rbrace=2^{S}.$$

Thus, if $S$ contains $m$ points, shattering requires the class to realize all

$$2^{m}$$

possible inclusion patterns on those points.

> [!TIP]
> **What shattering does not mean**
>
> Shattering does not mean that one single set $C$ realizes all subsets simultaneously. A different member of $\mathcal{C}$ may be selected for each target subset $S'\subseteq S$.

## 3. Example: intervals on the real line

Let $\mathcal{Z}=\mathbb{R}$ and let $\mathcal{C}$ be the class of all closed intervals:

$$\mathcal{C}=\left\lbrace [s,t]:s,t\in\mathbb{R},\;s\leq t\right\rbrace.$$

Take two distinct points

$$S=\{A,B\},\qquad A<B.$$

The four subsets of $S$ can all be produced by intersecting $S$ with an interval:

- $\varnothing$: choose an interval that contains neither $A$ nor $B$;
- $\{A\}$: choose a sufficiently small interval around $A$;
- $\{B\}$: choose a sufficiently small interval around $B$;
- $\{A,B\}$: choose an interval containing both points.

Therefore, every two-point set on the real line can be shattered by the interval class.

## 4. VC dimension

### Definition 2: VC dimension

The VC dimension of a class $\mathcal{C}$ is the largest cardinality of a finite set that can be shattered by $\mathcal{C}$:

$$\mathrm{VC}(\mathcal{C})=\sup\left\lbrace \lvert S\rvert:S\text{ is shattered by }\mathcal{C}\right\rbrace.$$

The quantity $\lvert S\rvert$ is the number of elements in $S$.

A larger VC dimension means that the class can realize more complicated binary patterns on finite samples.

### Monotonicity

If

$$\mathcal{C}_{1}\subseteq\mathcal{C}_{2},$$

then every set shattered by $\mathcal{C}_{1}$ is also shattered by $\mathcal{C}_{2}$. Therefore,

$$\mathrm{VC}(\mathcal{C}_{1})\leq \mathrm{VC}(\mathcal{C}_{2}).$$

## 5. VC dimension of intervals

The interval class satisfies

$$\mathrm{VC}(\mathcal{C})=2.$$

### Lower bound

The two-point construction above shows that

$$\mathrm{VC}(\mathcal{C})\geq 2.$$

### Upper bound

Take any three ordered points

$$A<B<C$$

and let

$$S=\{A,B,C\}.$$

To shatter $S$, an interval would need to realize the subset

$$S'=\{A,C\}.$$

However, every interval containing both $A$ and $C$ must also contain the point $B$ between them. Hence no interval satisfies

$$S\cap[s,t]=\{A,C\}.$$

Therefore, no three-point set can be shattered by the class of intervals, so

$$\mathrm{VC}(\mathcal{C})\leq 2.$$

Combining the lower and upper bounds gives

$$\mathrm{VC}(\mathcal{C})=2.$$

> [!TIP]
> **The key obstruction**
>
> Intervals cannot create a disconnected selection pattern on ordered points. Once both outer points are included, every point between them must also be included.

## 6. Half-spaces in two dimensions

Let $\mathcal{Z}=\mathbb{R}^{2}$. The source considers the class of all affine half-spaces:

$$\mathcal{C}=\left\lbrace \left\lbrace (x_{1},x_{2})\in\mathbb{R}^{2}:w_{1}x_{1}+w_{2}x_{2}+b\geq 0\right\rbrace:w_{1},w_{2},b\in\mathbb{R}\right\rbrace.$$

The boundary of a member of this class is the line

$$w_{1}x_{1}+w_{2}x_{2}+b=0.$$

One side of the line belongs to the half-space, and the other side does not.

### Lower-bound example

The course uses three points in the plane to obtain

$$\mathrm{VC}(\mathcal{C})\geq 3.$$

For three points in a noncollinear configuration, every binary inclusion pattern can be produced by an affine half-space:

- all or none of the points can be selected;
- one vertex can be separated from the other two by a line;
- a pair of vertices can be selected by taking the complementary side of such a separation.

Thus, a suitable three-point set is shattered.

> [!NOTE]
> **A necessary precision about the three-point statement**
>
> Page 56 says that any set of three points in $\mathbb{R}^{2}$ can be shattered. The diagram and the intended VC lower-bound argument use points in general position. Three collinear points cannot be shattered because a half-space cannot select the two outer points while excluding the middle point. The source-supported conclusion needed here is that there exists a three-point set that can be shattered.

### Four-point obstruction shown in the notes

The notes draw a four-point configuration and choose two diagonally opposed points as the desired positive subset. No line can place exactly those two points on one side while placing the other two on the opposite side. This example illustrates a labeling that affine half-spaces cannot realize.

The source then states

$$\mathrm{VC}(\mathcal{C})=3.$$

> [!NOTE]
> **Scope of the page-level argument**
>
> The displayed four-point configuration proves only that this particular set is not shattered. To establish the upper bound $\mathrm{VC}(\mathcal{C})\leq 3$, one must show that no four-point set in $\mathbb{R}^{2}$ can be shattered. The source does not provide that full argument on pages 55-56 and instead points to external reference theorems.

## 7. Half-spaces in $\mathbb{R}^{d}$

The course states the general result:

$$\mathrm{VC}(\mathcal{C}_{d})=d+1,$$

where $\mathcal{C}_{d}$ is the class of affine half-spaces in $\mathbb{R}^{d}$.

The page does not prove this theorem. It refers the reader to Theorem 7.1 and Theorem 7.2 in Section 7.3 of the course reference material.

> [!NOTE]
> **What is and is not proved in these notes**
>
> The interval result is proved directly on page 55. The two-dimensional half-space diagrams motivate the result, while the rigorous general half-space theorem is cited rather than derived in the notebook.

## 8. Relation to Rademacher complexity

The final statement of the chapter is that the Rademacher complexity of a function class can be bounded using its VC dimension.

The source does not state the explicit inequality on pages 55-56. It again directs the reader to the reference material. Therefore, this chapter records only the relationship supported by the notes:

- VC dimension is a combinatorial measure of the label patterns a class can realize;
- Rademacher complexity measures the class's ability to correlate with random signs;
- the course reference connects the two by upper-bounding Rademacher complexity in terms of VC dimension.

## 9. Chapter summary

1. A class $\mathcal{C}$ shatters a finite set $S$ when every subset of $S$ can be represented as $S\cap C$ for some $C\in\mathcal{C}$.
2. The VC dimension is the largest size of a set that the class can shatter.
3. The class of intervals on $\mathbb{R}$ has VC dimension $2$.
4. The class of affine half-spaces in $\mathbb{R}^{2}$ has VC dimension $3$.
5. More generally, affine half-spaces in $\mathbb{R}^{d}$ have VC dimension $d+1$.
6. The notes state that VC dimension can be used to control Rademacher complexity, but defer the quantitative theorem to the reference material.

---

[← Previous: Generalization Bounds and Rademacher Complexity](07_generalization_and_rademacher.md) · [Course map](../course_map.md) · [Next: Smooth Convex Optimization →](09_smooth_convex_optimization.md)
