---
course: "Statistical Learning Theory"
chapter: "04"
title: "Soft-Margin SVM and Hinge Loss"
source_pages: "598SLT.pdf, pp. 15-17"
status: "consolidated v1.0"
---

# Soft-Margin SVM and Hinge Loss

**Source:** 598SLT.pdf, pp. 15-17.

## 1. Why hard margins are insufficient

The hard-margin SVM requires every training point to satisfy

$$y_{i}\left(w^{\top}x_{i}+b\right)\geq 1.$$

This is possible only when the training sample is linearly separable in the chosen feature space. Noise, overlap, or outlying observations can make the constraints infeasible.

The soft-margin formulation relaxes the margin constraints by introducing nonnegative slack variables $\xi_{i}$.

## 2. Slack variables

The relaxed constraints are

$$y_{i}\left(w^{\top}x_{i}+b\right)\geq 1-\xi_{i},$$

with

$$\xi_{i}\geq 0.$$

The soft-margin SVM solves

$$\min_{w,b,\xi_{1},\ldots,\xi_{n}}\;\frac{1}{2}\lVert w\rVert_{2}^{2}+C\sum_{i=1}^{n}\xi_{i},$$

subject to the relaxed constraints for every training point.

The parameter $C>0$ controls the cost of margin violations and is selected in practice by cross-validation.

### Interpreting $\xi_{i}$

- If $\xi_{i}=0$, then the point satisfies the unit-margin constraint.
- If $0<\xi_{i}<1$, then the point is correctly classified but lies inside the margin.
- If $\xi_{i}\geq 1$, then the score no longer has the correct positive signed margin; in particular, sufficiently large violations can correspond to misclassification.

!!! clarification "What does $C$ trade off?"
    The regularization term prefers a smaller norm and therefore a wider margin. The slack penalty prefers fewer or smaller violations. Increasing $C$ makes violations more expensive relative to the margin term; decreasing $C$ tolerates more violation in exchange for stronger regularization.

## 3. Soft-margin SVM with feature mapping

Let $\phi:\mathcal{X}\to\mathcal{H}$ map the inputs into a Hilbert space. The feature-mapped soft-margin problem is

$$\min_{w\in\mathcal{H},b,\xi_{1},\ldots,\xi_{n}}\;\frac{1}{2}\langle w,w\rangle_{\mathcal{H}}+C\sum_{i=1}^{n}\xi_{i},$$

subject to

$$y_{i}\left(\langle w,\phi(x_{i})\rangle_{\mathcal{H}}+b\right)\geq 1-\xi_{i}\quad\text{for }i=1,\ldots,n,$$

and

$$\xi_{i}\geq 0\quad\text{for }i=1,\ldots,n.$$

## 4. Hinge loss

Define

$$\Phi(t)=\max\{1-t,0\}.$$

![Hinge loss as a function of the signed margin score](../assets/diagrams/hinge_loss.png)

*Redrawn course diagram: hinge loss decreases linearly until the signed margin reaches one and is zero beyond that point.*

For a signed score

$$t=y_{i}\left(\langle w,\phi(x_{i})\rangle_{\mathcal{H}}+b\right),$$

$\Phi(t)$ is zero when $t\geq 1$ and increases linearly when $t<1$.

The course rewrites the constrained problem as

$$\min_{w\in\mathcal{H},b}\;\frac{1}{2}\langle w,w\rangle_{\mathcal{H}}+C\sum_{i=1}^{n}\Phi\left(y_{i}\left(\langle w,\phi(x_{i})\rangle_{\mathcal{H}}+b\right)\right).$$

The second term is the total hinge loss on the training sample.

## 5. Equivalence of the slack and hinge formulations

### Theorem 1

Define the constrained optimum

$$\mathrm{LHS}=\min_{w\in\mathcal{H},b,\xi_{1},\ldots,\xi_{n}}\;\frac{1}{2}\langle w,w\rangle_{\mathcal{H}}+C\sum_{i=1}^{n}\xi_{i},$$

subject to

$$y_{i}\left(\langle w,\phi(x_{i})\rangle_{\mathcal{H}}+b\right)\geq 1-\xi_{i},\qquad\xi_{i}\geq 0.$$

Define the unconstrained optimum

$$\mathrm{RHS}=\min_{w\in\mathcal{H},b}\;\frac{1}{2}\langle w,w\rangle_{\mathcal{H}}+C\sum_{i=1}^{n}\Phi\left(y_{i}\left(\langle w,\phi(x_{i})\rangle_{\mathcal{H}}+b\right)\right).$$

Then

$$\mathrm{LHS}=\mathrm{RHS}.$$

### Proof, direction 1: $\mathrm{LHS}\geq\mathrm{RHS}$

Let an optimal constrained solution be

$$\left(w^{\star},b^{\star},\xi_{1}^{\star},\ldots,\xi_{n}^{\star}\right).$$

For fixed $w^{\star}$ and $b^{\star}$, the smallest feasible slack is

$$\xi_{i}^{\star}=\max\left\lbrace 1-y_{i}\left(\langle w^{\star},\phi(x_{i})\rangle_{\mathcal{H}}+b^{\star}\right),0\right\rbrace.$$

Otherwise, a larger slack could be reduced while preserving feasibility and lowering the objective.

Therefore,

$$\xi_{i}^{\star}=\Phi\left(y_{i}\left(\langle w^{\star},\phi(x_{i})\rangle_{\mathcal{H}}+b^{\star}\right)\right).$$

The constrained optimum is consequently the hinge objective evaluated at $w^{\star},b^{\star}$. Since $\mathrm{RHS}$ is the minimum of that hinge objective over every $w,b$,

$$\mathrm{LHS}\geq\mathrm{RHS}.$$

### Proof, direction 2: $\mathrm{LHS}\leq\mathrm{RHS}$

Let $w^{\star},b^{\star}$ minimize the hinge formulation. Define

$$\xi_{i}^{\star}=\Phi\left(y_{i}\left(\langle w^{\star},\phi(x_{i})\rangle_{\mathcal{H}}+b^{\star}\right)\right).$$

By construction,

$$\xi_{i}^{\star}\geq 0,$$

and

$$\xi_{i}^{\star}\geq 1-y_{i}\left(\langle w^{\star},\phi(x_{i})\rangle_{\mathcal{H}}+b^{\star}\right).$$

Thus,

$$y_{i}\left(\langle w^{\star},\phi(x_{i})\rangle_{\mathcal{H}}+b^{\star}\right)\geq 1-\xi_{i}^{\star}.$$

The constructed tuple is feasible for the constrained problem and has exactly the same objective value as the hinge solution. Since $\mathrm{LHS}$ is the minimum over all feasible tuples,

$$\mathrm{LHS}\leq\mathrm{RHS}.$$

Combining the two inequalities gives

$$\mathrm{LHS}=\mathrm{RHS}.$$

!!! clarification "Why does the hinge loss replace many constraints?"
    For fixed $w$ and $b$, each slack variable has an obvious smallest feasible value. Substituting that value removes the slack variables and their constraints. The positive-part operation $\max\{1-t,0\}$ records exactly how much the signed margin falls below one.

## 6. Convexity

The notes emphasize that the hinge function is convex. Together with the squared-norm regularizer, this makes the soft-margin objective convex in the model parameters under the course formulation.

A local minimizer of a convex objective is also a global minimizer.

## Chapter summary

- Slack variables make the SVM feasible when some points violate the margin.
- The parameter $C$ balances regularization against violations.
- The smallest feasible slack for a point is its hinge loss.
- Substituting the optimal slacks gives an equivalent unconstrained regularized objective.
- The resulting soft-margin objective is convex.

---

[← Previous: Feature Maps and Kernels](03_feature_maps_and_kernels.md) · [Course map](../course_map.md) · [Next: RKHS and the Representer Theorem →](05_rkhs_and_representer.md)
