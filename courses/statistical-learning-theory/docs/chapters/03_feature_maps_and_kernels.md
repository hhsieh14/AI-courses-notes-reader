# 3. Feature Maps and Kernels

A linear classifier fails when the classes aren't linearly separable. This chapter maps the inputs into a (possibly infinite-dimensional) Hilbert space where they might be, and then observes that the SVM dual never needs the mapped vectors themselves, only their inner products. A **kernel** supplies those inner products directly.

## 1. Why transform features?

Choose a feature map

$$\phi:\mathcal{X}\to\mathcal{H}$$

into a Hilbert space $\mathcal H$. Data that aren't linearly separable in $\mathcal X$ may be separable in $\mathcal H$. The classifier is linear in $\phi(x)$, and its boundary is nonlinear in $x$.

![Feature map and kernel trick](../assets/diagrams/kernel_trick.png)

*A feature map can make the classes linearly separable; the kernel computes feature-space inner products without constructing the mapped vectors.*

## 2. Hilbert spaces

### Definition 1

A **Hilbert space** $\mathcal H$ is a vector space with an inner product $\langle\cdot,\cdot\rangle_{\mathcal H}$ that is **complete**: every Cauchy sequence converges to an element of the space. For all $f,g,f_1,f_2\in\mathcal H$ and $\alpha,\beta\in\mathbb R$ the inner product satisfies

- **symmetry:** $\langle f,g\rangle_{\mathcal{H}}=\langle g,f\rangle_{\mathcal{H}}$;
- **linearity:** $\langle \alpha f_{1}+\beta f_{2},g\rangle_{\mathcal{H}}=\alpha\langle f_{1},g\rangle_{\mathcal{H}}+\beta\langle f_{2},g\rangle_{\mathcal{H}}$;
- **positive definiteness:** $\langle f,f\rangle_{\mathcal{H}}\geq 0$, with equality only for $f=0$.

The familiar example is $\mathbb R^d$ with $\langle x,x'\rangle=x^\top x'$. It helps to think of a Hilbert space as Euclidean space that is allowed to be infinite-dimensional, for example a space of functions. Completeness matters when we build a space by taking limits; it becomes important in Chapter 5, where the RKHS is constructed as a completion.

## 3. Kernels

### Definition 2

Let $\mathcal X$ be any nonempty set. $K:\mathcal{X}\times\mathcal{X}\to\mathbb{R}$ is a **kernel** if there exist a Hilbert space $\mathcal H$ and a map $\phi:\mathcal X\to\mathcal H$ with

$$K(x,x')=\langle\phi(x),\phi(x')\rangle_{\mathcal{H}} .$$

Evaluating $K$ gives the feature-space inner product without writing down the features.

## 4. The SVM in feature space

Replacing $x$ by $\phi(x)$ in the hard-margin primal gives

$$\min_{w\in\mathcal{H},\,b}\;\frac{1}{2}\langle w,w\rangle_{\mathcal{H}}\quad\text{s.t.}\quad y_{i}\left(\langle w,\phi(x_{i})\rangle_{\mathcal{H}}+b\right)\geq 1,$$

and the same derivation as Chapter 2 gives the dual

$$\max_{\alpha}\;\sum_{i}\alpha_{i}-\frac{1}{2}\sum_{i,j}\alpha_{i}\alpha_{j}y_{i}y_{j}\langle\phi(x_{i}),\phi(x_{j})\rangle_{\mathcal{H}}\quad\text{s.t.}\quad\alpha_{i}\geq 0,\;\sum_{i}\alpha_{i}y_{i}=0 .$$

## 5. The kernel trick

Replace each inner product by $K(x_i,x_j)$:

$$\max_{\alpha}\;\sum_{i}\alpha_{i}-\frac{1}{2}\sum_{i,j}\alpha_{i}\alpha_{j}y_{i}y_{j}K(x_{i},x_{j}),$$

with the same constraints. Predictions only need kernel evaluations too:

$$f(x)=\mathrm{sign}\Big(\sum_i\alpha_i y_i K(x_i,x)+b\Big).$$

> [!TIP]
> **Why this matters**
>
> For $\phi$ with all monomials up to degree $p$ in $d$ variables, $\dim\mathcal H=\binom{d+p}{p}$, and for the Gaussian kernel it is infinite. Storing $\phi(x)$ is expensive or impossible, but $K(x,x')=(1+x^\top x')^p$ or $e^{-\gamma\lVert x-x'\rVert^2}$ costs $O(d)$. The dual needs nothing else.

## 6. Kernel matrix and quadratic programming

Precompute the $n\times n$ Gram matrix $K_{ij}=K(x_i,x_j)$ and hand the dual to a QP solver. Memory is $O(n^2)$ and generic QP solvers scale roughly as $O(n^3)$, which is the practical limit of exact kernel methods. Approximations (random Fourier features, Nyström) trade exactness for linear scaling.

> [!NOTE]
> **What isn't answered yet**
>
> Definition 2 needs an existing $\phi$. Given a candidate function $K$, how do we know such a $\phi$ exists? Chapter 5 answers this: $K$ is a kernel **if and only if** it is symmetric and positive semidefinite, and it builds the canonical feature space, the RKHS, from $K$ alone.

## Summary

- Feature maps can make data separable; the classifier stays linear in feature space.
- A Hilbert space is a complete inner-product space, possibly infinite-dimensional.
- A kernel is an inner product of feature maps.
- The SVM dual and its predictions only need kernel values: the kernel trick.

## Questions to test yourself

<details>
<summary>What computation does the kernel trick avoid?</summary>

Constructing and storing $\phi(x_i)$, which may have enormous or infinite dimension. Only the $n\times n$ inner products are ever needed.
</details>

<details>
<summary>What is the main computational cost of exact kernel SVMs?</summary>

The Gram matrix: $O(n^2)$ memory and roughly $O(n^2)$–$O(n^3)$ time to solve the dual, which is why they struggle beyond ~10⁵ examples.
</details>

---

[← Previous: Hard-Margin SVM](02_hard_margin_svm.md) · [Course map](../course_map.md) · [Next: Soft-Margin SVM and Hinge Loss →](04_soft_margin_svm.md)
