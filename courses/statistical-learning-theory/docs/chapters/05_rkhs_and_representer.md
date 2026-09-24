# 5. RKHS and the Representer Theorem

Chapter 3 assumed a feature map existed. This chapter goes the other way: start from a function $K(x,x')$ and *build* the Hilbert space it is an inner product for, the reproducing kernel Hilbert space. The key property is that evaluating a function becomes taking an inner product. The payoff is the representer theorem: even in an infinite-dimensional space, the regularized optimum is a combination of the $n$ training points.

## 1. The regularized objective

Dividing the soft-margin objective by $C$ doesn't change the minimizer:

$$\min_{w\in\mathcal{H},b}\;\sum_{i=1}^{n}\Phi\left(y_{i}\left(\langle w,\phi(x_{i})\rangle_{\mathcal{H}}+b\right)\right)+\lambda\lVert w\rVert_{\mathcal{H}}^{2},\qquad\lambda=\frac{1}{2C},$$

with $\lVert u\rVert_{\mathcal H}^2=\langle u,u\rangle_{\mathcal H}$. Which Hilbert space should this live in?

## 2. Finite combinations of kernel sections

Let $h:\mathcal{X}\times\mathcal{X}\to\mathbb{R}$. For fixed $u$, $h(\cdot,u)$ is a function of its first argument (a **kernel section**). Collect all finite combinations:

$$\mathcal{H}_{0}=\left\lbrace \sum_{i=1}^{m}\alpha_{i}h(\cdot,u_{i})\;\middle|\;m\geq 1,\;\alpha_{i}\in\mathbb{R},\;u_{i}\in\mathcal{X}\right\rbrace .$$

The elements of $\mathcal H_0$ are **functions** on $\mathcal X$, such as $f=\sum_i\alpha_ih(\cdot,u_i)$, not coordinate vectors.

## 3. Positive-definite functions

### Definition 3

A symmetric $h$ is **positive definite** if for every $m$, every $u_1,\ldots,u_m$ and every $\alpha\in\mathbb R^m$,

$$\sum_{i=1}^{m}\sum_{j=1}^{m}\alpha_{i}\alpha_{j}h(u_{i},u_{j})\geq 0 .$$

Equivalently, every Gram matrix $[h(u_i,u_j)]$ is positive semidefinite. (Many texts call this "positive semidefinite kernel"; the naming varies.)

## 4. An inner product on $\mathcal H_0$

For $f=\sum_i\alpha_ih(\cdot,u_i)$ and $g=\sum_j\beta_jh(\cdot,v_j)$ define

$$\langle f,g\rangle_{\mathcal{H}_{0}}=\sum_{i=1}^{m}\sum_{j=1}^{m'}\alpha_{i}\beta_{j}h(u_{i},v_{j}).$$

- **Symmetry** follows from $h(u,v)=h(v,u)$.
- **Linearity** holds because the double sum is linear in the coefficients.
- **Non-negativity**, $\langle f,f\rangle=\sum_{i,j}\alpha_i\alpha_jh(u_i,u_j)\ge0$, is exactly positive definiteness.

> [!NOTE]
> **Two details that need care**
>
> (a) A function in $\mathcal H_0$ can have several representations. The definition is still well defined, because $\langle f,g\rangle=\sum_j\beta_jf(v_j)$ depends on $f$ only through its values. (b) Definiteness: if $\langle f,f\rangle=0$ then by Cauchy–Schwarz (valid for positive semidefinite forms) $|f(x)|=|\langle f,h(\cdot,x)\rangle|\le\sqrt{\langle f,f\rangle}\sqrt{h(x,x)}=0$ for all $x$, so $f=0$.

## 5. The reproducing property

For $f=\sum_i\alpha_ih(\cdot,u_i)$,

$$\langle f,h(\cdot,x)\rangle_{\mathcal{H}_{0}}=\sum_{i}\alpha_{i}h(u_{i},x)=f(x).$$

**Taking an inner product with the section at $x$ evaluates the function at $x$.** In particular $\langle h(\cdot,x),h(\cdot,x')\rangle=h(x,x')$, so $\phi(x)=h(\cdot,x)$ is a feature map for $h$.

## 6. Completion: the RKHS

$\mathcal H_0$ need not be complete. Its **completion** $\mathcal H$ adds the limits of Cauchy sequences, and those limits are still functions, with the inner product and the reproducing property extending to them. $\mathcal H$ is the **reproducing kernel Hilbert space** of $h$. For computations it is enough to think of $\mathcal H$ as $\mathcal H_0$ plus the limits needed to make it complete.

## 7. Every kernel is positive definite, and conversely

If $K(x,x')=\langle\phi(x),\phi(x')\rangle_{\mathcal H'}$ for some feature space $\mathcal H'$, then

$$\sum_{i,j}\alpha_{i}\alpha_{j}K(u_{i},u_{j})=\left\lVert\sum_{i}\alpha_{i}\phi(u_{i})\right\rVert_{\mathcal{H}'}^{2}\geq 0,$$

so every kernel is positive definite. §2–6 prove the converse: every positive definite $h$ is the kernel of its RKHS. Together this is the **Moore–Aronszajn theorem**, and it is the practical test for "is this a valid kernel?"

> [!TIP]
> **$\mathcal H'$ versus the RKHS**
>
> A kernel can have many feature maps and feature spaces $\mathcal H'$. The RKHS is the canonical one, a space of functions built from the kernel sections, and it is unique. Don't assume a given $\mathcal H'$ is the RKHS.

## 8. The soft-margin SVM in an RKHS

With $\phi(x)=K(\cdot,x)$ and the reproducing property, $\langle w,K(\cdot,x_i)\rangle_{\mathcal H}=w(x_i)$. The weight vector **is** a function, and the objective becomes

$$\min_{w\in\mathcal{H},b}\;\frac{1}{2}\lVert w\rVert_{\mathcal{H}}^{2}+C\sum_{i}\Phi\left(y_{i}\left(w(x_i)+b\right)\right).$$

## 9. The representer theorem

### Theorem 2

A minimizer of the regularized objective above has the form

$$w^{\star}=\sum_{i=1}^{n}\alpha_{i}K(\cdot,x_{i}).$$

The search over an infinite-dimensional space reduces to $n$ coefficients.

## 10. Proof

Let $\mathcal A=\mathrm{span}\{K(\cdot,x_1),\ldots,K(\cdot,x_n)\}$ and split any $w=w_1+w_2$ with $w_1\in\mathcal A$ and $w_2\perp\mathcal A$.

- **The loss only sees $w_1$:** $w(x_i)=\langle w,K(\cdot,x_i)\rangle=\langle w_1,K(\cdot,x_i)\rangle+\underbrace{\langle w_2,K(\cdot,x_i)\rangle}_{0}$.
- **The regularizer prefers $w_2=0$:** $\lVert w\rVert^2=\lVert w_1\rVert^2+\lVert w_2\rVert^2$.

If $w_2\neq0$, replacing $w$ by $w_1$ keeps every training prediction and strictly lowers the objective. So any minimizer has $w_2=0$, i.e. $w^\star\in\mathcal A$. $\square$

In finite dimensions without a feature map this is the familiar statement $w^\star=\sum_i\alpha_ix_i$. The same argument works for any loss that depends on $w$ only through $w(x_1),\ldots,w(x_n)$ and any regularizer that is increasing in $\lVert w\rVert_{\mathcal H}$, which covers kernel ridge regression, kernel logistic regression and more.

> [!TIP]
> **Why it matters computationally**
>
> Substituting $w=\sum_j\alpha_jK(\cdot,x_j)$ gives $w(x_i)=(K\alpha)_i$ and $\lVert w\rVert^2=\alpha^\top K\alpha$. Any kernel method becomes a finite problem in $\alpha\in\mathbb R^n$. For kernel ridge regression it even has a closed form, $\alpha=(K+\lambda I)^{-1}y$.

## Summary

- Positive-definite functions define an inner product on finite combinations of kernel sections.
- The reproducing property $f(x)=\langle f,K(\cdot,x)\rangle$ turns evaluation into geometry.
- Completing that space gives the RKHS; valid kernels are exactly the positive-definite functions.
- The representer theorem puts the regularized optimum in the span of the training sections.

## Questions to test yourself

<details>
<summary>Why must a kernel be positive definite?</summary>

Because $\sum_{ij}\alpha_i\alpha_jK(u_i,u_j)$ is the squared norm of $\sum_i\alpha_i\phi(u_i)$; if it could be negative, $K$ couldn't be an inner product.
</details>

<details>
<summary>What fails if the form is non-negative but not definite?</summary>

A nonzero function could have zero norm, so $\lVert\cdot\rVert$ wouldn't be a norm. For kernel sections this can't happen (see the note in §4), because $|f(x)|\le\lVert f\rVert\sqrt{K(x,x)}$.
</details>

<details>
<summary>Why does the representer theorem need the regularizer to be increasing in the norm?</summary>

The proof removes $w_2$ because that lowers $\lVert w\rVert$ without changing the loss; if the regularizer didn't strictly prefer smaller norms, the orthogonal part wouldn't have to vanish.
</details>

---

[← Previous: Soft-Margin SVM](04_soft_margin_svm.md) · [Course map](../course_map.md) · [Next: Concentration Inequalities →](06_concentration_inequalities.md)
