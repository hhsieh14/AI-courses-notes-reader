# 10. Nonsmooth and Proximal Optimization

Lasso and SVMs have objectives with kinks, where gradients don't exist. This chapter handles objectives of the form *smooth + nonsmooth-but-simple*: take a gradient step on the smooth part, then a **proximal** step on the rest. For the $\ell_1$ norm the proximal step is soft thresholding, which is exactly where lasso's exact zeros come from. The method keeps gradient descent's $O(1/k)$ rate.

## 1. Why gradient descent is not always enough

The previous chapter studied gradient descent for differentiable convex objectives. Many machine-learning objectives, however, include a convex term that is not differentiable everywhere.

Consider the composite problem

$$\min_{x\in\mathbb{R}^{d}}f(x)=g(x)+h(x),$$

where

- $g:\mathbb{R}^{d}\to\mathbb{R}$ is convex and differentiable;
- $h:\mathbb{R}^{d}\to\mathbb{R}$ is convex but may be nondifferentiable.

Because $h$ may not have an ordinary gradient, gradient descent cannot be applied directly to the full objective $g+h$. Proximal gradient descent handles the two terms differently: it uses a gradient step for $g$ and a proximal step for $h$.

## 2. Example: $\ell_{1}$-regularized linear regression

A motivating example is the $\ell_{1}$-regularized linear-regression problem

$$\min_{\theta\in\mathbb{R}^{d}}\lVert y-X\theta\rVert_{2}^{2}+\lambda\lVert\theta\rVert_{1},$$

where

$$\lVert\theta\rVert_{1}=\sum_{i=1}^{d}\lvert\theta_{i}\rvert.$$

The composite terms are

$$g(\theta)=\lVert y-X\theta\rVert_{2}^{2}$$

and

$$h(\theta)=\lambda\lVert\theta\rVert_{1}.$$

The squared-error term is differentiable, while the $\ell_{1}$ term is nondifferentiable whenever a coordinate is zero.

### Why the absolute-value function is nondifferentiable at zero

For the one-dimensional function

$$h(x)=\lvert x\rvert,$$

the right derivative at zero is

$$\lim_{t\to 0^{+}}\frac{\lvert t\rvert-\lvert 0\rvert}{t}=1,$$

while the left derivative is

$$\lim_{t\to 0^{-}}\frac{\lvert t\rvert-\lvert 0\rvert}{t}=-1.$$

Because the two one-sided derivatives are different, $\lvert x\rvert$ is not differentiable at $x=0$.

> [!TIP]
> **Why is this example important?**
>
> The nondifferentiability occurs precisely at zero, which is also the value encouraged by $\ell_{1}$ regularization. A method designed for this objective must therefore handle the point at which ordinary differentiation fails.

## 3. The two-step idea

For a current point $x$, first take a gradient step using only the differentiable term $g$:

$$\widehat{x}=x-\eta\nabla g(x),$$

where $\eta>0$ is the learning rate.

Next, choose a point that makes $h$ small while remaining close to $\widehat{x}$:

$$x^{\prime}=\underset{v\in\mathbb{R}^{d}}{\arg\min}\left\lbrace h(v)+\frac{1}{2\eta}\lVert v-\widehat{x}\rVert_{2}^{2}\right\rbrace.$$

The quadratic term prevents the second step from moving arbitrarily far from the gradient update. The parameter $\eta$ therefore affects both the gradient step and the strength of the proximity penalty.

## 4. Proximal mapping

The proximal mapping of $h$ is defined by

$$\mathrm{prox}_{\eta h}(u)=\underset{v\in\mathbb{R}^{d}}{\arg\min}\left\lbrace h(v)+\frac{1}{2\eta}\lVert v-u\rVert_{2}^{2}\right\rbrace.$$

Using this definition, the two-step update becomes

$$x^{\prime}=\mathrm{prox}_{\eta h}\left(x-\eta\nabla g(x)\right).$$

The proximal mapping is not simply a gradient step on $h$. It solves a local optimization problem that balances the value of $h$ against distance from its input.

## 5. Proximal mapping for the $\ell_{1}$ norm

Suppose

$$h(x)=\lambda\lVert x\rVert_{1},\qquad \lambda>0.$$

For an input vector

$$u=\left[u_{1},u_{2},\ldots,u_{d}\right]^{\top},$$

the proximal mapping acts coordinate by coordinate.

When $u_{i}>\lambda\eta$,

$$\left[\mathrm{prox}_{\eta h}(u)\right]_{i}=u_{i}-\lambda\eta.$$

When $-\lambda\eta\leq u_{i}\leq\lambda\eta$,

$$\left[\mathrm{prox}_{\eta h}(u)\right]_{i}=0.$$

When $u_{i}<-\lambda\eta$,

$$\left[\mathrm{prox}_{\eta h}(u)\right]_{i}=u_{i}+\lambda\eta.$$

This is **soft thresholding**. Inputs within the interval $[-\lambda\eta,\lambda\eta]$ are mapped exactly to zero, while larger-magnitude inputs are moved toward zero by $\lambda\eta$.

> [!TIP]
> **How does the proximal step promote sparsity?**
>
> The mapping has a nontrivial interval of inputs that all produce the exact output zero. Repeated proximal-gradient updates can therefore set many coordinates of $\theta$ exactly to zero, which is the mechanism behind lasso's sparse coefficients.

## 6. Subgradients of convex functions

For a differentiable convex function $h$, the first-order convexity inequality is

$$h(y)\geq h(x)+\nabla h(x)^{\top}(y-x).$$

When $h$ is convex but not differentiable, we replace the gradient by a subgradient.

The subdifferential of $h$ at $x$ is

$$\partial h(x)=\left\lbrace v\in\mathbb{R}^{d}:h(y)\geq h(x)+v^{\top}(y-x)\text{ for every }y\in\mathbb{R}^{d}\right\rbrace.$$

Every vector in $\partial h(x)$ is a subgradient of $h$ at $x$.

If $h$ is differentiable at $x$, then the subdifferential contains only the ordinary gradient:

$$\partial h(x)=\left\lbrace \nabla h(x)\right\rbrace.$$

For a convex $h:\mathbb R^d\to\mathbb R$ the subdifferential is nonempty everywhere.

> [!TIP]
> **Why is it called a subgradient?**
>
> A subgradient defines an affine function that lies below the convex function. At a smooth point there is one tangent slope; at a nonsmooth point there may be several valid supporting slopes.

## 7. Example: the subdifferential of $\lvert x\rvert$

Let

$$h(x)=\lvert x\rvert.$$

For $x>0$, the function is differentiable and

$$\partial h(x)=\left\lbrace 1\right\rbrace.$$

For $x<0$,

$$\partial h(x)=\left\lbrace -1\right\rbrace.$$

At $x=0$,

$$\partial h(0)=[-1,1].$$

To verify the final statement, take any $v\in[-1,1]$. The subgradient condition at zero is

$$\lvert y\rvert\geq\lvert 0\rvert+v(y-0)=vy.$$

For $y\geq 0$, this follows from $v\leq 1$. For $y<0$, it follows from $v\geq -1$. Hence every $v\in[-1,1]$ is a valid subgradient at zero.

## 8. Optimality condition for the proximal mapping

Let

$$v^{\star}=\mathrm{prox}_{\eta h}(u).$$

By definition, $v^{\star}$ minimizes

$$h(v)+\frac{1}{2\eta}\lVert v-u\rVert_{2}^{2}.$$

The first-order subgradient condition is

$$0\in\partial h\left(v^{\star}\right)+\frac{1}{\eta}\left(v^{\star}-u\right).$$

Rearranging gives the fact used in the convergence proof:

$$\frac{1}{\eta}\left(u-v^{\star}\right)\in\partial h\left(v^{\star}\right).$$

This condition links a proximal update to a subgradient of the nonsmooth term.

## 9. Proximal gradient descent algorithm

The objective is

$$\min_{x\in\mathbb{R}^{d}}f(x)=g(x)+h(x),$$

with $g$ convex and differentiable and $h$ convex but possibly nondifferentiable.

Start from an initialization $x^{(0)}$. For iteration $k\geq 1$, proximal gradient descent performs

$$x^{(k)}=\mathrm{prox}_{\eta h}\left(x^{(k-1)}-\eta\nabla g\left(x^{(k-1)}\right)\right).$$

Use a fixed learning rate $\eta\le1/L$ and a fixed iteration budget.

> [!NOTE]
> **Why the prox step is cheap**
>
> Proximal gradient is only useful when $\mathrm{prox}_{\eta h}$ has a closed form or is easy to compute. Common cases: $\ell_1$ gives soft thresholding; the indicator of a convex set gives Euclidean projection (so projected gradient descent is a special case); $\ell_2$ gives shrinkage $u/(1+2\eta\lambda)$; group lasso gives block soft thresholding.

## 10. Convergence theorem for proximal gradient descent

Assume that $g:\mathbb{R}^{d}\to\mathbb{R}$ is convex and differentiable and that its gradient is Lipschitz continuous with constant $L>0$:

$$\lVert\nabla g(x)-\nabla g(y)\rVert_{2}\leq L\lVert x-y\rVert_{2}.$$

Let

$$x^{\star}\in\underset{x\in\mathbb{R}^{d}}{\arg\min}\;f(x).$$

### Theorem 1

Run proximal gradient descent for $k\geq 1$ iterations with a fixed learning rate satisfying

$$\eta\leq\frac{1}{L}.$$

Then

$$\left\lvert f\left(x^{(k)}\right)-f\left(x^{\star}\right)\right\rvert\leq\frac{\lVert x^{\star}-x^{(0)}\rVert_{2}^{2}}{2\eta k}.$$

Therefore,

$$f\left(x^{(k)}\right)-f\left(x^{\star}\right)=O\left(\frac{1}{k}\right).$$

> Since $x^\star$ is a global minimizer, $f(x^{(k)})-f(x^\star)\ge0$, so no absolute value is needed.

## 11. Proof of Theorem 1

### Step 1: combine convexity and smoothness of $g$

By convexity of $g$, for every $v\in\mathbb{R}^{d}$,

$$g\left(x^{(k-1)}\right)\leq g(v)+\nabla g\left(x^{(k-1)}\right)^{\top}\left(x^{(k-1)}-v\right).$$

Because $\nabla g$ is $L$-Lipschitz, the descent lemma gives

$$g\left(x^{(k)}\right)\leq g\left(x^{(k-1)}\right)+\nabla g\left(x^{(k-1)}\right)^{\top}\left(x^{(k)}-x^{(k-1)}\right)+\frac{L}{2}\left\lVert x^{(k)}-x^{(k-1)}\right\rVert_{2}^{2}.$$

Combining the two inequalities yields

$$g\left(x^{(k)}\right)\leq g(v)+\nabla g\left(x^{(k-1)}\right)^{\top}\left(x^{(k)}-v\right)+\frac{L}{2}\left\lVert x^{(k)}-x^{(k-1)}\right\rVert_{2}^{2}.$$

### Step 2: use the proximal optimality condition

The PGD update is

$$x^{(k)}=\mathrm{prox}_{\eta h}\left(x^{(k-1)}-\eta\nabla g\left(x^{(k-1)}\right)\right).$$

Applying the proximal optimality condition gives

$$\frac{1}{\eta}\left(x^{(k-1)}-\eta\nabla g\left(x^{(k-1)}\right)-x^{(k)}\right)\in\partial h\left(x^{(k)}\right).$$

By the definition of a subgradient, for every $v\in\mathbb{R}^{d}$,

$$h\left(x^{(k)}\right)\leq h(v)+\frac{1}{\eta}\left(x^{(k-1)}-\eta\nabla g\left(x^{(k-1)}\right)-x^{(k)}\right)^{\top}\left(x^{(k)}-v\right).$$

### Step 3: obtain a one-step inequality for $f$

Add the bounds for $g(x^{(k)})$ and $h(x^{(k)})$. The two terms containing $\nabla g(x^{(k-1)})$ cancel, giving

$$f\left(x^{(k)}\right)\leq f(v)+\frac{1}{\eta}\left(x^{(k)}-x^{(k-1)}\right)^{\top}\left(v-x^{(k)}\right)+\frac{L}{2}\left\lVert x^{(k)}-x^{(k-1)}\right\rVert_{2}^{2}.$$

Write

$$v-x^{(k)}=v-x^{(k-1)}-\left(x^{(k)}-x^{(k-1)}\right).$$

Substitution gives

$$f\left(x^{(k)}\right)\leq f(v)+\frac{1}{\eta}\left(x^{(k)}-x^{(k-1)}\right)^{\top}\left(v-x^{(k-1)}\right)-\left(\frac{1}{\eta}-\frac{L}{2}\right)\left\lVert x^{(k)}-x^{(k-1)}\right\rVert_{2}^{2}.$$

Because $\eta\leq 1/L$,

$$\frac{1}{\eta}-\frac{L}{2}\geq\frac{1}{2\eta}.$$

Therefore,

$$f\left(x^{(k)}\right)\leq f(v)+\frac{1}{\eta}\left(x^{(k)}-x^{(k-1)}\right)^{\top}\left(v-x^{(k-1)}\right)-\frac{1}{2\eta}\left\lVert x^{(k)}-x^{(k-1)}\right\rVert_{2}^{2}.$$

### Step 4: telescope the distances to the minimizer

Set $v=x^{\star}$. Then

$$f\left(x^{(k)}\right)-f\left(x^{\star}\right)\leq\frac{1}{\eta}\left(x^{(k)}-x^{(k-1)}\right)^{\top}\left(x^{\star}-x^{(k-1)}\right)-\frac{1}{2\eta}\left\lVert x^{(k)}-x^{(k-1)}\right\rVert_{2}^{2}.$$

Using the identity

$$2\left(x^{(k)}-x^{(k-1)}\right)^{\top}\left(x^{\star}-x^{(k-1)}\right)-\left\lVert x^{(k)}-x^{(k-1)}\right\rVert_{2}^{2}=\left\lVert x^{(k-1)}-x^{\star}\right\rVert_{2}^{2}-\left\lVert x^{(k)}-x^{\star}\right\rVert_{2}^{2},$$

we obtain

$$f\left(x^{(k)}\right)-f\left(x^{\star}\right)\leq\frac{1}{2\eta}\left(\left\lVert x^{(k-1)}-x^{\star}\right\rVert_{2}^{2}-\left\lVert x^{(k)}-x^{\star}\right\rVert_{2}^{2}\right).$$

Summing from $k=1$ to $k=k^{\prime}$ gives

$$\sum_{k=1}^{k^{\prime}}\left(f\left(x^{(k)}\right)-f\left(x^{\star}\right)\right)\leq\frac{1}{2\eta}\left(\left\lVert x^{(0)}-x^{\star}\right\rVert_{2}^{2}-\left\lVert x^{(k^{\prime})}-x^{\star}\right\rVert_{2}^{2}\right).$$

Hence

$$\sum_{k=1}^{k^{\prime}}\left(f\left(x^{(k)}\right)-f\left(x^{\star}\right)\right)\leq\frac{1}{2\eta}\left\lVert x^{(0)}-x^{\star}\right\rVert_{2}^{2}.$$

### Step 5: pass from the average gap to the last iterate

Set $v=x^{(k-1)}$ in the one-step inequality from Step 3. The inner-product term vanishes, so

$$f\left(x^{(k)}\right)\leq f\left(x^{(k-1)}\right)-\frac{1}{2\eta}\left\lVert x^{(k)}-x^{(k-1)}\right\rVert_{2}^{2}\leq f\left(x^{(k-1)}\right).$$

Thus the sequence of objective gaps is nonincreasing. Therefore,

$$k^{\prime}\left(f\left(x^{(k^{\prime})}\right)-f\left(x^{\star}\right)\right)\leq\sum_{k=1}^{k^{\prime}}\left(f\left(x^{(k)}\right)-f\left(x^{\star}\right)\right).$$

Combining this with the telescoping bound gives

$$f\left(x^{(k^{\prime})}\right)-f\left(x^{\star}\right)\leq\frac{\left\lVert x^{(0)}-x^{\star}\right\rVert_{2}^{2}}{2\eta k^{\prime}}.$$

Renaming $k^{\prime}$ as $k$ completes the proof.

## 12. Accelerated proximal gradient descent

Acceleration carries over directly (FISTA):

Initialize $x^{(0)}$, set

$$x^{(1)}=x^{(0)},\qquad t_{0}=0,\qquad t_{1}=1.$$

At iteration $k\geq 1$, form the extrapolated point

$$y^{(k)}=x^{(k)}+\frac{t_{k-1}-1}{t_{k}}\left(x^{(k)}-x^{(k-1)}\right).$$

Then take a proximal-gradient step from $y^{(k)}$:

$$x^{(k+1)}=\mathrm{prox}_{\eta h}\left(y^{(k)}-\eta\nabla g\left(y^{(k)}\right)\right).$$

Update the momentum parameter using

$$t_{k+1}=\frac{\sqrt{1+4t_{k}^{2}}+1}{2}.$$

The rates are

$$f\left(x^{(k)}\right)-f\left(x^{\star}\right)=O\left(\frac{1}{k}\right)$$

for proximal gradient descent and

$$f\left(x^{(k)}\right)-f\left(x^{\star}\right)=O\left(\frac{1}{k^{2}}\right)$$

for accelerated proximal gradient descent.

> [!NOTE]
> **Initialization indices**
>
> Start with $x^{(0)}=x^{(1)}$ (the first extrapolation has no momentum), matching Chapter 9. The $O(1/k^2)$ proof is the same estimate-sequence argument as for AGD, with the prox step in place of the gradient step (Beck & Teboulle, 2009).

## 13. Main takeaways

- Composite objectives separate a differentiable term $g$ from a convex term $h$ that may be nonsmooth.
- A proximal update minimizes $h$ while remaining close to a gradient step on $g$.
- For an $\ell_{1}$ penalty, the proximal mapping is soft thresholding and can create exact zeros.
- Subgradients replace ordinary gradients at nondifferentiable points.
- The proximal optimality condition is the key link between the algorithm and the convergence proof.
- With $\eta\leq 1/L$, proximal gradient descent has an $O(1/k)$ objective-gap bound.
- Acceleration (FISTA) improves the rate to $O(1/k^{2})$.

---

[← Previous: Smooth Convex Optimization](09_smooth_convex_optimization.md) · [Course map](../course_map.md) · [Next: Stochastic Gradient Descent →](11_stochastic_gradient_descent.md)
