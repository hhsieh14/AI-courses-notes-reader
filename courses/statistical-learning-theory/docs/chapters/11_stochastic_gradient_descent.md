# 11. Stochastic Gradient Descent

When the objective is an average over millions of examples, even one full gradient is expensive. SGD uses the gradient of a single random example instead: a noisy but unbiased estimate. This chapter proves that with a fixed step size SGD converges to within $\eta\sigma^2$ of the optimum at rate $O(1/k)$, and explains how decaying step sizes remove that floor.

## 1. Why introduce stochastic gradients?

The previous chapters studied gradient descent for smooth convex objectives and proximal gradient descent for composite objectives. Both methods use information from the full objective at every iteration.

For linear regression,

$$J(\theta)=\lVert y-X\theta\rVert_{2}^{2}.$$

Given training data

$$\mathcal{D}=\left\lbrace \left(x^{(i)},y^{(i)}\right)\right\rbrace_{i=1}^{n},$$

the objective can be decomposed into per-sample losses:

$$J(\theta)=\sum_{i=1}^{n}J_{i}(\theta),$$

where

$$J_{i}(\theta)=\left(y^{(i)}-\left(x^{(i)}\right)^{\top}\theta\right)^{2}.$$

A full gradient-descent update is therefore

$$\theta^{(k)}=\theta^{(k-1)}-\eta\nabla J\left(\theta^{(k-1)}\right),$$

or equivalently,

$$\theta^{(k)}=\theta^{(k-1)}-\eta\sum_{i=1}^{n}\nabla J_{i}\left(\theta^{(k-1)}\right).$$

Every update requires gradients from all $n$ training examples. When $n$ is very large, this can be expensive.

Stochastic gradient descent replaces the full gradient by the gradient of one randomly selected sample:

$$\theta^{(k)}=\theta^{(k-1)}-\eta\nabla J_{i_{k}}\left(\theta^{(k-1)}\right),$$

where

$$i_{k}\in\{1,2,\ldots,n\}$$

is a random index.

> [!TIP]
> **What is gained and what is lost?**
>
> One SGD iteration uses only one sample gradient instead of all $n$ sample gradients. The resulting update is cheaper but noisy: it does not usually point in exactly the same direction as the full gradient.

> One SGD step costs $1/n$ of a full-gradient step in gradient evaluations. That doesn't make SGD $n$ times faster overall: it needs more steps, and hardware favors mini-batches. The win is largest early in training and when $n$ is huge.

## 2. Finite-sum optimization problem

The general problem is

$$\min_{x\in\mathbb{R}^{d}}f(x)=\frac{1}{n}\sum_{i=1}^{n}f_{i}(x).$$

Each $f_{i}$ is usually the loss associated with the $i$-th training sample. Examples: squared loss for linear regression, hinge loss for SVMs.

The component functions do not need to be identical:

$$f_{i}\neq f_{j}\quad\text{may hold when }i\neq j.$$

> Sums and averages differ by a factor of $n$, which rescales the gradient and so the effective learning rate. The theorem below uses the average.

## 3. SGD algorithm for the finite-sum problem

Choose an initialization

$$x^{(0)}\in\mathbb{R}^{d}.$$

At iteration $k\geq 1$, randomly choose

$$i_{k}\in\{1,2,\ldots,n\}$$

and update

$$x^{(k)}=x^{(k-1)}-\eta\nabla f_{i_{k}}\left(x^{(k-1)}\right),$$

where $\eta>0$ is the learning rate.

Here we use a fixed learning rate and a fixed number of iterations.

> [!TIP]
> **Why is the output averaged?**
>
> The sequence $x^{(1)},x^{(2)},\ldots$ is random because every update depends on a random index. The theorem controls the objective at the averaged iterate rather than directly controlling the final raw iterate.

## 4. Assumptions for the convergence result

We make three assumptions.

### Assumption 1: convex and smooth component functions

For every $1\leq i\leq n$, the function

$$f_{i}:\mathbb{R}^{d}\to\mathbb{R}$$

is convex and has an $L$-Lipschitz-continuous gradient:

$$\left\lVert\nabla f_{i}(x)-\nabla f_{i}(y)\right\rVert_{2}\leq L\lVert x-y\rVert_{2}$$

for every $x,y\in\mathbb{R}^{d}$.

Because $f$ is the average of the $f_{i}$ functions, $f$ is also convex and has an $L$-Lipschitz-continuous gradient.

### Assumption 2: unbiased stochastic gradient

For every $x\in\mathbb{R}^{d}$,

$$\mathbb{E}\left[\nabla f_{i_{k}}(x)\right]=\nabla f(x).$$

Thus the random component gradient is an unbiased estimator of the full gradient.

For uniform sampling of $i_{k}$,

$$\mathbb{E}\left[\nabla f_{i_{k}}(x)\right]=\frac{1}{n}\sum_{i=1}^{n}\nabla f_{i}(x)=\nabla f(x).$$

### Assumption 3: bounded gradient noise

The convergence proof uses the variance condition

$$\mathbb{E}\left[\left\lVert\nabla f_{i_{k}}(x)-\nabla f(x)\right\rVert_{2}^{2}\right]\leq\sigma^{2},$$

where $\sigma$ is a constant.

Equivalently, under the unbiasedness assumption,

$$\mathbb{E}\left[\left\lVert\nabla f_{i_{k}}(x)\right\rVert_{2}^{2}\right]\leq\sigma^{2}+\lVert\nabla f(x)\rVert_{2}^{2}.$$

> The proof needs a bound on the vector second moment, $\mathbb E\lVert\nabla f_{i_k}(x)-\nabla f(x)\rVert^2\le\sigma^2$, not just on the variance of the gradient's norm.

## 5. Optimal point and averaged iterate

Assume a global minimizer exists:

$$x^{\star}\in\underset{x\in\mathbb{R}^{d}}{\arg\min}\;f(x).$$

Define the average of the first $k$ SGD iterates by

$$\bar{x}^{(k)}=\frac{1}{k}\sum_{t=1}^{k}x^{(t)}.$$

The goal is to show that the expected objective value at $\bar{x}^{(k)}$ is close to the optimal value $f(x^{\star})$.

## 6. Convergence theorem

### Theorem 1

Suppose Assumptions 1-3 hold. Run SGD for $k\geq 1$ iterations using a fixed learning rate satisfying

$$\eta\leq\frac{1}{L}.$$

Then the averaged iterate satisfies

$$\left\lvert\mathbb{E}\left[f\left(\bar{x}^{(k)}\right)\right]-f\left(x^{\star}\right)\right\rvert\leq\frac{\left\lVert x^{\star}-x^{(0)}\right\rVert_{2}^{2}}{2\eta k}+\eta\sigma^{2}.$$

The right-hand side contains two terms:

$$\frac{\left\lVert x^{\star}-x^{(0)}\right\rVert_{2}^{2}}{2\eta k}$$

decays as the number of iterations increases, while

$$\eta\sigma^{2}$$

is the residual term caused by stochastic-gradient noise under a fixed learning rate.

> [!TIP]
> **The learning-rate tradeoff**
>
> A larger fixed learning rate reduces the first term more quickly but increases the noise term. A smaller learning rate reduces the residual term but makes the optimization term larger for a fixed $k$.

> $f(\bar x^{(k)})-f(x^\star)\ge0$ since $x^\star$ is a global minimizer, so no absolute value is needed.

## 7. Proof of Theorem 1

For concise notation, define the stochastic gradient at iteration $k$ by

$$v_{k}=\nabla f_{i_{k}}\left(x^{(k-1)}\right).$$

Then

$$x^{(k)}=x^{(k-1)}-\eta v_{k}.$$

When conditioning on the history before iteration $k$, the point $x^{(k-1)}$ is fixed and only $i_{k}$ is random. Therefore,

$$\mathbb{E}\left[v_{k}\right]=\nabla f\left(x^{(k-1)}\right).$$

### Step 1: apply smoothness of $f$

Since $f$ has an $L$-Lipschitz-continuous gradient, the descent lemma gives

$$f\left(x^{(k)}\right)\leq f\left(x^{(k-1)}\right)+\nabla f\left(x^{(k-1)}\right)^{\top}\left(x^{(k)}-x^{(k-1)}\right)+\frac{L}{2}\left\lVert x^{(k)}-x^{(k-1)}\right\rVert_{2}^{2}.$$

Substitute the SGD update:

$$f\left(x^{(k)}\right)\leq f\left(x^{(k-1)}\right)-\eta\nabla f\left(x^{(k-1)}\right)^{\top}v_{k}+\frac{L\eta^{2}}{2}\lVert v_{k}\rVert_{2}^{2}.$$

### Step 2: take expectation over the sampled index

The variance assumption gives

$$\mathbb{E}\left[\lVert v_{k}\rVert_{2}^{2}\right]\leq\sigma^{2}+\left\lVert\nabla f\left(x^{(k-1)}\right)\right\rVert_{2}^{2}.$$

Taking expectation in the smoothness inequality yields

$$\mathbb{E}\left[f\left(x^{(k)}\right)\right]\leq f\left(x^{(k-1)}\right)-\eta\left\lVert\nabla f\left(x^{(k-1)}\right)\right\rVert_{2}^{2}+\frac{L\eta^{2}}{2}\left(\sigma^{2}+\left\lVert\nabla f\left(x^{(k-1)}\right)\right\rVert_{2}^{2}\right).$$

Since $\eta\leq 1/L$,

$$\mathbb{E}\left[f\left(x^{(k)}\right)\right]\leq f\left(x^{(k-1)}\right)-\frac{\eta}{2}\left\lVert\nabla f\left(x^{(k-1)}\right)\right\rVert_{2}^{2}+\frac{\eta}{2}\sigma^{2}.$$

### Step 3: use convexity

Convexity of $f$ gives

$$f\left(x^{(k-1)}\right)\leq f\left(x^{\star}\right)+\nabla f\left(x^{(k-1)}\right)^{\top}\left(x^{(k-1)}-x^{\star}\right).$$

Combining this with the previous inequality gives

$$\mathbb{E}\left[f\left(x^{(k)}\right)\right]\leq f\left(x^{\star}\right)+\mathbb{E}\left[v_{k}\right]^{\top}\left(x^{(k-1)}-x^{\star}\right)-\frac{\eta}{2}\left\lVert\nabla f\left(x^{(k-1)}\right)\right\rVert_{2}^{2}+\frac{\eta}{2}\sigma^{2}.$$

Using

$$\left\lVert\nabla f\left(x^{(k-1)}\right)\right\rVert_{2}^{2}\geq\mathbb{E}\left[\lVert v_{k}\rVert_{2}^{2}\right]-\sigma^{2},$$

we obtain

$$\mathbb{E}\left[f\left(x^{(k)}\right)\right]\leq f\left(x^{\star}\right)+\mathbb{E}\left[v_{k}^{\top}\left(x^{(k-1)}-x^{\star}\right)-\frac{\eta}{2}\lVert v_{k}\rVert_{2}^{2}\right]+\eta\sigma^{2}.$$

### Step 4: rewrite the stochastic-gradient terms as a distance difference

From the update $x^{(k)}=x^{(k-1)}-\eta v_{k}$,

$$\left\lVert x^{(k)}-x^{\star}\right\rVert_{2}^{2}=\left\lVert x^{(k-1)}-x^{\star}\right\rVert_{2}^{2}-2\eta v_{k}^{\top}\left(x^{(k-1)}-x^{\star}\right)+\eta^{2}\lVert v_{k}\rVert_{2}^{2}.$$

Rearranging,

$$v_{k}^{\top}\left(x^{(k-1)}-x^{\star}\right)-\frac{\eta}{2}\lVert v_{k}\rVert_{2}^{2}=\frac{1}{2\eta}\left(\left\lVert x^{(k-1)}-x^{\star}\right\rVert_{2}^{2}-\left\lVert x^{(k)}-x^{\star}\right\rVert_{2}^{2}\right).$$

Therefore,

$$\mathbb{E}\left[f\left(x^{(k)}\right)\right]-f\left(x^{\star}\right)\leq\mathbb{E}\left[\frac{1}{2\eta}\left(\left\lVert x^{(k-1)}-x^{\star}\right\rVert_{2}^{2}-\left\lVert x^{(k)}-x^{\star}\right\rVert_{2}^{2}\right)\right]+\eta\sigma^{2}.$$

### Step 5: sum and telescope

Summing from $k=1$ to $k=k^{\prime}$ gives

$$\sum_{k=1}^{k^{\prime}}\left(\mathbb{E}\left[f\left(x^{(k)}\right)\right]-f\left(x^{\star}\right)\right)\leq\frac{1}{2\eta}\left(\left\lVert x^{(0)}-x^{\star}\right\rVert_{2}^{2}-\mathbb{E}\left[\left\lVert x^{(k^{\prime})}-x^{\star}\right\rVert_{2}^{2}\right]\right)+k^{\prime}\eta\sigma^{2}.$$

Dropping the nonnegative final squared distance yields

$$\sum_{k=1}^{k^{\prime}}\left(\mathbb{E}\left[f\left(x^{(k)}\right)\right]-f\left(x^{\star}\right)\right)\leq\frac{1}{2\eta}\left\lVert x^{(0)}-x^{\star}\right\rVert_{2}^{2}+k^{\prime}\eta\sigma^{2}.$$

### Step 6: apply Jensen's inequality to the averaged iterate

Because $f$ is convex,

$$f\left(\bar{x}^{(k^{\prime})}\right)=f\left(\frac{1}{k^{\prime}}\sum_{k=1}^{k^{\prime}}x^{(k)}\right)\leq\frac{1}{k^{\prime}}\sum_{k=1}^{k^{\prime}}f\left(x^{(k)}\right).$$

Taking expectation and multiplying by $k^{\prime}$ gives

$$k^{\prime}\left(\mathbb{E}\left[f\left(\bar{x}^{(k^{\prime})}\right)\right]-f\left(x^{\star}\right)\right)\leq\sum_{k=1}^{k^{\prime}}\left(\mathbb{E}\left[f\left(x^{(k)}\right)\right]-f\left(x^{\star}\right)\right).$$

Combining the last two bounds gives

$$\mathbb{E}\left[f\left(\bar{x}^{(k^{\prime})}\right)\right]-f\left(x^{\star}\right)\leq\frac{\left\lVert x^{(0)}-x^{\star}\right\rVert_{2}^{2}}{2\eta k^{\prime}}+\eta\sigma^{2}.$$

Renaming $k^{\prime}$ as $k$ completes the proof.

> Each step's expectation is conditional on the indices drawn so far, $\mathbb E[\,\cdot\mid i_1,\ldots,i_{k-1}]$; the tower property turns the chain of conditional bounds into the unconditional statement.

## 8. What the theorem does and does not say

The theorem guarantees convergence of the expected objective at the averaged iterate to a neighborhood whose size is controlled by $\eta\sigma^{2}$.

With a fixed $\eta$, the bound doesn't go to zero as $k\to\infty$ unless $\sigma=0$: SGD settles into a noise ball of radius about $\eta\sigma^2$. The note below shows how to shrink it.

> [!NOTE]
> **Beyond the lecture: removing the $\eta\sigma^2$ floor**
>
> Balance the two terms. For a budget of $k$ steps, $\eta=\min\{1/L,\;\lVert x^{(0)}-x^\star\rVert/(\sigma\sqrt{2k})\}$ gives $\mathbb E f(\bar x^{(k)})-f^\star\le\frac{L\lVert x^{(0)}-x^\star\rVert^2}{2k}+\frac{\sqrt2\,\sigma\lVert x^{(0)}-x^\star\rVert}{\sqrt k}$, the familiar $O(1/\sqrt k)$ SGD rate. Decaying steps $\eta_t\propto1/\sqrt t$ achieve the same without knowing $k$, and with strong convexity $\eta_t\propto1/t$ gives $O(1/k)$. Almost-sure (not just in-expectation) convergence also holds under weighted averaging schemes (Sebbouh, Gower & Defazio, 2021). Mini-batches of size $B$ divide $\sigma^2$ by $B$.

## 9. Main takeaways

- Full gradient descent uses all $n$ component gradients at every iteration.
- SGD uses one randomly selected component gradient, reducing the work per update but introducing noise.
- Unbiasedness makes the stochastic gradient correct on average.
- A bounded-noise assumption controls its second moment.
- Under convexity and smoothness, a fixed-step SGD bound has an optimization term of order $1/k$ and a residual term $\eta\sigma^{2}$.
- Averaging the iterates is essential in the proof because Jensen's inequality converts the average of objective values into an objective value at the average iterate.

---

[← Previous: Nonsmooth and Proximal Optimization](10_nonsmooth_proximal_optimization.md) · [Course map](../course_map.md) · [Next: Optimization of Neural Networks →](12_neural_network_optimization.md)
