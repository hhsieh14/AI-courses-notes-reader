---
course: "Statistical Learning Theory"
chapter: "11"
title: "Stochastic Gradient Descent"
source_pages: "598SLT.pdf, pp. 71-74"
status: "consolidated v1.0"
---

# Stochastic Gradient Descent

**Source:** 598SLT.pdf, pp. 71-74.

## 1. Why introduce stochastic gradients?

The previous chapters studied gradient descent for smooth convex objectives and proximal gradient descent for composite objectives. Both methods use information from the full objective at every iteration.

For linear regression, the course writes

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

> [!NOTE]
> **The source's factor-$n$ speed statement**
>
> Page 72 describes one SGD iteration as $n$ times faster than one full-gradient iteration. This is a comparison of the number of per-sample gradient evaluations. It does not claim that SGD needs the same number of iterations or that implementation overhead is exactly zero.

## 2. Finite-sum optimization problem

The general problem considered by the course is

$$\min_{x\in\mathbb{R}^{d}}f(x)=\frac{1}{n}\sum_{i=1}^{n}f_{i}(x).$$

Each $f_{i}$ is usually the loss associated with the $i$-th training sample. The source gives squared loss for linear regression and hinge loss for SVMs as examples.

The component functions do not need to be identical:

$$f_{i}\neq f_{j}\quad\text{may hold when }i\neq j.$$

> [!NOTE]
> **Two normalization conventions**
>
> Page 71 writes the regression objective as an unnormalized sum, while page 72 defines the general finite-sum objective using the average $n^{-1}\sum_{i=1}^{n}f_{i}$. The two conventions differ by a constant factor, which also rescales the gradient and the effective learning rate. This chapter follows the averaged convention for the convergence theorem.

## 3. SGD algorithm for the finite-sum problem

Choose an initialization

$$x^{(0)}\in\mathbb{R}^{d}.$$

At iteration $k\geq 1$, randomly choose

$$i_{k}\in\{1,2,\ldots,n\}$$

and update

$$x^{(k)}=x^{(k-1)}-\eta\nabla f_{i_{k}}\left(x^{(k-1)}\right),$$

where $\eta>0$ is the learning rate.

The notebook presents a fixed learning rate and a fixed maximum number of iterations.

> [!TIP]
> **Why is the output averaged?**
>
> The sequence $x^{(1)},x^{(2)},\ldots$ is random because every update depends on a random index. The theorem controls the objective at the averaged iterate rather than directly controlling the final raw iterate.

## 4. Assumptions for the convergence result

The source uses three assumptions.

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

> [!NOTE]
> **Variance notation on page 72**
>
> The displayed assumption on page 72 is written using the variance of the gradient norm. The proof on page 74 instead uses the vector second-moment identity above. This chapter states the proof-consistent interpretation and does not silently treat the two expressions as identical.

## 5. Optimal point and averaged iterate

Because $f$ is convex, the source assumes that a global minimizer exists:

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

> [!NOTE]
> **The absolute value is unnecessary**
>
> Since $x^{\star}$ is a global minimizer and $\bar{x}^{(k)}$ is feasible, $f(\bar{x}^{(k)})-f(x^{\star})\geq 0$. The source states the theorem with an absolute value, but the proof bounds the nonnegative objective gap.

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

> [!TIP]
> **Conditional and total expectations**
>
> The source says to take expectation with respect to the random index $i_{k}$ at each step. Formally, this is a conditional expectation given all earlier sampled indices. Applying total expectation afterward produces the unconditional expectations in the theorem.

## 8. What the theorem does and does not say

The theorem guarantees convergence of the expected objective at the averaged iterate to a neighborhood whose size is controlled by $\eta\sigma^{2}$.

With a fixed $\eta$, the displayed upper bound does not go to zero as $k\to\infty$ unless $\sigma=0$. The source therefore points to separate reference material for almost-sure convergence and a different weighted averaging scheme, but those results are not derived in the notebook.

> [!NOTE]
> **Almost-sure convergence is only referenced**
>
> Page 74 cites the Canvas reference “Almost sure convergence rates for Stochastic Gradient Descent and Stochastic Heavy Ball.” The notebook states that it uses a different weighted average of the iterates and obtains a more recent almost-sure result. No theorem or proof from that reference is reproduced here.

## 9. Main takeaways

- Full gradient descent uses all $n$ component gradients at every iteration.
- SGD uses one randomly selected component gradient, reducing the work per update but introducing noise.
- Unbiasedness makes the stochastic gradient correct on average.
- A bounded-noise assumption controls its second moment.
- Under convexity and smoothness, a fixed-step SGD bound has an optimization term of order $1/k$ and a residual term $\eta\sigma^{2}$.
- Averaging the iterates is essential in the source proof because Jensen's inequality converts the average of objective values into an objective value at the average iterate.

---

[← Previous: Nonsmooth and Proximal Optimization](10_nonsmooth_proximal_optimization.md) · [Course map](../course_map.md) · [Next: Optimization of Neural Networks →](12_neural_network_optimization.md)
