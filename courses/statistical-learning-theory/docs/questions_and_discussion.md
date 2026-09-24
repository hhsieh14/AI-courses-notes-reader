# Questions and Answers

These are the questions I kept asking myself while working through the course, with the answers I settled on. Try each one before opening the answer.

## Probabilistic prediction

<details><summary>Why do we need learning if the Bayes classifier is already optimal?</summary>

It needs $\eta(x)=P(Y=1\mid X=x)$, which depends on the unknown distribution. Learning estimates it (or the decision rule) from finite data.
</details>

<details><summary>Why does partitioning the input space help estimate a conditional probability?</summary>

With continuous inputs we rarely see the same $x$ twice. Pooling points in a cell treats nearby inputs as comparable, so we can count labels.
</details>

<details><summary>How does cell size trade accuracy against sample size?</summary>

Small cells approximate $\eta$ locally (low bias) but hold few points (high variance); large cells do the reverse.
</details>

<details><summary>Why does disagreeing with the Bayes classifier matter less near η(x) = 1/2?</summary>

The excess-risk weight is $|2\eta(x)-1|$, which vanishes at $1/2$: both labels are almost equally likely there.
</details>

## SVM and duality

<details><summary>Why encode labels as {−1, 1}?</summary>

Both correctness constraints become the single inequality $y_i(w^\top x_i+b)\ge1$.
</details>

<details><summary>Why is max-margin equivalent to minimizing ‖w‖² under canonical constraints?</summary>

With $\min_iy_i(w^\top x_i+b)=1$, the margin is $1/\lVert w\rVert$; maximizing it is minimizing $\lVert w\rVert$, and squaring is monotone.
</details>

<details><summary>Why solve the dual instead of the primal?</summary>

The dual has one variable per example and depends only on inner products, so it can be kernelized and handles infinite-dimensional features. Its solution is also sparse (support vectors).
</details>

<details><summary>Which assumptions make the KKT conditions sufficient here?</summary>

Convexity plus Slater's condition; for separable data a strictly feasible point exists.
</details>

<details><summary>Why does w* depend only on support vectors?</summary>

$w^\star=\sum_i\alpha_i^\star y_ix_i$, and complementary slackness sets $\alpha_i^\star=0$ off the margin.
</details>

## Kernels and RKHS

<details><summary>Why must a kernel be positive definite?</summary>

$\sum_{ij}\alpha_i\alpha_jK(u_i,u_j)=\lVert\sum_i\alpha_i\phi(u_i)\rVert^2\ge0$; any kernel that is an inner product must satisfy it.
</details>

<details><summary>What fails if the inner product is only non-negative, not definite?</summary>

Nonzero functions could have zero norm, so the norm wouldn't separate points. For kernel sections this can't happen because $|f(x)|\le\lVert f\rVert\sqrt{K(x,x)}$.
</details>

<details><summary>What is the reproducing property, and why is it useful?</summary>

$f(x)=\langle f,K(\cdot,x)\rangle$. Evaluating a function becomes an inner product, so losses on training points become linear functionals, which is the key to the representer theorem.
</details>

<details><summary>How is the pre-Hilbert space completed into an RKHS?</summary>

Add limits of Cauchy sequences of finite kernel combinations. The limits are still functions, and the inner product and reproducing property extend by continuity.
</details>

<details><summary>Why does the representer theorem restrict the optimizer to the span of training features?</summary>

Any component orthogonal to that span doesn't change training predictions but increases the norm, so the optimizer removes it.
</details>

## Concentration and generalization

<details><summary>Which assumptions does each concentration inequality need?</summary>

Markov: non-negativity. Chebyshev: finite variance. Hoeffding: independence and bounded ranges. McDiarmid: independence and bounded differences.
</details>

<details><summary>Why are moment-generating functions useful for tails?</summary>

$e^{sX}$ turns a sum of independent variables into a product whose expectation factorizes, and optimizing over $s$ gives exponential tails.
</details>

<details><summary>Why does Hoeffding's lemma only need a second-order Taylor expansion?</summary>

Taylor's theorem with the Lagrange remainder is exact, and the second derivative is uniformly bounded by $1/4$.
</details>

<details><summary>Why can't the standard hinge loss go directly into the bounded-loss Hoeffding step?</summary>

It is unbounded above. Use the clipped hinge, which is bounded, 1-Lipschitz, and still upper-bounds the 0-1 loss.
</details>

<details><summary>How does bounded coordinate sensitivity bound the martingale differences in McDiarmid's proof?</summary>

Given the first $i-1$ coordinates, $V_i$ ranges between the inf and sup over $x_i$ of a conditional expectation of $g$; bounded differences make that range at most $c_i$.
</details>

<details><summary>Why does a union bound fail for an infinite collection?</summary>

Its failure probability grows linearly in the number of sets, so it becomes infinite.
</details>

<details><summary>What is the difference between a pointwise and a uniform generalization bound?</summary>

Pointwise holds for one fixed $f$ chosen before seeing data. Uniform holds simultaneously for all $f\in\mathcal F$, so it covers the data-dependent ERM output.
</details>

<details><summary>Why does complexity appear through an expected supremum?</summary>

$\sup_f(R-\widehat R_n)$ concentrates around its mean (McDiarmid), so the mean is what's left to bound, and it grows with the richness of $\mathcal F$.
</details>

<details><summary>Why does the supremum make the bound valid for ERM?</summary>

The ERM output is some element of $\mathcal F$, and its gap is at most the sup.
</details>

<details><summary>What does the ghost sample do?</summary>

It replaces $R(f)$ by an average over an independent copy, so the gap becomes a difference of two sample averages that can be symmetrized.
</details>

<details><summary>Why do random signs measure a class's ability to fit noise?</summary>

$\sup_f\frac1n\sum\sigma_if(x_i)$ is large exactly when some $f$ correlates with arbitrary random labels.
</details>

<details><summary>What does the contraction principle need?</summary>

A loss that is $M$-Lipschitz in the prediction, applied coordinate-wise.
</details>

<details><summary>Why does an RKHS norm bound control Rademacher complexity?</summary>

Cauchy–Schwarz gives $\sup_{\lVert w\rVert\le B}\langle w,\sum\sigma_i\phi(x_i)\rangle=B\lVert\sum\sigma_i\phi(x_i)\rVert$, and Jensen bounds that by $B\sqrt{\sum K(x_i,x_i)}$.
</details>

<details><summary>Why does the SVM regularizer have both a margin and a generalization interpretation?</summary>

Small $\lVert w\rVert$ means a large geometric margin, and it is also a small RKHS ball, i.e. low Rademacher complexity.
</details>

## VC dimension

<details><summary>Why must a shattered set realize all 2^|S| subsets?</summary>

VC dimension asks whether the class can fit every labeling, including adversarial ones.
</details>

<details><summary>Why can intervals shatter two points but not three?</summary>

$\{A,C\}$ without the middle point $B$ can't be cut out by an interval.
</details>

<details><summary>What condition lets three points in the plane be shattered by half-spaces?</summary>

They must not be collinear.
</details>

<details><summary>Why doesn't one non-shattered 4-point set prove VC ≤ 3?</summary>

The claim is about every 4-point set; Radon's theorem covers them all.
</details>

<details><summary>How do VC dimension and Rademacher complexity differ?</summary>

VC is combinatorial and distribution-free; Rademacher depends on the data distribution and handles real-valued, norm-bounded classes. Sauer–Shelah links them: $\mathfrak R_n\lesssim\sqrt{d\log n/n}$.
</details>

## Optimization

<details><summary>Why is quadratic loss a convenient first objective?</summary>

It is convex, smooth, has an explicit gradient and Hessian, and its smoothness constant is $2\lambda_{\max}(X^\top X)$.
</details>

<details><summary>Why add an intercept feature x₀ = 1?</summary>

It turns the affine model into a linear one, $x^\top\theta$, so the bias is just another coordinate.
</details>

<details><summary>How does L constrain the learning rate?</summary>

The descent lemma guarantees decrease only for $\eta\le1/L$ (any $\eta<2/L$ still decreases, with a smaller constant).
</details>

<details><summary>Why does the descent lemma give an upper bound while convexity gives a lower bound?</summary>

A Lipschitz gradient limits curvature from above (a quadratic cap); convexity says the function lies above every tangent plane.
</details>

<details><summary>Why do the distance terms telescope?</summary>

Each step's bound is $\frac{1}{2\eta}(\lVert x^{(k-1)}-x^\star\rVert^2-\lVert x^{(k)}-x^\star\rVert^2)$, so consecutive terms cancel.
</details>

<details><summary>Why is the last gap no larger than the average gap?</summary>

Gradient descent with $\eta\le1/L$ is monotone, so the last value is the smallest.
</details>

<details><summary>What gives accelerated gradient descent its faster rate?</summary>

Momentum with a carefully growing coefficient $(t_{k-1}-1)/t_k$; an estimate-sequence potential decreases by a factor tied to $t_k^2\sim k^2$.
</details>

<details><summary>Can AGD stop on a tolerance?</summary>

Yes, on $\lVert\nabla f\rVert$; but AGD isn't monotone, so don't stop on "objective increased".
</details>

<details><summary>Why split an objective into g + h?</summary>

$g$ is smooth (use its gradient), and $h$ is nonsmooth but has an easy prox (handle it exactly).
</details>

<details><summary>Why does the proximal map solve an optimization problem instead of taking a gradient step on h?</summary>

$h$ may have no gradient, and the prox is well defined for any convex $h$. It is an implicit (backward) step, which is more stable.
</details>

<details><summary>Why does soft thresholding give exact zeros?</summary>

Every input in $[-\lambda\eta,\lambda\eta]$ maps to exactly 0.
</details>

<details><summary>Why is ∂|x| at 0 the whole interval [−1, 1]?</summary>

Every slope in $[-1,1]$ gives a line through the origin that stays below $|x|$.
</details>

<details><summary>How does the proximal optimality condition give a subgradient inequality?</summary>

$(u-v^\star)/\eta\in\partial h(v^\star)$, and the subgradient definition turns that into $h(v)\ge h(v^\star)+\frac1\eta(u-v^\star)^\top(v-v^\star)$.
</details>

<details><summary>Where is η ≤ 1/L used in the proximal-gradient proof?</summary>

To get $\frac1\eta-\frac L2\ge\frac{1}{2\eta}$, which makes the quadratic term telescope-friendly.
</details>

## Stochastic gradient descent

<details><summary>How does gradient noise enter the analysis?</summary>

Through $\mathbb E\lVert v_k\rVert^2\le\sigma^2+\lVert\nabla f\rVert^2$; it adds the $\eta\sigma^2$ term.
</details>

<details><summary>Why is a uniformly sampled component gradient unbiased?</summary>

$\mathbb E[\nabla f_{i_k}]=\frac1n\sum_i\nabla f_i=\nabla f$.
</details>

<details><summary>Why bound the averaged iterate instead of the last one?</summary>

SGD isn't monotone. Averaging plus Jensen converts the average of $f$-values into $f$ at the average.
</details>

<details><summary>Why does a fixed learning rate leave an ησ² floor?</summary>

Each step injects noise of size $\eta^2\sigma^2$ that the descent can't cancel. Use $\eta\propto1/\sqrt k$ or mini-batches.
</details>

## Neural-network optimization

<details><summary>Why fix the output signs and train only the hidden layer?</summary>

It keeps the analysis linear in the trainable parameters' tangent space; training both layers gives the same conclusions with more bookkeeping.
</details>

<details><summary>Which iteration index belongs in the activation indicator?</summary>

$w_r(k-1)$, the point where the gradient is evaluated.
</details>

<details><summary>Why does over-parameterization keep H(k) close to H(0)?</summary>

Each weight moves $O(1/\sqrt m)$, so only a vanishing fraction of ReLU patterns flip.
</details>

<details><summary>Why is λ₀ = λ_min(K) > 0 essential?</summary>

The residual contracts by $(1-\eta\lambda_0)$ per step; with $\lambda_0=0$ some direction never shrinks.
</details>

<details><summary>Why does staying near initialization limit complexity?</summary>

The reachable functions lie in a small parameter ball, whose Rademacher complexity scales with its radius $\sqrt{y^\top K^{-1}y}$.
</details>

<details><summary>How does the eigendecomposition of K explain y⊤K⁻¹y?</summary>

$y^\top K^{-1}y=\sum_j(u_j^\top y)^2/\lambda_j$: labels aligned with large-eigenvalue directions are cheap to learn and generalize well.
</details>

<details><summary>Does the network directly minimize population risk?</summary>

No. It minimizes empirical squared loss; the theorem bounds population risk afterwards.
</details>
