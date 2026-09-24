# 9. Smooth Convex Optimization

The last part of the course turns from *whether* a learned model generalizes to *how* we compute it. This chapter proves the basic convergence rate of gradient descent, $O(1/k)$ for smooth convex objectives, using two inequalities (the descent lemma and first-order convexity) and a telescoping sum. It then states the accelerated $O(1/k^2)$ rate.

## 1. Optimization methods for machine learning

Three methods are covered in this part:

- gradient descent;
- proximal gradient descent;
- stochastic gradient descent.

Gradient descent is presented first because it is the foundation of the other two methods. Linear regression provides the initial concrete objective function.

## 2. Linear regression

The training set is

$$\mathcal{D}=\left\lbrace \left(x^{(i)},y^{(i)}\right)\right\rbrace_{i=1}^{m},$$

where each input-output pair satisfies

$$x^{(i)}\in\mathbb{R}^{d},\qquad y^{(i)}\in\mathbb{R}.$$

The input $x^{(i)}$ is a feature vector, and $y^{(i)}$ is the corresponding target or label.

For a two-feature example, the parameter vector is written as

$$\theta=\left[\theta_{0},\theta_{1},\theta_{2}\right]^{\top}.$$

The linear hypothesis is

$$h_{\theta}(x)=\theta_{0}x_{0}+\theta_{1}x_{1}+\theta_{2}x_{2}=\widehat{y},$$

where the intercept feature is fixed as

$$x_{0}=1.$$

Equivalently,

$$h_{\theta}(x)=x^{\top}\theta.$$

The goal is to make every prediction $\widehat{y}^{(i)}$ close to its ground-truth target $y^{(i)}$. We use the sum of squared errors:

$$J(\theta)=\sum_{i=1}^{m}\left(y^{(i)}-\widehat{y}^{(i)}\right)^{2}.$$

Substituting the linear prediction gives

$$J(\theta)=\sum_{i=1}^{m}\left(y^{(i)}-\theta_{0}x_{0}^{(i)}-\theta_{1}x_{1}^{(i)}-\cdots-\theta_{d}x_{d}^{(i)}\right)^{2}.$$

The optimization problem is

$$\min_{\theta\in\mathbb{R}^{d+1}}J(\theta).$$

> [!TIP]
> **Why begin with squared loss?**
>
> A quadratic objective is easy to optimize: squared loss supplies a differentiable convex function whose gradient can be computed explicitly, making it a convenient first example for gradient descent.

## 3. Vectorized formulation

To include the intercept in the matrix representation, define the augmented feature vector

$$\widetilde{x}^{(i)}=\left[1,x_{1}^{(i)},x_{2}^{(i)},\ldots,x_{d}^{(i)}\right]^{\top}\in\mathbb{R}^{d+1}.$$

The prediction for sample $i$ becomes

$$\widehat{y}^{(i)}=\left(\widetilde{x}^{(i)}\right)^{\top}\theta.$$

Stack the targets into

$$y=\left[y^{(1)},y^{(2)},\ldots,y^{(m)}\right]^{\top}\in\mathbb{R}^{m},$$

and stack the augmented feature vectors as rows of the design matrix:

$$X=\left[\left(\widetilde{x}^{(1)}\right)^{\top};\left(\widetilde{x}^{(2)}\right)^{\top};\ldots;\left(\widetilde{x}^{(m)}\right)^{\top}\right]\in\mathbb{R}^{m\times(d+1)}.$$

All predictions are then written compactly as

$$\widehat{y}=X\theta.$$

The objective becomes

$$J(\theta)=\lVert y-X\theta\rVert_{2}^{2}.$$

Equivalently,

$$J(\theta)=\left(y-X\theta\right)^{\top}\left(y-X\theta\right).$$

### Example: house-price prediction

Take five training examples with square footage and number of rooms as the observed features. After adding the intercept column, the design matrix has three columns. The fitted parameter vector is

$$\widehat{\theta}=\left[-235.63,0.19,73.13\right]^{\top}.$$

For a new house with $2000$ square feet and $3$ rooms, the augmented feature vector is

$$\widetilde{x}=\left[1,2000,3\right]^{\top},$$

and the predicted price is computed from

$$\widehat{y}=\widetilde{x}^{\top}\widehat{\theta}.$$

The fitted model predicts $371.25$ (thousand dollars).

> [!NOTE]
> **Rounded coefficients in the example**
>
> The coefficients are rounded to two decimals. Plugging the rounded values in gives $363.76$; the $371.25$ comes from the unrounded fit. A square-footage slope of about $0.1937$ instead of $0.19$ accounts for almost all of the gap ($0.0037\times2000\approx7.4$).

> [!NOTE]
> **Notation**
>
> $d$ is the number of original features, so with the intercept $\theta\in\mathbb{R}^{d+1}$. In the house example $d=2$ and $X$ has three columns.

## 4. Gradient descent for linear regression

The linear-regression objective is

$$J(\theta)=\lVert y-X\theta\rVert_{2}^{2}.$$

Its gradient is

$$\nabla J(\theta)=2X^{\top}\left(X\theta-y\right).$$

Gradient descent starts from an initialization $\theta^{(0)}$. At iteration $k\geq 1$, it performs the update

$$\theta^{(k)}=\theta^{(k-1)}-\eta\nabla J\left(\theta^{(k-1)}\right),$$

where

$$\eta>0$$

is the learning rate.

Two common stopping rules:

- stop when $\lVert\nabla J(\theta^{(k)})\rVert_{2}$ is smaller than a predetermined threshold $\varepsilon>0$;
- stop when a predetermined maximum number of iterations has been reached.

The geometric idea is that the negative gradient points in a local descent direction, so the update attempts to reduce the objective at each step.

## 5. General smooth convex optimization problem

The convergence analysis is stated for the general problem

$$\min_{x\in\mathbb{R}^{d}}f(x),$$

where $f:\mathbb{R}^{d}\to\mathbb{R}$ is convex and differentiable.

Let

$$x^{\star}\in\underset{x\in\mathbb{R}^{d}}{\arg\min}\;f(x)$$

be a global minimizer. Gradient descent starts from $x^{(0)}$ and updates

$$x^{(k)}=x^{(k-1)}-\eta\nabla f\left(x^{(k-1)}\right).$$

The desired conclusion is that

$$f\left(x^{(k)}\right)\to f\left(x^{\star}\right)$$

as the number of iterations increases.

> [!NOTE]
> **Differentiability versus smoothness**
>
> "Smooth" is used loosely for "differentiable", but the convergence theorem needs more: an $L$-Lipschitz gradient. Below, smooth always means that.

## 6. Lipschitz-continuous gradients

The gradient of $f$ is Lipschitz continuous with constant $L>0$ when

$$\lVert\nabla f(x)-\nabla f(y)\rVert_{2}\leq L\lVert x-y\rVert_{2}$$

for every $x,y\in\mathbb{R}^{d}$.

This condition controls how quickly the gradient can change. The constant $L$ determines a safe fixed learning rate.

> [!TIP]
> **Why does L affect the learning rate?**
>
> The theorem sets $\eta=1/L$. A smaller valid Lipschitz constant permits a larger guaranteed step size. A larger step size means fewer iterations, and the bound makes the dependence explicit through $1/(\eta k)$. For least squares, $J(\theta)=\lVert y-X\theta\rVert^2$ has $L=2\lambda_{\max}(X^\top X)$.

## 7. Descent lemma for smooth functions

### Lemma 1

Suppose $f$ is differentiable and its gradient is Lipschitz continuous with constant $L>0$. Then, for every $x,y\in\mathbb{R}^{d}$,

$$f(y)\leq f(x)+\nabla f(x)^{\top}(y-x)+\frac{L}{2}\lVert y-x\rVert_{2}^{2}.$$

The right-hand side is a quadratic upper approximation of $f(y)$ around $x$.

### Proof

Define the one-dimensional function

$$g(t)=f\left(x+t(y-x)\right),\qquad 0\leq t\leq 1.$$

Its derivative is

$$g'(t)=\nabla f\left(x+t(y-x)\right)^{\top}(y-x).$$

At $t=0$,

$$g'(0)=\nabla f(x)^{\top}(y-x).$$

The difference between the derivatives satisfies

$$\left\lvert g'(t)-g'(0)\right\rvert=\left\lvert\left(\nabla f\left(x+t(y-x)\right)-\nabla f(x)\right)^{\top}(y-x)\right\rvert.$$

By the Cauchy-Schwarz inequality,

$$\left\lvert g'(t)-g'(0)\right\rvert\leq\left\lVert\nabla f\left(x+t(y-x)\right)-\nabla f(x)\right\rVert_{2}\lVert y-x\rVert_{2}.$$

Using the Lipschitz-gradient assumption,

$$\left\lvert g'(t)-g'(0)\right\rvert\leq Lt\lVert y-x\rVert_{2}^{2}.$$

The fundamental theorem of calculus gives

$$g(1)=g(0)+\int_{0}^{1}g'(t)\,dt.$$

Add and subtract $g'(0)$ inside the integral:

$$g(1)=g(0)+g'(0)+\int_{0}^{1}\left(g'(t)-g'(0)\right)\,dt.$$

Therefore,

$$g(1)\leq g(0)+g'(0)+\int_{0}^{1}Lt\lVert y-x\rVert_{2}^{2}\,dt.$$

Evaluating the integral yields

$$g(1)\leq g(0)+g'(0)+\frac{L}{2}\lVert y-x\rVert_{2}^{2}.$$

Because $g(0)=f(x)$, $g(1)=f(y)$, and $g'(0)=\nabla f(x)^{\top}(y-x)$, we obtain

$$f(y)\leq f(x)+\nabla f(x)^{\top}(y-x)+\frac{L}{2}\lVert y-x\rVert_{2}^{2}.$$

## 8. First-order property of convexity

For a differentiable convex function $f$,

$$f(y)\geq f(x)+\nabla f(x)^{\top}(y-x).$$

Geometrically, the graph of a convex function lies above every tangent hyperplane. In one dimension, the tangent line at $x$ is

$$f(x)+f'(x)(y-x),$$

and the function value $f(y)$ cannot lie below that line.

Taking $y=x^{\star}$ gives

$$f\left(x^{\star}\right)\geq f(x)+\nabla f(x)^{\top}\left(x^{\star}-x\right).$$

Equivalently,

$$f(x)-f\left(x^{\star}\right)\leq\nabla f(x)^{\top}\left(x-x^{\star}\right).$$

## 9. Convergence theorem for gradient descent

### Theorem 1

Suppose $f:\mathbb{R}^{d}\to\mathbb{R}$ is convex and differentiable, and its gradient is Lipschitz continuous with constant $L>0$. Run gradient descent with the fixed learning rate

$$\eta=\frac{1}{L}.$$

Then, after $k\geq 1$ iterations,

$$0\leq f\left(x^{(k)}\right)-f\left(x^{\star}\right)\leq\frac{\lVert x^{\star}-x^{(0)}\rVert_{2}^{2}}{2\eta k}.$$

Consequently,

$$f\left(x^{(k)}\right)-f\left(x^{\star}\right)=O\left(\frac{1}{k}\right).$$

### Proof, Step 1: one gradient step decreases the objective

For an arbitrary point $x$, define

$$y=x-\eta\nabla f(x).$$

Applying Lemma 1 gives

$$f(y)\leq f(x)+\nabla f(x)^{\top}(y-x)+\frac{L}{2}\lVert y-x\rVert_{2}^{2}.$$

Because $y-x=-\eta\nabla f(x)$,

$$f(y)\leq f(x)-\eta\lVert\nabla f(x)\rVert_{2}^{2}+\frac{L\eta^{2}}{2}\lVert\nabla f(x)\rVert_{2}^{2}.$$

Thus,

$$f(y)\leq f(x)-\eta\left(1-\frac{L\eta}{2}\right)\lVert\nabla f(x)\rVert_{2}^{2}.$$

For $\eta\leq 1/L$,

$$1-\frac{L\eta}{2}\geq\frac{1}{2},$$

so

$$f(y)\leq f(x)-\frac{\eta}{2}\lVert\nabla f(x)\rVert_{2}^{2}.$$

This also shows that the objective values generated by the update are nonincreasing.

### Proof, Step 2: relate the objective gap to a change in distance

The convexity inequality gives

$$f(x)-f\left(x^{\star}\right)\leq\nabla f(x)^{\top}\left(x-x^{\star}\right).$$

Combining this with the one-step descent inequality yields

$$f(y)-f\left(x^{\star}\right)\leq\nabla f(x)^{\top}\left(x-x^{\star}\right)-\frac{\eta}{2}\lVert\nabla f(x)\rVert_{2}^{2}.$$

Rewrite the right-hand side as

$$f(y)-f\left(x^{\star}\right)\leq\frac{1}{2\eta}\left(2\eta\nabla f(x)^{\top}\left(x-x^{\star}\right)-\eta^{2}\lVert\nabla f(x)\rVert_{2}^{2}\right).$$

Using $y=x-\eta\nabla f(x)$, the expression in parentheses equals

$$\lVert x-x^{\star}\rVert_{2}^{2}-\lVert y-x^{\star}\rVert_{2}^{2}.$$

Therefore,

$$f(y)-f\left(x^{\star}\right)\leq\frac{1}{2\eta}\left(\lVert x-x^{\star}\rVert_{2}^{2}-\lVert y-x^{\star}\rVert_{2}^{2}\right).$$

### Proof, Step 3: apply the inequality to consecutive iterates

Set

$$x=x^{(k-1)},\qquad y=x^{(k)}.$$

Then

$$f\left(x^{(k)}\right)-f\left(x^{\star}\right)\leq\frac{1}{2\eta}\left(\lVert x^{(k-1)}-x^{\star}\rVert_{2}^{2}-\lVert x^{(k)}-x^{\star}\rVert_{2}^{2}\right).$$

Summing from $k=1$ to $k=K$ makes the distance terms telescope:

$$\sum_{k=1}^{K}\left(f\left(x^{(k)}\right)-f\left(x^{\star}\right)\right)\leq\frac{1}{2\eta}\left(\lVert x^{(0)}-x^{\star}\rVert_{2}^{2}-\lVert x^{(K)}-x^{\star}\rVert_{2}^{2}\right).$$

Since squared norms are nonnegative,

$$\sum_{k=1}^{K}\left(f\left(x^{(k)}\right)-f\left(x^{\star}\right)\right)\leq\frac{\lVert x^{(0)}-x^{\star}\rVert_{2}^{2}}{2\eta}.$$

### Proof, Step 4: use monotonicity of the objective values

Because the objective values are nonincreasing,

$$f\left(x^{(K)}\right)-f\left(x^{\star}\right)\leq\frac{1}{K}\sum_{k=1}^{K}\left(f\left(x^{(k)}\right)-f\left(x^{\star}\right)\right).$$

Combining the last two inequalities gives

$$f\left(x^{(K)}\right)-f\left(x^{\star}\right)\leq\frac{\lVert x^{(0)}-x^{\star}\rVert_{2}^{2}}{2\eta K}.$$

Renaming $K$ as $k$ completes the proof.

> [!TIP]
> **Why does the final iterate lie below the average?**
>
> The descent lemma shows $f(x^{(1)})\geq f(x^{(2)})\geq\cdots\geq f(x^{(K)})$. Therefore, the last objective gap is no larger than the average of the first $K$ objective gaps. This is the step that turns an average bound into a last-iterate bound.

## 10. Accelerated gradient descent

Can we do better than $O(1/k)$? Nesterov's **accelerated gradient descent** (AGD) can.

The idea is to add a momentum-like extrapolation before taking a gradient step.

Initialize

$$x^{(1)}=x^{(0)},\qquad t_{0}=0,\qquad t_{1}=1.$$

For $k\geq 1$, compute the extrapolated point

$$y^{(k)}=x^{(k)}+\frac{t_{k-1}-1}{t_{k}}\left(x^{(k)}-x^{(k-1)}\right).$$

Then take a gradient step from $y^{(k)}$:

$$x^{(k+1)}=y^{(k)}-\eta\nabla f\left(y^{(k)}\right).$$

Update the momentum parameter by

$$t_{k+1}=\frac{1+\sqrt{1+4t_{k}^{2}}}{2}.$$

Run for a fixed number of iterations (a tolerance on $\lVert\nabla f\rVert$ works too).

> [!TIP]
> **Can AGD also use a tolerance-based stopping rule?**
>
> Yes. A gradient-norm threshold works for AGD as well. A fixed iteration budget is just the natural way to state the theorem. One caveat: AGD is not a descent method (the objective can go up temporarily), so don't stop on "objective increased".

## 11. Convergence theorem for accelerated gradient descent

### Theorem 2

Suppose $f:\mathbb{R}^{d}\to\mathbb{R}$ is convex and differentiable, and its gradient is Lipschitz continuous with constant $L>0$. Run AGD with a fixed learning rate satisfying

$$\eta\leq\frac{1}{L}.$$

Then the iterate $x^{(k)}$ satisfies

$$0\leq f\left(x^{(k)}\right)-f\left(x^{\star}\right)\leq\frac{\lVert x^{\star}-x^{(0)}\rVert_{2}^{2}}{2\eta k^{2}}.$$

Therefore,

$$f\left(x^{(k)}\right)-f\left(x^{\star}\right)=O\left(\frac{1}{k^{2}}\right).$$

> [!NOTE]
> **Check the constant.** The standard guarantee for this scheme (Nesterov; Beck and Teboulle's FISTA with $h=0$) is $f(x^{(k)})-f(x^{\star})\leq\dfrac{2\lVert x^{(0)}-x^{\star}\rVert_{2}^{2}}{\eta(k+1)^{2}}$. The constant displayed above is smaller than the usual result, so treat it as the $O(1/k^{2})$ rate rather than an exact bound.

Comparing the rates:

$$\mathrm{GD}:\quad O\left(\frac{1}{k}\right),$$

$$\mathrm{AGD}:\quad O\left(\frac{1}{k^{2}}\right).$$

Thus, under the assumptions of the two theorems, AGD has the faster objective-value convergence guarantee.

> [!NOTE]
> **Beyond the lecture: where acceleration comes from**
>
> The proof (Beck & Teboulle, 2009) builds an estimate sequence: it tracks a potential $t_k^2\,(f(x^{(k)})-f^\star)+\frac{1}{2\eta}\lVert u^{(k)}-x^\star\rVert^2$ with $u^{(k)}$ an auxiliary "momentum" point, and shows it doesn't increase. Since $t_k\ge(k+1)/2$, the gap falls like $1/k^2$. The rate is optimal: no first-order method can beat $O(1/k^2)$ on this class of problems in the worst case (Nesterov's lower bound).

## 12. Chapter summary

1. Linear regression minimizes a quadratic sum-of-squares objective.
2. Vectorization expresses all predictions as $\widehat{y}=X\theta$ and the objective as $\lVert y-X\theta\rVert_{2}^{2}$.
3. Gradient descent repeatedly subtracts the gradient multiplied by a learning rate.
4. An $L$-Lipschitz gradient provides a quadratic upper bound through the descent lemma.
5. Convexity provides a tangent-hyperplane lower bound.
6. Combining the two inequalities produces a telescoping proof of the $O(1/k)$ gradient-descent rate.
7. Accelerated gradient descent adds an extrapolation step and achieves the stated $O(1/k^{2})$ rate.
8. The accelerated rate is optimal among first-order methods for smooth convex problems.

---

[← Previous: VC Dimension](08_vc_dimension.md) · [Course map](../course_map.md) · [Next: Nonsmooth and Proximal Optimization →](10_nonsmooth_proximal_optimization.md)
