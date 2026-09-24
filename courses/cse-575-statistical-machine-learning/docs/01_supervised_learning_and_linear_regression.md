# 1. Supervised Learning and Linear Regression

This chapter sets up the vocabulary used for the rest of the course: what "learning" means, how supervised problems are posed, and how the simplest useful model, linear regression, is fit two different ways, in closed form and by gradient descent.

## 1.1 What does it mean for a machine to learn?

Two classic definitions are worth keeping side by side.

- **Arthur Samuel (1959):** machine learning gives computers the ability to learn without being explicitly programmed.
- **Tom Mitchell (1997):** a program *learns* from experience $E$ with respect to a task $T$ and performance measure $P$ if its performance on $T$, as measured by $P$, improves with $E$.

Mitchell's version is the one I find useful in practice, because it forces three decisions:

| Symbol | Meaning | Spam-filter example |
|---|---|---|
| $T$ (task) | what the system must do | label an email as spam / not spam |
| $P$ (performance) | how we score it | fraction of emails labeled correctly, or recall on spam |
| $E$ (experience) | what it learns from | a set of emails already labeled by users |

> [!TIP]
> **Intuition**
>
> "The model learns" is vague until $T$, $P$ and $E$ are written down. Most confusion in ML projects comes from one of the three being left implicit, usually $P$.

## 1.2 Three modes of learning

1. **Supervised learning:** every training input comes with a known target.
2. **Unsupervised learning:** we only have inputs and look for structure in them (Chapters 9–11).
3. **Reinforcement learning:** an agent learns from rewards obtained by acting; not covered in this course.

A supervised training set is written

$$ \mathcal{D}_{\mathrm{train}} = \left\lbrace \left(x^{(i)}, y^{(i)}\right)\right\rbrace_{i=1}^{n_{\mathrm{train}}}, $$

where $x^{(i)}$ is the input of example $i$ and $y^{(i)}$ its observed target. An unsupervised dataset drops the targets:

$$ \mathcal{D}_{\mathrm{train}} = \left\lbrace x^{(i)}\right\rbrace_{i=1}^{n_{\mathrm{train}}}. $$

## 1.3 Regression and classification

| Task | Output | Examples |
|---|---|---|
| Regression | a real number, $y^{(i)}\in\mathbb{R}$ | hours studied → exam score; movie metadata → box-office revenue |
| Classification | a category, e.g. $y^{(i)}\in\lbrace 0,1\rbrace$ | spam vs. not spam; which animal is in a photo |

Every supervised method has the same two phases:

| Phase | What happens |
|---|---|
| **Training** | Combine labeled data with a parameterized model and choose the parameters that fit the data. |
| **Prediction** | Apply the fitted model to new inputs whose answers are unknown. |

## 1.4 Hypotheses and parameters

A model is written $h_\theta(x)$ or $h(x;\theta)$, where $\theta$ collects the learned parameters. The older name for this object is a **hypothesis**: an assumed form for the relationship between input and output.

Training picks the parameters that minimize a loss over the training set:

$$ \theta^{\star} = \mathrm{arg\,min}_{\theta}\; L\left(\left\lbrace y^{(i)}, h_\theta\left(x^{(i)}\right)\right\rbrace_{i=1}^{n_{\mathrm{train}}}\right). $$

The form of $L$ depends on the problem. For linear regression it is squared error, which we will later see is not an arbitrary choice (Chapter 3 shows it is the negative log-likelihood under Gaussian noise).

## 1.5 The one-feature linear model

The simplest regression hypothesis is a line:

$$ h_\theta(x) = \theta_0 + \theta_1x, \qquad \theta = \left[\theta_0,\theta_1\right]^\top, $$

with $\theta_0$ the intercept (bias) and $\theta_1$ the slope. Each choice of $\theta$ is a different candidate line; learning means choosing the best one under the loss.

## 1.6 Measuring error

Before choosing a loss it helps to recall the common distances between vectors $x^{(1)}$ and $x^{(2)}$:

$$ d_1 = \sum_j \left|x_j^{(1)}-x_j^{(2)}\right|, \qquad d_2 = \left(\sum_j \left|x_j^{(1)}-x_j^{(2)}\right|^2\right)^{1/2}, \qquad d_p = \left(\sum_j \left|x_j^{(1)}-x_j^{(2)}\right|^p\right)^{1/p}. $$

For regression we measure the vertical gap between the observation and the prediction, the **residual**

$$ r^{(i)} = y^{(i)} - h_\theta\left(x^{(i)}\right), $$

and minimize half the residual sum of squares (RSS):

$$ L(\theta) = \frac{1}{2}\sum_{i=1}^{n_{\mathrm{train}}}\left(y^{(i)}-h_\theta\left(x^{(i)}\right)\right)^2. $$

The mean squared error divides by $n_{\mathrm{train}}$ instead of 2:

$$ L_{\mathrm{MSE}}(\theta) = \frac{1}{n_{\mathrm{train}}}\sum_{i=1}^{n_{\mathrm{train}}}\left(y^{(i)}-h_\theta\left(x^{(i)}\right)\right)^2. $$

> [!NOTE]
> **RSS, MSE and the ½**
>
> These differ only by a positive constant, so they have the same minimizer. The $\tfrac12$ is there purely so the 2 from differentiating a square cancels. What *does* matter in practice: MSE's scale does not grow with the dataset, which makes learning rates transferable between dataset sizes.

## 1.7 Vector form

Append a constant 1 to each input so the intercept becomes an ordinary weight:

$$ \tilde{x}^{(i)} = \left[1,x^{(i)}\right]^\top, \qquad h_\theta\left(x^{(i)}\right) = \left(\tilde{x}^{(i)}\right)^\top\theta. $$

Stack the augmented inputs as rows of the **design matrix** $X$ (row $i$ is $(\tilde{x}^{(i)})^\top$) and the targets into $y=[y^{(1)},\ldots,y^{(n_{\mathrm{train}})}]^\top$. All predictions are then $X\theta$ and all residuals $y-X\theta$.

## 1.8 Least squares in closed form

In matrix form the loss is

$$ L(\theta) = \frac{1}{2}(y-X\theta)^\top(y-X\theta) = \frac{1}{2}\left(y^\top y - 2y^\top X\theta + \theta^\top X^\top X\theta\right). $$

Its gradient is

$$ \nabla_\theta L(\theta) = X^\top X\theta - X^\top y. $$

The loss is a convex quadratic, so any point with zero gradient is a global minimum. Setting the gradient to zero gives the **normal equations**

$$ X^\top X\theta = X^\top y, $$

and, when $X^\top X$ is invertible,

$$ \theta^{\star} = (X^\top X)^{-1}X^\top y. $$

> [!NOTE]
> **Beyond the lecture: when $X^\top X$ is singular, and how it is really computed**
>
> $X^\top X$ is singular whenever features are perfectly collinear or there are more features than examples. Then the normal equations have infinitely many solutions; the minimum-norm one is $\theta^\star = X^{+}y$ with $X^{+}$ the Moore–Penrose pseudo-inverse. Adding a ridge penalty (Chapter 3) makes the matrix $X^\top X+\lambda I$ invertible for any $\lambda>0$.
>
> In code, never form the inverse. `numpy.linalg.lstsq` solves the problem through a QR or SVD factorization, which is faster and much more numerically stable than `inv(X.T @ X) @ X.T @ y`, whose error grows with the *square* of $X$'s condition number.

## 1.9 Gradient descent

Gradient descent is the general-purpose alternative. Start from $\theta^{(0)}$ and repeatedly step against the gradient:

$$ \theta^{(k+1)} = \theta^{(k)} - \eta\nabla_\theta L\left(\theta^{(k)}\right), $$

where $\eta>0$ is the **learning rate**. For intuition take $f(x)=x^2$ with $f'(x)=2x$: when the slope is positive, moving left lowers $f$; when it is negative, moving right does. The negative gradient always points downhill locally.

For the one-feature model,

$$ \frac{\partial L}{\partial\theta_0} = -\sum_{i}\left(y^{(i)}-\theta_0-\theta_1x^{(i)}\right), \qquad \frac{\partial L}{\partial\theta_1} = -\sum_{i}\left(y^{(i)}-\theta_0-\theta_1x^{(i)}\right)x^{(i)}, $$

so the **simultaneous** update is

$$ \theta_0^{(k+1)} = \theta_0^{(k)} + \eta\sum_i\left(y^{(i)}-\theta_0^{(k)}-\theta_1^{(k)}x^{(i)}\right), $$

$$ \theta_1^{(k+1)} = \theta_1^{(k)} + \eta\sum_i\left(y^{(i)}-\theta_0^{(k)}-\theta_1^{(k)}x^{(i)}\right)x^{(i)}. $$

In matrix form this is simply

$$ \theta^{(k+1)} = \theta^{(k)} - \eta X^\top\left(X\theta^{(k)}-y\right). $$

| | Closed form | Gradient descent |
|---|---|---|
| Result | exact optimum in one solve | approaches the optimum iteratively |
| Cost | $O(nd^2+d^3)$ | $O(nd)$ per step |
| Works for | this quadratic loss | any differentiable loss |

> [!NOTE]
> **Beyond the lecture: how large can $\eta$ be?**
>
> For this loss, gradient descent converges iff $0<\eta<2/\lambda_{\max}(X^\top X)$. Too large and the iterates oscillate and blow up; too small and progress is slow along the directions with small eigenvalues. This is also why standardizing features (Chapter 4) speeds up gradient descent: it makes the eigenvalues of $X^\top X$ more similar. The general theory is in [Statistical Learning Theory, Chapter 9](../../statistical-learning-theory/docs/chapters/09_smooth_convex_optimization.md).

```python
import numpy as np

rng = np.random.default_rng(0)
x = rng.uniform(0, 10, 50)
y = 3.0 + 2.0 * x + rng.normal(0, 1, 50)
X = np.column_stack([np.ones_like(x), x])

theta_closed, *_ = np.linalg.lstsq(X, y, rcond=None)

theta = np.zeros(2)
eta = 1.0 / np.linalg.eigvalsh(X.T @ X).max()   # safe step size
for _ in range(20_000):
    theta -= eta * X.T @ (X @ theta - y)

print(theta_closed, theta)   # both [2.757, 2.052]; the true values are [3, 2]
```

## 1.10 Nonlinear features, still linear regression

Linear regression can fit curves by adding transformed features:

$$ h_\theta(x) = \theta_0 + \theta_1x + \theta_2x^2 + \theta_3x^3 = \theta^\top\phi(x), \qquad \phi(x) = \left[1,x,x^2,x^3\right]^\top. $$

The prediction is nonlinear in $x$, but it is still **linear in the parameters** $\theta$, so every formula above applies with $X$ built from $\phi(x^{(i)})$. Logarithms, sines, interactions and so on work the same way.

Raising the polynomial degree makes the model more flexible. A low degree misses the curve; a very high degree fits the noise and oscillates between points. That tension is the subject of Chapter 2.

## 1.11 Common mistakes

1. **Confusing the model with the algorithm.** The model is $h_\theta$; least squares and gradient descent are two ways to choose $\theta$.
2. **Calling every squared-error expression "MSE".** A sum, a mean and a half-sum have the same minimizer but different scales.
3. **Updating coordinates in place.** In the displayed update every coordinate uses the values from step $k$. Updating $\theta_0$ first and then using the new $\theta_0$ for $\theta_1$ is a different algorithm.
4. **Thinking polynomial regression is nonlinear regression.** It is linear in the coefficients.
5. **Inverting $X^\top X$ blindly.** It may be singular or badly conditioned; solve with `lstsq` or add regularization.

## 1.12 Summary

- Supervised learning fits a parameterized mapping from inputs to known targets.
- Regression predicts numbers; classification predicts categories.
- Linear regression minimizes squared residuals.
- Setting the gradient to zero gives the normal equations and a closed-form solution.
- Gradient descent reaches the same solution iteratively and generalizes to other losses.
- Nonlinear features keep the model linear in its parameters.

## 1.13 Self-check

1. What are $T$, $P$ and $E$ for a spam classifier?
2. What is the difference between a residual and the loss?
3. Why does the factor $\tfrac12$ not change the minimizer?
4. Derive the normal equations from the matrix form of the loss.
5. When does the closed-form solution not exist, and what can you do instead?
6. Why is cubic polynomial regression still "linear" regression?
7. What limits the learning rate in gradient descent for least squares?

<details>
<summary>Answers</summary>

1. $T$: classify emails; $P$: e.g. accuracy, or recall at a fixed false-positive rate; $E$: a labeled email corpus.
2. A residual is one example's error $y^{(i)}-h_\theta(x^{(i)})$; the loss aggregates all squared residuals into one number.
3. Multiplying a function by a positive constant does not move its minimum.
4. Expand $\tfrac12(y-X\theta)^\top(y-X\theta)$, differentiate to get $X^\top X\theta-X^\top y$, set it to zero.
5. When $X^\top X$ is singular (collinear features or $d>n$). Use the pseudo-inverse, `lstsq`, or ridge regression.
6. Because the prediction is a linear combination of the parameters; only the features are nonlinear.
7. The largest eigenvalue of $X^\top X$: convergence needs $\eta<2/\lambda_{\max}$.

</details>
