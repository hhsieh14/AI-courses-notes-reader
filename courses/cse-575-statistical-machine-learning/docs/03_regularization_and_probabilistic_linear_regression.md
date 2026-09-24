# 3. Regularization and Probabilistic Linear Regression

Chapter 2 showed that flexible models overfit. This chapter gives the standard fix, penalizing large coefficients, and compares ridge, lasso and elastic net. It then re-derives least squares from probability: minimizing squared error is maximum likelihood under Gaussian noise. That second view explains where the loss functions in the rest of the course come from, and it turns regularization into a prior.

## 3.1 Why regularize?

A high-capacity model can fit accidental details of the training sample. Regularization adds a penalty to the data-fitting loss:

$$ L_{\text{reg}}(\theta) = L_{\text{data}}(\theta) + \lambda\,\Omega(\theta), $$

where $L_{\text{data}}$ measures fit (for polynomial regression, $\tfrac12\sum_i(y^{(i)}-h_\theta(x^{(i)}))^2$), $\Omega(\theta)$ penalizes the parameters, and $\lambda\ge 0$ sets the trade-off. $\lambda=0$ recovers ordinary least squares; larger $\lambda$ pushes the solution toward small coefficients.

> [!TIP]
> **Intuition**
>
> Regularization doesn't remove any function from the model class. It changes which functions are *preferred*: of two models with similar training error, the one with smaller coefficients now wins.

## 3.2 Penalty form and constraint form

The same idea can be written two ways:

$$ \text{penalty: } \min_\theta L_{\text{data}}(\theta)+\lambda\Omega(\theta) \qquad\qquad \text{constraint: } \min_\theta L_{\text{data}}(\theta) \;\text{ s.t. }\; \Omega(\theta)\leq c. $$

For convex problems they are equivalent: for every $\lambda$ there is a $c$ giving the same solution, and vice versa (the $\lambda$ is the Lagrange multiplier of the constraint). Larger $\lambda$ corresponds to smaller $c$. The penalty form is what we optimize; the constraint form is what we draw.

## 3.3 Ridge regression (L2)

$$ L_{\mathrm{ridge}}(\theta) = \frac{1}{2}\sum_{i=1}^{n}\left(y^{(i)}-h_\theta\left(x^{(i)}\right)\right)^2 + \lambda\sum_{j=1}^{d}\theta_j^2 . $$

The sum starts at $j=1$: the intercept $\theta_0$ is **not** penalized, since shifting all targets by a constant shouldn't change how "complex" the model is.

Ridge shrinks every coefficient, and it punishes large ones disproportionately: going from 1 to 2 raises the penalty from 1 to 4, going from 2 to 4 raises it from 4 to 16. On a degree-9 polynomial:

- $\lambda=0$: wild oscillations, overfitting;
- a small $\lambda$: a smooth curve that still follows the data;
- a large $\lambda$: coefficients squashed toward zero, underfitting.

Ridge also has a closed form (with centered data and no intercept):

$$ \hat\theta_{\text{ridge}} = (X^\top X + 2\lambda I)^{-1}X^\top y , $$

where the 2 comes from the $\tfrac12$ convention above. The matrix is invertible for any $\lambda>0$, which is why ridge also fixes the singular-$X^\top X$ problem from Chapter 1.

> [!WARNING]
> **Regularization can be too strong**
>
> Regularization trades variance for bias. It isn't automatically helpful at every strength; choose $\lambda$ by cross-validation.

## 3.4 Lasso regression (L1)

$$ L_{\mathrm{lasso}}(\theta) = \frac{1}{2}\sum_{i=1}^{n}\left(y^{(i)}-h_\theta\left(x^{(i)}\right)\right)^2 + \lambda\sum_{j=1}^{d}\left|\theta_j\right| . $$

Lasso shrinks selectively and can set coefficients **exactly** to zero. If $\theta_j=0$, feature $\phi_j(x)$ drops out of the model, so lasso performs feature selection as part of fitting. On a high-degree polynomial it typically keeps only a few terms.

$|\theta_j|$ isn't differentiable at 0, so plain gradient descent doesn't apply directly. There is no closed form either. In practice lasso is solved with **coordinate descent** (each coordinate has a closed-form soft-threshold update), **proximal gradient** methods (see [SLT Chapter 10](../../statistical-learning-theory/docs/chapters/10_nonsmooth_proximal_optimization.md)), or LARS, which computes the whole path over $\lambda$. These are fast. The honest difference from ridge is the missing closed form, not speed.

## 3.5 Why L1 gives sparse solutions

Let $\beta^\star$ be the unregularized minimizer. Contours of equal loss are ellipses around it, and the constrained solution is the point where the smallest contour first touches the feasible region:

$$ \text{L1: } |\beta_1|+|\beta_2|\leq c \;\;(\text{a diamond}), \qquad \text{L2: } \beta_1^2+\beta_2^2\leq c \;\;(\text{a disk}). $$

![L1 and L2 constraint geometry](assets/diagrams/03_l1_l2_geometry.svg)

*The expanding ellipse typically meets the diamond at a corner, where one coordinate is zero. The disk has no corners, so contact happens away from the axes and both coefficients shrink but stay nonzero.*

In higher dimensions the L1 ball (a cross-polytope) has corners and edges on every coordinate subspace, so sparse solutions become even more likely.

## 3.6 Ridge versus lasso

| | Ridge | Lasso |
|---|---|---|
| Penalty | $\sum_j\theta_j^2$ | $\sum_j\lvert\theta_j\rvert$ |
| Effect | shrinks all coefficients | shrinks some, zeroes others |
| Feature selection | no | yes |
| Constraint shape | sphere | cross-polytope with corners |
| Closed form | yes | no (coordinate descent / proximal methods) |
| With correlated features | spreads weight across them | tends to pick one arbitrarily |

## 3.7 Elastic net

Elastic net uses both penalties:

$$ L_{\mathrm{EN}}(\theta) = \frac{1}{2}\sum_{i}\left(y^{(i)}-h_\theta\left(x^{(i)}\right)\right)^2 + \lambda_1\sum_{j}\theta_j^2 + \lambda_2\sum_{j}\left|\theta_j\right| . $$

Here $\lambda_1$ weights the L2 term and $\lambda_2$ the L1 term. (scikit-learn instead uses one overall strength `alpha` and a mixing ratio `l1_ratio`.) Elastic net keeps lasso's sparsity but, thanks to the L2 part, handles groups of correlated features more stably. The cost is a second hyperparameter.

## 3.8 Regularization and bias–variance

As $\lambda$ increases, training fit gets worse, coefficients shrink, variance falls and bias rises. The goal is the $\lambda$ that minimizes validation error, not the smallest coefficient norm.

## 3.9 The probabilistic view: maximum likelihood

Switch perspectives. Suppose data come from an unknown distribution $p_{\mathrm{data}}$ and we have a parameterized model $p_{\mathrm{model}}(x;\theta)$. With i.i.d. samples $x^{(1)},\ldots,x^{(n)}$ the likelihood is

$$ L(\theta)=\prod_{i=1}^{n}p_{\mathrm{model}}\left(x^{(i)};\theta\right), \qquad \theta_{\mathrm{MLE}}=\mathrm{arg\,max}_{\theta}\,L(\theta). $$

A product of many numbers below 1 underflows in floating point, so we maximize the log-likelihood instead. The log is increasing, so the maximizer is the same:

$$ \ell(\theta)=\sum_{i=1}^{n}\log p_{\mathrm{model}}\left(x^{(i)};\theta\right). $$

## 3.10 Maximum likelihood minimizes KL divergence

$$ D_{\mathrm{KL}}\left(p_{\mathrm{data}}\,\|\,p_{\mathrm{model}}\right)=\mathbb{E}_{x\sim p_{\mathrm{data}}}\left[\log p_{\mathrm{data}}(x)-\log p_{\mathrm{model}}(x;\theta)\right]. $$

The first term doesn't depend on $\theta$, so minimizing the KL divergence is the same as maximizing $\mathbb{E}_{p_{\text{data}}}[\log p_{\text{model}}(x;\theta)]$. Replace the expectation with the training average and you get exactly the log-likelihood divided by $n$. **MLE = minimizing the forward KL from the data to the model.**

Reversing the arguments, $D_{\mathrm{KL}}(p_{\mathrm{model}}\,\|\,p_{\mathrm{data}})$, takes the expectation under the *model* and needs $\log p_{\text{data}}$, which we can't evaluate. So it doesn't give a usable training objective here. (Reverse KL does show up in variational inference, Chapter 10, where the roles are different.)

## 3.11 Gaussian noise gives least squares

Assume

$$ y=h_\theta(x)+\epsilon, \qquad \epsilon\sim\mathcal{N}(0,\sigma^2), $$

so that

$$ p(y\mid x;\theta)=\frac{1}{\sqrt{2\pi}\sigma}\exp\left(-\frac{\left(y-h_\theta(x)\right)^2}{2\sigma^2}\right), \qquad y\mid x;\theta\sim\mathcal{N}\left(h_\theta(x),\sigma^2\right). $$

For each input, the model predicts the center of a Gaussian over possible targets.

## 3.12 The likelihood of a regression dataset

$$ \ell(\theta)=\sum_{i=1}^{n}\log p\left(y^{(i)}\mid x^{(i)};\theta\right)=-n\log\left(\sqrt{2\pi}\sigma\right)-\frac{1}{2\sigma^2}\sum_{i=1}^{n}\left(y^{(i)}-h_\theta\left(x^{(i)}\right)\right)^2. $$

With $\sigma$ fixed, the first term is constant and $1/(2\sigma^2)>0$, so

$$ \mathrm{arg\,max}_{\theta}\,\ell(\theta)=\mathrm{arg\,min}_{\theta}\sum_{i=1}^{n}\left(y^{(i)}-h_\theta\left(x^{(i)}\right)\right)^2 . $$

**Maximizing Gaussian likelihood is minimizing the residual sum of squares.** Squared error is the negative log-likelihood of a Gaussian noise model, not just a convenient geometric choice. (If $\sigma$ is also estimated, its MLE is the mean squared residual, $\hat\sigma^2=\frac1n\sum_i r_i^2$.)

| View | Object | Choose $\theta$ to… |
|---|---|---|
| Least squares | prediction $h_\theta(x)$ | minimize squared residuals |
| Probabilistic | density $p(y\mid x;\theta)$ | maximize the probability of the observed targets |

The probabilistic view makes the hidden assumptions explicit: $h_\theta(x)$ is the conditional mean, residuals are Gaussian, the variance is the same everywhere, and examples are independent. When those fail, for example with heavy-tailed noise, squared error is no longer the natural loss (Laplace noise gives absolute error, for instance).

## 3.13 Regularization is a prior

> [!NOTE]
> **Beyond the lecture: MAP estimation**
>
> The two halves of this chapter connect. Put a prior $p(\theta)$ on the parameters and take the **maximum a posteriori** estimate:
>
> $$ \hat\theta_{\text{MAP}}=\mathrm{arg\,max}_\theta\;\log p(y\mid X,\theta)+\log p(\theta). $$
>
> - Gaussian prior $\theta_j\sim\mathcal N(0,\tau^2)$ gives $\log p(\theta)=-\tfrac{1}{2\tau^2}\sum_j\theta_j^2+\text{const}$: **ridge**, with $\lambda=\sigma^2/(2\tau^2)$ under the ½-RSS convention.
> - Laplace prior $p(\theta_j)\propto e^{-|\theta_j|/b}$ gives $\log p(\theta)=-\tfrac1b\sum_j|\theta_j|+\text{const}$: **lasso**, with $\lambda=\sigma^2/b$.
>
> So $\lambda$ encodes how strongly you believe coefficients are small. The Laplace density has a sharp peak at zero, which is the probabilistic counterpart of the diamond's corners.

## 3.14 Preview: latent-variable generative models

The same likelihood thinking extends to data that we believe are generated from hidden causes. An image of a handwritten digit, for example, is produced by a latent code $z$ describing which digit it is, the handwriting style, the stroke width, and so on. A generative model specifies $p(x\mid z)$ and a prior $p(z)$. A variational autoencoder learns both an encoder $q(z\mid x)$ and a decoder $p(x\mid z)$.

A question I noted here and return to later: when extra information (text, audio) is available, should it enter as an input, as part of $z$, or as a separate conditioning variable $p(x\mid z,c)$? Conditional VAEs take the last option. The machinery for training such models, the ELBO, is developed in [Chapter 10](10_gaussian_mixture_models_and_em.md).

## 3.15 Common mistakes

1. **Assuming larger $\lambda$ is always better.** It eventually underfits.
2. **Treating ridge and lasso as interchangeable.** Only lasso produces exact zeros.
3. **Penalizing the intercept.** Usually excluded.
4. **Regularizing unscaled features.** The penalty treats all coefficients equally, so a feature measured in millimetres is penalized differently than the same feature in metres. Standardize first.
5. **Reading small coefficients as unimportant** without considering feature scale.
6. **Multiplying probabilities directly.** Work in log space.
7. **Reversing the KL and calling it maximum likelihood.**
8. **Using squared error without noticing the Gaussian, constant-variance assumption.**

## 3.16 Summary

- Regularization adds $\lambda\Omega(\theta)$ to the data loss.
- Ridge (L2) shrinks all coefficients and has a closed form; lasso (L1) produces sparse solutions via the corner geometry.
- Elastic net combines them.
- MLE minimizes the forward KL divergence from data to model.
- Under Gaussian noise, MLE is least squares.
- Ridge and lasso are MAP estimates under Gaussian and Laplace priors.

## 3.17 Self-check

1. What roles do $L_{\text{data}}$, $\Omega$ and $\lambda$ play?
2. Why does ridge penalize large coefficients more than small ones?
3. Why can lasso produce exact zeros while ridge generally can't?
4. How are the penalty and constraint forms related?
5. Why optimize log-likelihood rather than likelihood?
6. Why does minimizing $D_{\mathrm{KL}}(p_{\text{data}}\|p_{\text{model}})$ give MLE?
7. Which assumptions make least squares the MLE?
8. Which prior corresponds to lasso?

<details>
<summary>Answers</summary>

1. Fit, complexity penalty, and the trade-off weight between them.
2. The penalty is quadratic, so its marginal cost $2\lambda\theta_j$ grows with $|\theta_j|$.
3. The L1 ball has corners on the axes; the L1 subgradient at 0 is an interval, so zero can be optimal for a range of data. L2's gradient at 0 is 0, so there's no force holding a coefficient exactly at zero.
4. For convex problems each $\lambda$ corresponds to some radius $c$ (Lagrange duality); larger $\lambda$ ↔ smaller $c$.
5. The log turns products into sums, avoids underflow, and keeps the same maximizer.
6. The entropy of $p_{\text{data}}$ doesn't depend on $\theta$; what remains is the expected log-likelihood, estimated by the sample average.
7. Additive, zero-mean Gaussian noise with constant variance, and independent examples.
8. A zero-mean Laplace prior.

</details>
