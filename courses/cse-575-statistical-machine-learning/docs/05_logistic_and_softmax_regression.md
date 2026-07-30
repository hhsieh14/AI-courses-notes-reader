# 5. Logistic and Softmax Regression

**Source pages:** 19–23  
**Status:** reconstructed and equation-checked

This chapter moves from similarity-based classification to parameterized probabilistic classifiers. The source first explains why ordinary linear regression is poorly matched to binary labels, introduces the sigmoid function and Bernoulli likelihood, extends logistic regression to multiple input features and one-vs-all classification, and then develops softmax regression for mutually exclusive multiclass outcomes. Page 23 connects linear, logistic, and softmax regression through the exponential family and generalized linear models.

## 5.1 Why ordinary linear regression is not a binary classifier

Suppose the target is binary:

$$ y\in\lbrace 0,1\rbrace. $$

A linear-regression model produces an unrestricted score:

$$ h_\theta(x)=\theta_0+\theta_1x. $$

Page 19 illustrates a threshold rule that predicts one class when the linear output exceeds $0.5$ and the other class otherwise. The linear output is not constrained to the interval $[0,1]$, so it is not naturally interpretable as a class probability. Thresholding that unrestricted output also does not provide a Bernoulli probability model for the binary target.

The source therefore replaces the unrestricted linear output with a function that maps every real-valued score to a number between zero and one.

> [!TIP]
> **Review intuition**
>
> Logistic regression keeps a linear score, but it does not use that score directly as the prediction. It passes the score through a nonlinear probability mapping.


## 5.2 The logistic or sigmoid function

The logistic function is:

$$ \sigma(z)=\frac{1}{1+e^{-z}}. $$

Its range is:

$$ 0<\sigma(z)<1. $$

The source also records its derivative:

$$ \frac{d\sigma(z)}{dz}=\sigma(z)\left(1-\sigma(z)\right). $$

Logistic regression uses the linear score:

$$ z=\theta^\top x, $$

and defines the hypothesis:

$$ h_\theta(x)=\sigma\left(\theta^\top x\right)=\frac{1}{1+e^{-\theta^\top x}}. $$

For a one-dimensional input with an intercept, this is:

$$ h_\theta(x)=\frac{1}{1+e^{-\left(\theta_0+\theta_1x\right)}}. $$

As the score $\theta^\top x$ becomes very negative, the predicted value approaches zero. As the score becomes very positive, it approaches one.

![Redrawn logistic and softmax intuition](assets/diagrams/05_logistic_softmax_intuition.svg)

*Redrawn from the ideas on source pages 19, 21, and 22: a sigmoid converts a linear score into a binary probability, the threshold corresponds to a linear decision boundary, and softmax normalizes multiple class scores.*

## 5.3 Probability interpretation and the decision threshold

The probabilistic interpretation is:

$$ h_\theta(x)=p(y=1\mid x;\theta). $$

Therefore:

$$ p(y=0\mid x;\theta)=1-h_\theta(x). $$

With the common threshold $0.5$:

$$ \hat{y}=1 \quad \text{when} \quad h_\theta(x)\geq 0.5. $$

Because $\sigma(0)=0.5$, this condition is equivalent to:

$$ \theta^\top x\geq 0. $$

For one input feature, the decision point satisfies:

$$ \theta_0+\theta_1x=0. $$

When $\theta_1\neq 0$, the threshold location is:

$$ x=-\frac{\theta_0}{\theta_1}. $$

> [!NOTE]
> **Added clarification: probability and class prediction**
>
> The model output and the final class label are different objects. Logistic regression produces a probability; a chosen threshold converts that probability into a discrete prediction. The source uses $0.5$, but later evaluation settings may use another threshold.


## 5.4 Bernoulli model for a binary target

Page 20 models the binary target with a Bernoulli distribution. For $y\in\lbrace 0,1\rbrace$:

$$ p(y=1\mid x;\theta)=h_\theta(x), $$

$$ p(y=0\mid x;\theta)=1-h_\theta(x). $$

Both cases can be written in one expression:

$$ p(y\mid x;\theta)=h_\theta(x)^y\left(1-h_\theta(x)\right)^{1-y}. $$

To verify the expression:

- when $y=1$, it becomes $h_\theta(x)$;
- when $y=0$, it becomes $1-h_\theta(x)$.

The handwritten annotation on page 20 identifies this as the Bernoulli distribution.

## 5.5 Likelihood of the binary training data

For independent training examples, the likelihood is the product of the conditional probabilities:

$$ L(\theta)=\prod_{i=1}^{n_{\mathrm{train}}}p\left(y^{(i)}\mid x^{(i)};\theta\right). $$

Substituting the Bernoulli form gives:

$$ L(\theta)=\prod_{i=1}^{n_{\mathrm{train}}}h_\theta\left(x^{(i)}\right)^{y^{(i)}}\left(1-h_\theta\left(x^{(i)}\right)\right)^{1-y^{(i)}}. $$

The source notes that multiplying many probabilities can produce an extremely small number. It therefore uses the log-likelihood:

$$ \ell(\theta)=\log L(\theta). $$

Using $\log(ab)=\log a+\log b$ and $\log(a^r)=r\log a$:

$$ \ell(\theta)=\sum_{i=1}^{n_{\mathrm{train}}}\left[y^{(i)}\log h_\theta\left(x^{(i)}\right)+\left(1-y^{(i)}\right)\log\left(1-h_\theta\left(x^{(i)}\right)\right)\right]. $$

Maximum likelihood chooses:

$$ \theta_{\mathrm{MLE}}=\mathrm{arg\,max}_{\theta}\ell(\theta). $$

> [!NOTE]
> **Connection to the usual loss**
>
> This is an added naming clarification. The negative of the displayed log-likelihood is commonly called binary cross-entropy or logistic loss. Maximizing $\ell(\theta)$ is equivalent to minimizing $-\ell(\theta)$.


## 5.6 Derivative of the binary log-likelihood

For one training example, define:

$$ \ell^{(i)}(\theta)=y^{(i)}\log h_\theta\left(x^{(i)}\right)+\left(1-y^{(i)}\right)\log\left(1-h_\theta\left(x^{(i)}\right)\right). $$

The source uses the sigmoid derivative to obtain a particularly simple result. Since:

$$ \nabla_\theta h_\theta(x)=h_\theta(x)\left(1-h_\theta(x)\right)x, $$

applying the chain rule gives:

$$ \nabla_\theta\ell^{(i)}(\theta)=\left(y^{(i)}-h_\theta\left(x^{(i)}\right)\right)x^{(i)}. $$

Summing over all examples:

$$ \nabla_\theta\ell(\theta)=\sum_{i=1}^{n_{\mathrm{train}}}\left(y^{(i)}-h_\theta\left(x^{(i)}\right)\right)x^{(i)}. $$

For the one-feature model with an explicit intercept, page 20 writes the two partial derivatives as:

$$ \frac{\partial\ell}{\partial\theta_0}=y-h_\theta(x), $$

$$ \frac{\partial\ell}{\partial\theta_1}=\left(y-h_\theta(x)\right)x. $$

The parameter update for gradient ascent is:

$$ \theta^{(t+1)}=\theta^{(t)}+\eta\nabla_\theta\ell\left(\theta^{(t)}\right). $$

If the negative log-likelihood is used as the objective, the same training procedure is written as gradient descent with a minus sign.

> [!WARNING]
> **Ascent versus descent**
>
> Page 20 explicitly asks “Gradient descent? Gradient ascent!” because it formulates training as maximizing log-likelihood. Both descriptions are correct only when the sign of the objective is handled consistently.


## 5.7 Multiple input features

For an input vector with several features:

$$ x=\left[1,x_1,x_2,\ldots,x_d\right]^\top. $$

logistic regression uses:

$$ h_\theta(x)=\frac{1}{1+e^{-\theta^\top x}}. $$

The $0.5$ decision boundary is:

$$ \theta^\top x=0. $$

For two measured features and an intercept:

$$ \theta_0+\theta_1x_1+\theta_2x_2=0. $$

This is a line in a two-dimensional feature space. Page 21 illustrates the boundary using age and number of malignant nodes.

The source's boundary is linear because the score is linear in the displayed input features. As in Chapter 1, transformed features could change the shape of the boundary while remaining linear in the parameters, but that extension is not developed on these pages.

## 5.8 One-vs-all multiclass classification

Page 21 first extends binary logistic regression using one-vs-all classification. For $K$ classes, train one binary classifier per class:

$$ h_{\theta_j}(x)\approx p(y=j\mid x), \qquad j=1,2,\ldots,K. $$

Classifier $j$ treats class $j$ as the positive class and all other classes as the negative class. At prediction time, choose the class with the largest score:

$$ \hat{y}=\mathrm{arg\,max}_{j\in\lbrace 1,\ldots,K\rbrace}h_{\theta_j}(x). $$

The source's three-class picture shows three separately learned binary regions whose scores are compared to assign each location to a class.

> [!NOTE]
> **Added clarification: independent binary models**
>
> In one-vs-all classification, the separate sigmoid outputs are not guaranteed to sum to one. The method can still choose the largest score, but the outputs are not automatically a single normalized categorical distribution.


## 5.9 From Bernoulli to categorical outcomes

Binary logistic regression models a Bernoulli target. For $K$ mutually exclusive classes, the source moves to a categorical distribution.

Let:

$$ \phi_j=p(y=j), \qquad j=1,2,\ldots,K, $$

with:

$$ \sum_{j=1}^{K}\phi_j=1. $$

Using indicator notation, the categorical probability mass function can be written as:

$$ p(y;\phi)=\prod_{j=1}^{K}\phi_j^{\mathbf{1}[y=j]}. $$

The source writes the same idea as a product such as $\phi_1^{\mathbf{1}[y=1]}\phi_2^{\mathbf{1}[y=2]}\cdots\phi_K^{\mathbf{1}[y=K]}$.

## 5.10 Softmax regression

For each class $j$, define a linear score:

$$ s_j(x)=\theta_j^\top x. $$

Exponentiation makes every unnormalized class weight positive:

$$ e^{s_j(x)}>0. $$

Softmax normalizes those weights:

$$ p(y=j\mid x;\theta)=\frac{e^{\theta_j^\top x}}{\sum_{r=1}^{K}e^{\theta_r^\top x}}. $$

The probabilities are nonnegative and sum to one:

$$ \sum_{j=1}^{K}p(y=j\mid x;\theta)=1. $$

The predicted class is:

$$ \hat{y}=\mathrm{arg\,max}_{j\in\lbrace 1,\ldots,K\rbrace}p(y=j\mid x;\theta). $$

Because the denominator is shared across classes, this is also:

$$ \hat{y}=\mathrm{arg\,max}_{j\in\lbrace 1,\ldots,K\rbrace}\theta_j^\top x. $$

Page 22 presents the last class as the remaining probability:

$$ p(y=K\mid x;\theta)=1-\sum_{j=1}^{K-1}p(y=j\mid x;\theta). $$

Page 23 then explains why only $K-1$ independent probability parameters are needed: once the first $K-1$ probabilities are known, the final probability is fixed by the sum-to-one constraint.

## 5.11 Logistic regression as the two-class softmax case

Page 22 rewrites the sigmoid as:

$$ h_\theta(x)=\frac{e^{\theta^\top x}}{e^{\theta^\top x}+1}. $$

Its complement is:

$$ 1-h_\theta(x)=\frac{1}{e^{\theta^\top x}+1}. $$

These are softmax probabilities for two class scores, $\theta^\top x$ and $0$:

$$ p(y=1\mid x)=\frac{e^{\theta^\top x}}{e^{\theta^\top x}+e^0}, $$

$$ p(y=0\mid x)=\frac{e^0}{e^{\theta^\top x}+e^0}. $$

Thus logistic regression is the binary special case of softmax regression after choosing one class as the reference score.

## 5.12 Multiclass log-likelihood

For one training example, the softmax probability of its observed class can be written as:

$$ p\left(y^{(i)}\mid x^{(i)};\theta\right)=\prod_{j=1}^{K}p\left(y=j\mid x^{(i)};\theta\right)^{\mathbf{1}[y^{(i)}=j]}. $$

The full log-likelihood is:

$$ \ell(\theta)=\sum_{i=1}^{n_{\mathrm{train}}}\sum_{j=1}^{K}\mathbf{1}[y^{(i)}=j]\log p\left(y=j\mid x^{(i)};\theta\right). $$

Substituting softmax gives:

$$ \ell(\theta)=\sum_{i=1}^{n_{\mathrm{train}}}\sum_{j=1}^{K}\mathbf{1}[y^{(i)}=j]\left[\theta_j^\top x^{(i)}-\log\left(\sum_{r=1}^{K}e^{\theta_r^\top x^{(i)}}\right)\right]. $$

The source page 23 records the log-likelihood structure but does not develop the complete multiclass gradient update.

> [!NOTE]
> **Source boundary**
>
> The negative of this objective is commonly called categorical cross-entropy. That name is an added clarification; the source primarily frames the expression as log-likelihood.


## 5.13 The exponential family

Page 23 places Gaussian, Bernoulli, and categorical distributions into a common form. A distribution belongs to the exponential family when it can be written as:

$$ p(y;\eta)=b(y)\exp\left(\eta^\top T(y)-a(\eta)\right). $$

The source identifies:

- $\eta$ as the natural or canonical parameter;
- $T(y)$ as the sufficient statistic;
- $a(\eta)$ as the log-partition function;
- $b(y)$ as the part that depends only on the observation.

The log-partition function ensures that the distribution is normalized. In the continuous case:

$$ \int p(y;\eta)\,dy=1. $$

In the discrete case:

$$ \sum_y p(y;\eta)=1. $$

The notation $\exp(a(\eta))$ is the partition function, while $a(\eta)$ itself is its logarithm.

## 5.14 Gaussian distribution in exponential-family form

The page 23 derivation uses a Gaussian example with fixed unit variance:

$$ p(y;\mu)=\frac{1}{\sqrt{2\pi}}\exp\left(-\frac{1}{2}(y-\mu)^2\right). $$

Expanding the square:

$$ p(y;\mu)=\frac{1}{\sqrt{2\pi}}\exp\left(-\frac{y^2}{2}\right)\exp\left(\mu y-\frac{\mu^2}{2}\right). $$

Matching terms with the exponential-family form gives:

$$ b(y)=\frac{1}{\sqrt{2\pi}}\exp\left(-\frac{y^2}{2}\right), $$

$$ T(y)=y, $$

$$ \eta=\mu, $$

$$ a(\eta)=\frac{\eta^2}{2}. $$

The source uses this example to connect the Gaussian conditional model with linear regression.

## 5.15 Bernoulli distribution in exponential-family form

For a Bernoulli parameter $\phi$:

$$ p(y;\phi)=\phi^y(1-\phi)^{1-y}. $$

Taking the exponential form:

$$ p(y;\phi)=\exp\left(y\log\phi+(1-y)\log(1-\phi)\right). $$

Rearranging:

$$ p(y;\phi)=\exp\left(y\log\frac{\phi}{1-\phi}+\log(1-\phi)\right). $$

Therefore the natural parameter is:

$$ \eta=\log\frac{\phi}{1-\phi}. $$

This is the log-odds or logit. Solving for $\phi$:

$$ e^\eta=\frac{\phi}{1-\phi}, $$

$$ \phi=\frac{e^\eta}{1+e^\eta}=\frac{1}{1+e^{-\eta}}. $$

Thus the inverse mapping from the Bernoulli natural parameter to its mean is exactly the sigmoid function.

A matching exponential-family identification is:

$$ b(y)=1, \qquad T(y)=y, \qquad a(\eta)=\log\left(1+e^\eta\right). $$

## 5.16 Generalized linear model assumptions

The handwritten summary on page 23 gives three assumptions for a generalized linear model.

1. The conditional target distribution belongs to an exponential family:

$$ y\mid x;\theta\sim\mathrm{ExponentialFamily}(\eta). $$

2. The hypothesis predicts the conditional expectation of the sufficient statistic. For the examples on this page, $T(y)=y$:

$$ h_\theta(x)=\mathbb{E}[T(y)\mid x;\theta]. $$

3. The natural parameter is a linear function of the input:

$$ \eta=\theta^\top x. $$

These assumptions generate different prediction functions depending on the selected response distribution.

### Linear regression from the GLM assumptions

For a Gaussian with fixed variance:

$$ \mathbb{E}[y\mid x;\theta]=\mu. $$

The page identifies $\eta=\mu$, so with $\eta=\theta^\top x$:

$$ h_\theta(x)=\theta^\top x. $$

### Logistic regression from the GLM assumptions

For a Bernoulli target:

$$ \mathbb{E}[y\mid x;\theta]=\phi. $$

The inverse natural-parameter mapping is:

$$ \phi=\frac{1}{1+e^{-\eta}}. $$

Substituting $\eta=\theta^\top x$ gives:

$$ h_\theta(x)=\frac{1}{1+e^{-\theta^\top x}}. $$

The sigmoid is therefore not introduced arbitrarily in the GLM view. It follows from choosing a Bernoulli conditional distribution and linking its natural parameter linearly to the input.

## 5.17 Categorical distribution and the softmax link

Page 23 applies the same reasoning to a categorical target. Since the probabilities sum to one, choose class $K$ as a reference and define $K-1$ natural parameters:

$$ \eta_j=\log\frac{\phi_j}{\phi_K}, \qquad j=1,2,\ldots,K-1. $$

Then:

$$ e^{\eta_j}=\frac{\phi_j}{\phi_K}, $$

so:

$$ \phi_j=e^{\eta_j}\phi_K. $$

Using the normalization condition:

$$ \phi_K+\sum_{j=1}^{K-1}\phi_j=1, $$

we obtain:

$$ \phi_K\left(1+\sum_{j=1}^{K-1}e^{\eta_j}\right)=1. $$

Therefore:

$$ \phi_K=\frac{1}{1+\sum_{j=1}^{K-1}e^{\eta_j}}. $$

For the other classes:

$$ \phi_j=\frac{e^{\eta_j}}{1+\sum_{r=1}^{K-1}e^{\eta_r}}. $$

The source expresses this symmetrically by defining a reference natural parameter $\eta_K=0$:

$$ \phi_j=\frac{e^{\eta_j}}{\sum_{r=1}^{K}e^{\eta_r}}. $$

Finally, with $\eta_j=\theta_j^\top x$:

$$ p(y=j\mid x;\theta)=\frac{e^{\theta_j^\top x}}{\sum_{r=1}^{K}e^{\theta_r^\top x}}. $$

This is softmax regression derived as a generalized linear model for categorical outcomes.

## 5.18 One-vs-all and softmax compared

| Property | One-vs-all logistic regression | Softmax regression |
|---|---|---|
| Number of fitted binary tasks | one per class | one joint multiclass model |
| Output form | separate sigmoid scores | normalized categorical probabilities |
| Sum of outputs | not necessarily one | exactly one |
| Prediction | largest classifier score | largest class probability |
| Source location | page 21 | pages 22–23 |

The two approaches can produce similar class decisions, but they make different probability-modeling assumptions.

## 5.19 Common mistakes

1. **Using the linear score itself as a probability.** The score $\theta^\top x$ is unrestricted; the sigmoid maps it into $(0,1)$.
2. **Confusing the probability with the thresholded class.** Logistic regression outputs $p(y=1\mid x)$ before a decision rule is applied.
3. **Using squared-error reasoning for the Bernoulli model without noticing the likelihood change.** The source derives logistic training from Bernoulli log-likelihood.
4. **Performing gradient descent on the log-likelihood without changing its sign.** The source maximizes log-likelihood using gradient ascent.
5. **Assuming one-vs-all probabilities must sum to one.** The separately trained sigmoid outputs are not jointly normalized.
6. **Forgetting the softmax denominator.** Exponentiated scores are unnormalized weights until divided by their sum.
7. **Treating all $K$ categorical probabilities as independent.** The sum-to-one constraint leaves only $K-1$ independent probability parameters.
8. **Confusing the partition function with the log-partition function.** The partition function is $e^{a(\eta)}$; $a(\eta)$ is its logarithm.
9. **Memorizing sigmoid and softmax without the GLM connection.** Page 23 derives them from Bernoulli and categorical exponential-family models.
10. **Assuming page 23 gives a complete optimization algorithm for softmax.** It derives the model and likelihood structure but does not finish the full gradient procedure.

## 5.20 Chapter summary

- Linear regression is not naturally suited to binary probabilities because its output is unrestricted.
- Logistic regression applies the sigmoid to a linear score.
- The sigmoid output is interpreted as $p(y=1\mid x;\theta)$.
- Binary targets are modeled with a Bernoulli distribution.
- Maximum likelihood leads to the binary log-likelihood and its simple residual-like gradient.
- With multiple input features, the $0.5$ decision boundary is $\theta^\top x=0$.
- One-vs-all trains separate binary classifiers and chooses the largest output.
- Softmax jointly normalizes class scores into a categorical probability distribution.
- Logistic regression is the two-class softmax model with one reference score fixed to zero.
- Gaussian, Bernoulli, and categorical distributions can be written in exponential-family form.
- Generalized linear models combine an exponential-family response, an expected sufficient statistic, and a linear natural parameter.
- The Bernoulli GLM produces the sigmoid, and the categorical GLM produces softmax.

## 5.21 Self-check questions

1. Why is an unrestricted linear-regression output unsuitable as a binary probability?
2. What is the derivative of the sigmoid function?
3. How does the Bernoulli probability expression represent both $y=0$ and $y=1$?
4. Why does the source optimize log-likelihood instead of the raw likelihood product?
5. What is the gradient of one-example logistic log-likelihood?
6. Why is the $0.5$ decision boundary defined by $\theta^\top x=0$?
7. How does one-vs-all multiclass classification work?
8. Why do softmax probabilities sum to one?
9. In what sense is logistic regression a two-class softmax model?
10. What are $\eta$, $T(y)$, $a(\eta)$, and $b(y)$ in the exponential-family form?
11. How does the Bernoulli natural parameter lead to the sigmoid?
12. Why are only $K-1$ categorical probability parameters independent?
13. Which three assumptions on page 23 define the generalized linear model construction?
