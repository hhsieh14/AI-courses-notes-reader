# 7. Generative Classification

**Source pages:** 26–30  
**Status:** reconstructed and equation-checked

This chapter introduces generative classification. Instead of modeling the posterior class probability directly, a generative classifier models how each class produces the input and combines that class-conditional model with the class prior through Bayes' rule. The source develops two examples: Gaussian discriminant analysis and Naive Bayes.

## 7.1 Discriminative and generative learning

The discriminative models developed earlier learn the conditional distribution directly:

$$ p(y\mid x). $$

For example, logistic regression parameterizes $p(y=1\mid x;\theta)$ and uses it to classify an input.

A generative learning algorithm instead models:

$$ p(x\mid y) $$

and the class prior:

$$ p(y). $$

Bayes' rule then gives the posterior distribution:

$$ p(y\mid x)=\frac{p(x\mid y)p(y)}{p(x)}. $$

For binary classification, the evidence can be expanded by the law of total probability:

$$ p(x)=p(x\mid y=0)p(y=0)+p(x\mid y=1)p(y=1). $$

The denominator is the same for every candidate class. Therefore classification can compare the unnormalized class scores:

$$ p(y\mid x)\propto p(x\mid y)p(y). $$

![Redrawn generative-classification overview](assets/diagrams/07_generative_classification.svg)

*Redrawn from source pages 26–30: discriminative learning models the posterior directly, while generative learning combines a class prior and class-conditional input model. Shared Gaussian covariance gives a linear boundary; class-specific covariances can give a quadratic boundary.*

> [!TIP]
> **Review intuition**
>
> A discriminative model asks, “Which label is likely for this input?” A generative model first asks, “How likely is this input under each class?” and then converts those class scores into posterior probabilities.


## 7.2 Why generative models require distributional assumptions

The handwritten note on page 26 emphasizes that class frequencies alone are not enough. We may estimate $p(y)$ from label counts, but we still need functions that assign probabilities to possible inputs through $p(x\mid y)$.

Gaussian discriminant analysis is therefore a **parametric** generative model. It makes two main assumptions:

1. the class label follows a Bernoulli distribution;
2. the input conditioned on the class follows a multivariate Gaussian distribution.

These assumptions specify the form of both $p(y)$ and $p(x\mid y)$.

## 7.3 Multivariate Gaussian model

For a $d$-dimensional input, a multivariate Gaussian distribution is parameterized by a mean vector and covariance matrix:

$$ x\sim\mathcal{N}(\mu,\Sigma). $$

The mean is:

$$ \mu=\mathbb{E}[x], $$

and the covariance matrix is:

$$ \Sigma=\mathbb{E}\left[(x-\mu)(x-\mu)^\top\right]. $$

The covariance matrix is symmetric. Its diagonal entries are feature variances, while its off-diagonal entries describe covariance between pairs of features. The handwritten annotation notes that when two features tend to increase together, their covariance is positive.

The multivariate Gaussian density is:

$$ p(x;\mu,\Sigma)=\frac{1}{(2\pi)^{d/2}\det(\Sigma)^{1/2}}\exp\left(-\frac{1}{2}(x-\mu)^\top\Sigma^{-1}(x-\mu)\right). $$

## 7.4 Gaussian discriminant analysis

For binary Gaussian discriminant analysis, or **GDA**, the source assumes:

$$ y\sim\mathrm{Bernoulli}(\pi). $$

Thus:

$$ p(y)=\pi^y(1-\pi)^{1-y}. $$

The class-conditional input distributions are:

$$ x\mid y=0\sim\mathcal{N}(\mu_0,\Sigma), $$

$$ x\mid y=1\sim\mathcal{N}(\mu_1,\Sigma). $$

Equivalently:

$$ p(x\mid y=0)=\frac{1}{(2\pi)^{d/2}\det(\Sigma)^{1/2}}\exp\left(-\frac{1}{2}(x-\mu_0)^\top\Sigma^{-1}(x-\mu_0)\right), $$

$$ p(x\mid y=1)=\frac{1}{(2\pi)^{d/2}\det(\Sigma)^{1/2}}\exp\left(-\frac{1}{2}(x-\mu_1)^\top\Sigma^{-1}(x-\mu_1)\right). $$

The two classes have different means but share the same covariance matrix in the first GDA model presented by the source.

## 7.5 GDA likelihood

For a training set of independent examples, the joint likelihood is:

$$ L(\pi,\mu_0,\mu_1,\Sigma)=\prod_{i=1}^{n_{\mathrm{train}}}p\left(x^{(i)},y^{(i)};\pi,\mu_0,\mu_1,\Sigma\right). $$

Using the product rule:

$$ p(x,y)=p(x\mid y)p(y), $$

so the log-likelihood becomes:

$$ \ell(\pi,\mu_0,\mu_1,\Sigma)=\sum_{i=1}^{n_{\mathrm{train}}}\log p\left(x^{(i)}\mid y^{(i)};\mu_0,\mu_1,\Sigma\right)+\sum_{i=1}^{n_{\mathrm{train}}}\log p\left(y^{(i)};\pi\right). $$

The maximum-likelihood estimates shown on page 27 have direct counting and averaging interpretations.

### Class-prior estimate

$$ \hat{\pi}=\frac{1}{n_{\mathrm{train}}}\sum_{i=1}^{n_{\mathrm{train}}}\mathbb{1}\left[y^{(i)}=1\right]. $$

This is the fraction of training examples in class $1$.

### Class-mean estimates

$$ \hat{\mu}_0=\frac{\sum_{i=1}^{n_{\mathrm{train}}}\mathbb{1}\left[y^{(i)}=0\right]x^{(i)}}{\sum_{i=1}^{n_{\mathrm{train}}}\mathbb{1}\left[y^{(i)}=0\right]}, $$

$$ \hat{\mu}_1=\frac{\sum_{i=1}^{n_{\mathrm{train}}}\mathbb{1}\left[y^{(i)}=1\right]x^{(i)}}{\sum_{i=1}^{n_{\mathrm{train}}}\mathbb{1}\left[y^{(i)}=1\right]}. $$

Each estimate averages the inputs belonging to one class.

### Shared-covariance estimate

$$ \hat{\Sigma}=\frac{1}{n_{\mathrm{train}}}\sum_{i=1}^{n_{\mathrm{train}}}\left(x^{(i)}-\hat{\mu}_{y^{(i)}}\right)\left(x^{(i)}-\hat{\mu}_{y^{(i)}}\right)^\top. $$

Every example is centered using the mean of its own class, and the centered outer products are pooled into one shared covariance estimate.

## 7.6 Linear discriminant analysis

With a shared covariance matrix, GDA produces a linear decision boundary. The classifier compares the two posterior class probabilities, or equivalently the log-posterior odds:

$$ \log\frac{p(y=1\mid x)}{p(y=0\mid x)}=\log\frac{p(x\mid y=1)p(y=1)}{p(x\mid y=0)p(y=0)}. $$

Substituting the Gaussian class-conditionals gives:

$$ \log\frac{p(y=1\mid x)}{p(y=0\mid x)}=\log\frac{\pi}{1-\pi}-\frac{1}{2}(x-\mu_1)^\top\Sigma^{-1}(x-\mu_1)+\frac{1}{2}(x-\mu_0)^\top\Sigma^{-1}(x-\mu_0). $$

> [!NOTE]
> **Added clarification: why the boundary is linear**
>
> Expanding both quadratic forms produces the same $x^\top\Sigma^{-1}x$ term. Because the covariance is shared, those quadratic terms cancel. The remaining expression is affine in $x$:
>
> $$ \log\frac{p(y=1\mid x)}{p(y=0\mid x)}=\theta_0+\theta^\top x. $$


The equality point between the two classes therefore satisfies:

$$ \theta_0+\theta^\top x=0, $$

which is a linear decision boundary. This shared-covariance version is called **linear discriminant analysis**, or **LDA**, on page 27.

## 7.7 Quadratic discriminant analysis

The handwritten note on page 27 considers separate covariance matrices:

$$ x\mid y=0\sim\mathcal{N}(\mu_0,\Sigma_0), $$

$$ x\mid y=1\sim\mathcal{N}(\mu_1,\Sigma_1). $$

When $\Sigma_0\neq\Sigma_1$, the quadratic terms do not generally cancel. The decision equation can contain terms such as:

$$ x^\top\Sigma_0^{-1}x-x^\top\Sigma_1^{-1}x. $$

The resulting boundary can therefore be quadratic. The source calls this model **quadratic discriminant analysis**, or **QDA**.

| Model | Covariance assumption | Boundary described in the source |
|---|---|---|
| LDA | one shared $\Sigma$ | linear |
| QDA | class-specific $\Sigma_0,\Sigma_1$ | potentially quadratic |

## 7.8 Relationship between GDA and logistic regression

The handwritten note on page 30 observes that the posterior produced by shared-covariance GDA has logistic form:

$$ p(y=1\mid x;\pi,\mu_0,\mu_1,\Sigma)=\frac{1}{1+\exp\left(-\theta_0-\theta^\top x\right)}, $$

where $\theta_0$ and $\theta$ are functions of the GDA parameters.

Thus GDA implies a logistic posterior when its Gaussian assumptions hold.

The reverse implication does not follow. A logistic-regression model specifies $p(y\mid x)$ directly, but it does not require either $p(x\mid y=0)$ or $p(x\mid y=1)$ to be Gaussian.

> [!NOTE]
> **Source distinction**
>
> The handwritten note summarizes this as “GDA implies logistic regression,” while logistic regression does not necessarily imply a multivariate-Gaussian class-conditional model.


## 7.9 Naive Bayes

The second generative classifier in the source is **Naive Bayes**. For a class $y$ and feature vector:

$$ x=(x_1,x_2,\ldots,x_n), $$

Bayes' rule gives:

$$ p(y\mid x)=\frac{p(x\mid y)p(y)}{p(x)}. $$

For classification, the common denominator can be omitted:

$$ p(y\mid x)\propto p(x\mid y)p(y). $$

The difficulty is modeling the joint conditional distribution:

$$ p(x_1,x_2,\ldots,x_n\mid y). $$

Using the chain rule, it can be expanded as:

$$ p(x_1,\ldots,x_n\mid y)=p(x_1\mid x_2,\ldots,x_n,y)p(x_2\mid x_3,\ldots,x_n,y)\cdots p(x_n\mid y). $$

This exact factorization can require many complicated conditional distributions.

## 7.10 The naive conditional-independence assumption

Naive Bayes simplifies the model by assuming that the features are conditionally independent once the class is known:

$$ p(x_i\mid x_1,\ldots,x_{i-1},x_{i+1},\ldots,x_n,y)=p(x_i\mid y). $$

Therefore:

$$ p(x\mid y)=\prod_{i=1}^{n}p(x_i\mid y), $$

and the posterior score becomes:

$$ p(y\mid x)\propto p(y)\prod_{i=1}^{n}p(x_i\mid y). $$

For $K$ possible classes, the maximum-a-posteriori prediction rule is:

$$ \hat{y}=\underset{k\in\lbrace 1,\ldots,K\rbrace}{\mathrm{arg\,max}}\;p(y=k)\prod_{i=1}^{n}p(x_i\mid y=k). $$

> [!WARNING]
> **Meaning of the naive assumption**
>
> The features are not assumed to be independent in the dataset overall. The assumption is conditional independence **given the class label**.


## 7.11 Naive Bayes example: Play Tennis

Page 29 uses the following 14-example training set.

| Day | Outlook | Temperature | Humidity | Wind | Play tennis |
|---|---|---|---|---|---|
| D1 | Sunny | Hot | High | Weak | No |
| D2 | Sunny | Hot | High | Strong | No |
| D3 | Overcast | Hot | High | Weak | Yes |
| D4 | Rain | Mild | High | Weak | Yes |
| D5 | Rain | Cool | Normal | Weak | Yes |
| D6 | Rain | Cool | Normal | Strong | No |
| D7 | Overcast | Cool | Normal | Strong | Yes |
| D8 | Sunny | Mild | High | Weak | No |
| D9 | Sunny | Cool | Normal | Weak | Yes |
| D10 | Rain | Mild | Normal | Weak | Yes |
| D11 | Sunny | Mild | Normal | Strong | Yes |
| D12 | Overcast | Mild | High | Strong | Yes |
| D13 | Overcast | Hot | Normal | Weak | Yes |
| D14 | Rain | Mild | High | Strong | No |

The class priors are:

$$ p(\mathrm{Yes})=\frac{9}{14}, \qquad p(\mathrm{No})=\frac{5}{14}. $$

The source constructs frequency lookup tables.

### Outlook

| Value | $p(\text{value}\mid\mathrm{Yes})$ | $p(\text{value}\mid\mathrm{No})$ |
|---|---:|---:|
| Sunny | $2/9$ | $3/5$ |
| Overcast | $4/9$ | $0/5$ |
| Rain | $3/9$ | $2/5$ |

### Temperature

| Value | $p(\text{value}\mid\mathrm{Yes})$ | $p(\text{value}\mid\mathrm{No})$ |
|---|---:|---:|
| Hot | $2/9$ | $2/5$ |
| Mild | $4/9$ | $2/5$ |
| Cool | $3/9$ | $1/5$ |

### Humidity and wind

| Feature value | $p(\text{value}\mid\mathrm{Yes})$ | $p(\text{value}\mid\mathrm{No})$ |
|---|---:|---:|
| Humidity = High | $3/9$ | $4/5$ |
| Humidity = Normal | $6/9$ | $1/5$ |
| Wind = Strong | $3/9$ | $3/5$ |
| Wind = Weak | $6/9$ | $2/5$ |

The query example is:

- Outlook = Sunny;
- Temperature = Cool;
- Humidity = High;
- Wind = Strong.

The unnormalized score for **Yes** is:

$$ s_{\mathrm{Yes}}=\frac{9}{14}\cdot\frac{2}{9}\cdot\frac{3}{9}\cdot\frac{3}{9}\cdot\frac{3}{9}\approx 0.0053. $$

The unnormalized score for **No** is:

$$ s_{\mathrm{No}}=\frac{5}{14}\cdot\frac{3}{5}\cdot\frac{1}{5}\cdot\frac{4}{5}\cdot\frac{3}{5}\approx 0.0206. $$

Because:

$$ s_{\mathrm{No}}>s_{\mathrm{Yes}}, $$

the model predicts **No**.

> [!NOTE]
> **Source scope: unsmoothed frequencies**
>
> The source uses the raw lookup-table frequencies, including $0/5$ for Overcast under the No class. It does not introduce a smoothing method in these pages.


## 7.12 Naive Bayes example: spam filtering

Page 30 represents an email with a binary vocabulary vector:

$$ x\in\lbrace 0,1\rbrace^{50000}. $$

Each coordinate indicates whether one vocabulary word is present. For example, $x_j=1$ means the $j$th vocabulary word occurs in the email.

Let $y=1$ denote spam and $y=0$ denote not spam. The prior estimate is:

$$ \hat{p}(\mathrm{spam})=\frac{\sum_{i=1}^{n}\mathbb{1}\left[y^{(i)}=1\right]}{n}, $$

$$ \hat{p}(\mathrm{not\ spam})=1-\hat{p}(\mathrm{spam}). $$

For each vocabulary coordinate $j$, the source estimates:

$$ \hat{p}(x_j=1\mid y=1)=\frac{\sum_{i=1}^{n}\mathbb{1}\left[x_j^{(i)}=1\ \text{and}\ y^{(i)}=1\right]}{\sum_{i=1}^{n}\mathbb{1}\left[y^{(i)}=1\right]}, $$

$$ \hat{p}(x_j=1\mid y=0)=\frac{\sum_{i=1}^{n}\mathbb{1}\left[x_j^{(i)}=1\ \text{and}\ y^{(i)}=0\right]}{\sum_{i=1}^{n}\mathbb{1}\left[y^{(i)}=0\right]}. $$

The two class scores are proportional to:

$$ p(y=1\mid x)\propto p(y=1)\prod_{j=1}^{50000}p(x_j\mid y=1), $$

$$ p(y=0\mid x)\propto p(y=0)\prod_{j=1}^{50000}p(x_j\mid y=0). $$

The class with the larger product score is selected.

> [!NOTE]
> **Technical note: binary feature likelihood**
>
> Because the page uses a binary word vector, a complete Bernoulli feature factor uses the estimated presence probability when $x_j=1$ and its complement when $x_j=0$. The source itself focuses on estimating the presence probabilities and comparing the two class products.


## 7.13 Assumptions and inductive bias

The final handwritten sketch on page 30 contrasts a highly flexible polynomial model with a structured sinusoidal model.

A high-degree polynomial can be written as:

$$ h_\theta(z)=\theta_0+\theta_1z+\theta_2z^2+\cdots+\theta_{100}z^{100}. $$

With weak structural assumptions, many functions may fit a limited training sample.

The alternative sketch assumes a sinusoidal form such as:

$$ h_\theta(t)=\theta_1\sin(\theta_2t+\theta_3). $$

This assumption restricts the set of candidate functions. The source associates such structure with:

- **inductive bias**;
- model structure;
- domain knowledge;
- interpretability;
- easier training.

> [!NOTE]
> **Interpretation of the handwritten comparison**
>
> A structural assumption can be useful when it matches the phenomenon being modeled. It can also introduce bias when the assumed family does not match the data. This connects the generative-model assumptions in this chapter to the earlier bias-variance discussion.


## 7.14 Common mistakes

1. **Confusing $p(y\mid x)$ and $p(x\mid y)$.** The former predicts a class from an input; the latter models inputs generated within a class.
2. **Forgetting the class prior.** Generative classification combines $p(x\mid y)$ with $p(y)$.
3. **Treating $p(x)$ as class-specific.** The evidence is common to all candidate classes and can be omitted only when comparing proportional class scores.
4. **Using frequencies without specifying a probability model for continuous inputs.** GDA supplies a Gaussian form for $p(x\mid y)$.
5. **Assuming LDA and QDA have the same covariance structure.** LDA shares one covariance; QDA allows class-specific covariances.
6. **Assuming a shared-covariance Gaussian boundary remains quadratic.** The common quadratic term cancels, leaving a linear boundary.
7. **Assuming logistic regression requires Gaussian features.** The source states only that GDA's posterior has logistic form; logistic regression does not impose the reverse generative assumptions.
8. **Interpreting Naive Bayes as unconditional feature independence.** The assumption is conditional on the class.
9. **Multiplying $p(y\mid x_i)$ terms instead of $p(x_i\mid y)$ terms.** Naive Bayes factorizes the class-conditional input likelihood.
10. **Ignoring zero frequencies in the Play Tennis table.** The source uses unsmoothed estimates, so an unseen class-feature value can make a product zero.
11. **Treating the unnormalized Naive Bayes scores as probabilities that must sum to one.** They are sufficient for the source's argmax comparison but require normalization to become posterior probabilities.
12. **Assuming a flexible hypothesis class contains no assumptions.** Choosing a polynomial degree or any model family already introduces an inductive bias.

## 7.15 Chapter summary

- Discriminative learning models $p(y\mid x)$ directly.
- Generative learning models $p(x\mid y)$ and $p(y)$, then applies Bayes' rule.
- GDA uses a Bernoulli class prior and Gaussian class-conditional input distributions.
- Maximum likelihood estimates the class prior by counting class-one examples, class means by within-class averages, and covariance by pooled centered outer products.
- Shared covariance gives LDA and a linear decision boundary.
- Class-specific covariances give QDA and can produce a quadratic boundary.
- Shared-covariance GDA implies a logistic posterior, but logistic regression does not require Gaussian class-conditionals.
- Naive Bayes assumes features are conditionally independent given the class.
- Its MAP classifier compares a prior multiplied by class-conditional feature probabilities.
- The Play Tennis example predicts No because its unnormalized class score is larger.
- The spam example represents messages with a $50000$-dimensional binary word vector and estimates class-conditional word-presence probabilities.
- The final handwritten note connects model assumptions to inductive bias, structure, domain knowledge, interpretability, and trainability.

## 7.16 Self-check questions

1. What distribution does a discriminative classifier model directly?
2. What two distributions does a generative classifier model before applying Bayes' rule?
3. Why can $p(x)$ be omitted when comparing candidate classes?
4. What are the two main distributional assumptions in binary GDA?
5. What do $\pi$, $\mu_0$, $\mu_1$, and $\Sigma$ represent?
6. How is the maximum-likelihood estimate of $\pi$ computed?
7. Why is the GDA covariance estimate centered around the mean associated with each example's class?
8. Why does shared covariance produce a linear discriminant boundary?
9. Why can class-specific covariance matrices produce a quadratic boundary?
10. In what direction does the source connect GDA and logistic regression?
11. What exact conditional-independence assumption does Naive Bayes make?
12. What is the Naive Bayes MAP decision rule?
13. How are the Play Tennis class priors obtained?
14. Why does the Play Tennis query receive the No prediction?
15. What does one coordinate in the spam-filtering vector represent?
16. How are class-conditional word-presence probabilities estimated?
17. What is the role of an inductive bias in the final handwritten sketch?
