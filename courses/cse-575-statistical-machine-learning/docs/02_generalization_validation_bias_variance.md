# 2. Generalization, Validation, Bias, and Variance

A model that fits its training data is easy to build. The real question is how well it predicts data it has never seen. This chapter covers the tools for answering that question (train/validation/test splits and cross-validation) and the bias–variance decomposition that explains *why* flexible models can predict worse.

## 2.1 Model complexity and generalization

Take the polynomial model from Chapter 1,

$$ h_\theta(x) = \theta_0 + \theta_1x + \theta_2x^2 + \cdots + \theta_dx^d, $$

and fit it with degree 1, a moderate degree, and degree 15 to the same noisy sample from a curved function:

| Degree | Training fit | New data |
|---|---|---|
| 1 | poor, misses the curve | poor |
| moderate | good | usually best |
| 15 | almost perfect, oscillates through every point | poor |

Three terms describe what we are doing here:

- **Generalization:** how well a model predicts unseen data.
- **Model selection:** comparing candidate models (degrees, $\lambda$ values, architectures) to pick one.
- **Model assessment:** after picking, estimating the chosen model's error on new data.

> [!TIP]
> **Intuition**
>
> Training error answers "how well did the model memorize what it saw?" Generalization error answers "how well does the rule it learned work on what it didn't see?" Only the second one matters in deployment.

## 2.2 Underfitting and overfitting

| Regime | Training error | Validation/test error | Diagnosis |
|---|---:|---:|---|
| Underfitting | high | high | model too restricted to capture the structure |
| Good fit | low | low | captures the signal, ignores the noise |
| Overfitting | very low | high | model follows noise specific to this sample |

## 2.3 Training, validation and test sets

With a single train/test split you fit on the training data and measure error on the held-out test data. That is enough to *assess* one model, but not to *choose* among many and then report an honest number. For that we need three sets:

- **Training set:** fit the parameters.
- **Validation set:** choose between models or hyperparameters.
- **Test set:** touched once, at the end, to estimate the error of the chosen model.

Typical splits are 50/25/25 or 70/20/10, depending on how much data there is.

> [!WARNING]
> **Why the test set must stay untouched**
>
> If you look at test error to pick a model, the test set has become part of model selection, and the number you report is optimistically biased. With enough comparisons you can overfit the test set just as you overfit a training set.

## 2.4 K-fold cross-validation

When data is scarce, holding out a fixed validation set wastes examples. K-fold cross-validation splits the model-selection data into $K$ folds, trains $K$ times (each time holding out one fold) and averages the held-out errors:

$$ J_{\mathrm{CV}}(m) = \frac{1}{K}\sum_{j=1}^{K} J_{\text{holdout}}^{(j)}(m), $$

where $m$ is the candidate setting (for example a polynomial degree). Pick the $m$ with the smallest $J_{\mathrm{CV}}(m)$, retrain on all the model-selection data, and only then evaluate on the test set.

```text
fold 1:  [hold] [train] [train] [train]
fold 2:  [train] [hold] [train] [train]
fold 3:  [train] [train] [hold] [train]
fold 4:  [train] [train] [train] [hold]
                          → average the four held-out errors
```

The rotating held-out fold plays the **validation** role, even though it is often called a "test fold". A separate test set is still needed for the final number.

> [!NOTE]
> **Beyond the lecture: when shuffled K-fold is wrong**
>
> K-fold assumes examples are exchangeable. For time series, or data with groups (several rows per patient or per user), shuffled folds leak information from the future or from the same group into training. Use forward-chaining splits for time (see [Temporal Learning, Chapter 8](../../modern-temporal-learning/notes/08_rnn_lstm_gru_and_seq2seq.md)) and group K-fold for clustered data.

## 2.5 Complexity versus error

As model complexity grows:

- training error $J_{\mathrm{train}}$ keeps going down;
- cross-validation error $J_{\mathrm{CV}}$ first goes down (less underfitting), then up (more overfitting).

The goal is the complexity at the bottom of the $J_{\mathrm{CV}}$ curve, not the one with the lowest training error.

> [!NOTE]
> **Beyond the lecture: double descent**
>
> The U-shaped curve is not the whole story for heavily over-parameterized models. As the number of parameters passes the point where the model can exactly interpolate the training set, test error can peak and then *decrease again*. This "double descent" (Belkin et al., 2019) is one reason large neural networks generalize despite having far more parameters than examples. It does not contradict bias–variance; it shows that parameter count is a poor measure of effective complexity.

## 2.6 The learned model is random

The training set is a random sample, $D\sim P^n$, and the learning algorithm $\mathcal{A}$ maps it to a predictor $h_D=\mathcal{A}(D)$. So $h_D$ is itself random: draw a different training set and you get different parameters $\hat\theta_1,\hat\theta_2,\ldots$ and different fitted curves. A low-degree polynomial fitted to many resampled datasets gives nearly the same line every time; a high-degree one gives wildly different curves. That spread is what "variance" means below. It is a property of the model class *plus* the learning algorithm, not of any single fitted curve.

## 2.7 Bias and variance

For an estimator $\hat\theta$ of a true parameter $\theta$:

$$ \mathrm{Bias}(\hat{\theta}) = \mathbb{E}_D[\hat{\theta}] - \theta, \qquad \mathrm{Var}(\hat{\theta}) = \mathbb{E}_D\left[\left(\hat{\theta}-\mathbb{E}_D[\hat{\theta}]\right)^2\right]. $$

For predictions we use the same ideas pointwise. Let $\bar{h}(x) = \mathbb{E}_D[h_D(x)]$ be the average prediction over training sets and $h^\star(x)$ the true regression function. Then

$$ \mathrm{Bias}_h(x) = \bar{h}(x) - h^{\star}(x), \qquad \mathrm{Var}_h(x) = \mathbb{E}_D\left[\left(h_D(x)-\bar{h}(x)\right)^2\right]. $$

Flexible models make fewer assumptions and have lower bias, but depend more on the particular sample, which means higher variance.

## 2.8 Setting up the decomposition

Data come from a joint distribution $p(x,y)=p(y\mid x)p(x)$. At a fixed $x$ the best possible squared-error prediction is the conditional mean

$$ \bar{y}(x) = \mathbb{E}_{y\mid x}[y] = \int y\,p(y\mid x)\,dy, $$

and we write $y = h^\star(x) + \epsilon$ with $h^\star(x)=\bar y(x)$ and $\mathbb{E}[\epsilon\mid x]=0$.

## 2.9 Error of a model versus error of an algorithm

The expected test error of one fitted model is

$$ R(h_D) = \mathbb{E}_{(x,y)\sim P}\left[\left(y-h_D(x)\right)^2\right] = \int_x\int_y \left(y-h_D(x)\right)^2p(x,y)\,dy\,dx. $$

To judge the *algorithm*, average over training sets too:

$$ R(\mathcal{A}) = \mathbb{E}_{D\sim P^n}\,\mathbb{E}_{(x,y)\sim P}\left[\left(y-h_D(x)\right)^2\right]. $$

Conceptually: sample a training set, train, sample a fresh test point, measure the squared error, and repeat forever.

## 2.10 Decomposing expected test error

**Step 1: separate noise from model error.** Add and subtract $h^\star(x)$:

$$ y-h_D(x) = \left(y-h^{\star}(x)\right) + \left(h^{\star}(x)-h_D(x)\right). $$

Square and take expectations. The cross term vanishes because $\mathbb{E}[y-h^\star(x)\mid x]=0$ and the test point is independent of $D$:

$$ \mathbb{E}_{D,(x,y)}\left[\left(y-h_D(x)\right)^2\right] = \underbrace{\mathbb{E}_{x,y}\left[\left(y-h^{\star}(x)\right)^2\right]}_{\text{noise}} + \mathbb{E}_{D,x}\left[\left(h^{\star}(x)-h_D(x)\right)^2\right]. $$

**Step 2: split model error around the average predictor.** Add and subtract $\bar h(x)$:

$$ h^{\star}(x)-h_D(x) = \left(h^{\star}(x)-\bar{h}(x)\right) + \left(\bar{h}(x)-h_D(x)\right). $$

The cross term vanishes because $\mathbb{E}_D[\bar h(x)-h_D(x)]=0$, giving

$$ \mathbb{E}_{D,x}\left[\left(h^{\star}(x)-h_D(x)\right)^2\right] = \underbrace{\mathbb{E}_x\left[\left(h^{\star}(x)-\bar{h}(x)\right)^2\right]}_{\text{bias}^2} + \underbrace{\mathbb{E}_{D,x}\left[\left(h_D(x)-\bar{h}(x)\right)^2\right]}_{\text{variance}}. $$

**Result:**

$$ \text{expected test error} = \text{noise} + \text{bias}^2 + \text{variance}. $$

The noise term cannot be reduced by any model; it comes from the data-generating process. Bias and variance are what our modeling choices trade against each other. If we measure error against $h^\star(x)$ instead of the noisy $y$, the noise term drops out and error $=$ bias$^2$ + variance.

## 2.11 Why the cross terms disappear

Both cancellations are the same fact: a quantity minus its own mean has mean zero.

$$ \mathbb{E}_D\left[h_D(x)-\bar{h}(x)\right] = 0, \qquad \mathbb{E}_{y\mid x}\left[y-\bar{y}(x)\right] = 0. $$

## 2.12 Using the decomposition

| Term | Meaning | Symptom | Typical fix |
|---|---|---|---|
| Noise | irreducible randomness in $y$ | floor under both errors | better features or measurements |
| Bias² | average prediction is systematically off | training error high | richer model, more features, less regularization |
| Variance | prediction changes a lot across samples | big gap between training and validation error | more data, regularization, simpler model, averaging/bagging |

Bias can be positive or negative; squaring it makes both directions cost. The practical value of the decomposition is diagnostic: a high-bias problem and a high-variance problem need opposite remedies.

```python
# Bias–variance by simulation: fit many datasets, look at the spread at one x.
import numpy as np
rng = np.random.default_rng(1)
f = lambda x: np.sin(2 * np.pi * x)
x0 = 0.3
for degree in (1, 3, 9):
    preds = []
    for _ in range(500):
        x = rng.uniform(0, 1, 20)
        y = f(x) + rng.normal(0, 0.3, 20)
        preds.append(np.polyval(np.polyfit(x, y, degree), x0))
    preds = np.array(preds)
    print(degree, "bias^2=%.3f  var=%.3f" % ((preds.mean() - f(x0))**2, preds.var()))

# 1 bias^2=0.317  var=0.026   <- underfits: bias dominates
# 3 bias^2=0.002  var=0.016   <- about right
# 9 bias^2=0.018  var=2.200   <- overfits: variance explodes
```

## 2.13 Common mistakes

1. **Choosing the model on the test set.** That mixes selection with assessment.
2. **Picking the lowest training error.** Training error keeps falling long after generalization gets worse.
3. **Treating the fitted model as fixed.** It depends on the random training sample.
4. **Confusing noise with variance.** Noise lives in $p(y\mid x)$; variance lives in how $h_D$ changes across training sets.
5. **Forgetting the square on bias.** The decomposition contains bias², not bias.
6. **Assuming test error is always U-shaped in parameter count.** See double descent.
7. **Reporting the CV score as the final test error.** CV folds did model selection; a separate test set gives the unbiased estimate.

## 2.14 Summary

- Generalization is performance on unseen data.
- Model selection and model assessment need separate data.
- K-fold CV averages held-out error across rotating folds.
- Underfitting ↔ high bias; overfitting ↔ high variance.
- Expected squared test error = noise + bias² + variance.

## 2.15 Self-check

1. What is the difference between model selection and model assessment?
2. How does 4-fold cross-validation work?
3. Why can training error decrease while test error increases?
4. Why is $h_D$ a random function?
5. Which expectation defines prediction variance?
6. Why do the cross terms vanish in the derivation?
7. Which part of test error is irreducible?

<details>
<summary>Answers</summary>

1. Selection chooses among candidates using validation data; assessment estimates the chosen model's error on untouched test data.
2. Split into 4 folds; train on 3, measure error on the held-out one; rotate 4 times; average.
3. Extra flexibility lets the model fit noise that is specific to the training sample.
4. It is a function of the training set, which is a random draw from $P^n$.
5. $\mathbb{E}_D[(h_D(x)-\bar h(x))^2]$, averaged over $x$.
6. Each cross term contains a quantity minus its own expectation, whose mean is zero; independence of the test point from $D$ lets the expectation factor.
7. The noise term $\mathbb{E}[(y-h^\star(x))^2]$.

</details>
