# 2. Generalization, Validation, Bias, and Variance

**Source pages:** 7–11  
**Status:** reconstructed and equation-checked

This chapter asks a central question: after fitting a model to the available examples, how well will it predict new data? The source develops this question through model complexity, data splitting, cross-validation, and the bias–variance decomposition of expected test error.

## 2.1 Model complexity and generalization

The previous chapter introduced polynomial feature construction:

$$ h_\theta(x) = \theta_0 + \theta_1x + \theta_2x^2 + \theta_3x^3 + \cdots. $$

Increasing the polynomial degree increases the model's flexibility. The source compares three representative cases:

| Model behavior | Training fit | Prediction on unseen data |
|---|---|---|
| Low complexity | poor | poor |
| Intermediate complexity | good | potentially best |
| Very high complexity | very good | potentially poor |

A low-degree model may fail to represent the shape of the data. A very high-degree model may follow small fluctuations in the training sample rather than the underlying pattern.

The source uses three related terms:

- **Generalization:** prediction performance measured on unseen data.
- **Model selection:** estimating the performance of multiple candidate models in order to choose one.
- **Model assessment:** after choosing a final model, estimating its prediction or generalization error on new unseen data.

> [!TIP]
> **Review intuition**
>
> Training error answers, “How well did this model fit the examples it saw?” Generalization error asks, “How well will the learned rule work on examples it did not see?”


## 2.2 Underfitting and overfitting

The source labels the low-complexity case **underfitting** and the high-complexity case **overfitting**.

### Underfitting

An underfit model is too restricted to capture the relevant structure. It performs poorly even on the training data and also predicts poorly on new data.

### Overfitting

An overfit model follows the training sample very closely but does not reproduce the same performance on unseen examples. Its training error can be small while its generalization error remains large.

### A useful comparison

| Regime | Training error | Validation or test error | Main issue |
|---|---:|---:|---|
| Underfitting | high | high | model is not flexible enough |
| Appropriate complexity | lower | lower | model captures useful structure |
| Overfitting | very low | higher | model is too sensitive to the observed sample |

The source's polynomial plots make the distinction visual: degree 1 misses the curved trend, a moderate degree follows it, and degree 15 oscillates strongly around individual points.

## 2.3 Training, validation, and test sets

The source first presents a two-way training/test split:

- **Training data:** fit the model.
- **Test data:** predict the held-out labels, compare predictions with the actual values, and measure error.

It then introduces a three-way split so that model selection and final assessment are separated:

- **Training set:** fit model parameters.
- **Validation set:** select a model based on prediction error.
- **Test set:** assess the generalization error of the selected final model.

The source gives example proportions of 50/25/25 and 70/20/10 for training, validation, and test data.

> [!WARNING]
> **Why the test set has a separate role**
>
> If test performance is repeatedly used to choose the model, then the test set becomes part of the selection process. It no longer provides an independent final assessment. The source's three-way split prevents this role confusion.


## 2.4 K-fold cross-validation

K-fold cross-validation divides the available model-selection data into $K$ folds. Each fold is held out once, while the remaining folds are used for fitting. The held-out errors are then averaged.

Let $J_{\text{holdout}}^{(j)}$ denote the error measured when fold $j$ is held out. The average cross-validation error is:

$$ J_{\mathrm{CV}} = \frac{1}{K}\sum_{j=1}^{K} J_{\text{holdout}}^{(j)}. $$

For a candidate model setting $m$, such as a polynomial degree, the comparison can be written as:

$$ J_{\mathrm{CV}}(m) = \frac{1}{K}\sum_{j=1}^{K} J_{\text{holdout}}^{(j)}(m). $$

The source's diagram uses four folds. In each row, one quarter is marked as the held-out split and the other three quarters are training splits. The final cross-validation result is the average of the four held-out results.

> [!NOTE]
> **Added clarification: the diagram's label**
>
> The source labels each held-out fold “Test Split.” In a three-way workflow, these rotating folds serve the validation or model-selection role. A separate untouched test set is still needed for final model assessment.


## 2.5 Model complexity versus error

The source shows the classical relationship between model complexity and error:

- Training error usually decreases as complexity increases.
- Cross-validation or test error initially decreases because the model becomes less underfit.
- After an intermediate point, cross-validation or test error may increase because the model becomes more sensitive to the particular training sample.

Using the source notation, training error is written as $J_{\mathrm{train}}(\theta)$ and cross-validation error as $J_{\mathrm{CV}}(\theta)$.

The model-selection goal is not to minimize training error alone. It is to choose a complexity with low estimated generalization error.

> [!NOTE]
> **Handwritten extension: the overparameterized regime**
>
> The handwritten sketch on page 9 extends the classical U-shaped test-error curve. It suggests that near an interpolation threshold, test error may rise and then decrease again as the model becomes heavily overparameterized. This is a second-descent or double-descent pattern. The note therefore cautions that the relationship between a raw parameter count and generalization need not always be a single U-shaped curve.


## 2.6 The learned model depends on the sampled training set

Suppose the training dataset is sampled from an underlying distribution:

$$ D \sim P^n. $$

Let a learning algorithm $\mathcal{A}$ map the dataset to a fitted predictor:

$$ h_D = \mathcal{A}(D). $$

Because $D$ is random, the learned parameters and the learned function are also random. Different training samples can produce different parameter estimates:

$$ \hat{\theta}_1, \qquad \hat{\theta}_2, \qquad \hat{\theta}_3, \qquad \ldots $$

and therefore different fitted functions:

$$ f(x;\hat{\theta}_1), \qquad f(x;\hat{\theta}_2), \qquad f(x;\hat{\theta}_3), \qquad \ldots $$

The source's linear-regression examples compare repeated fits on different sampled datasets:

- **Low variability:** the fitted curves remain similar across samples.
- **High variability:** the fitted curves change substantially across samples.

This variability is a property of the learning procedure together with its model class, not merely of one fitted curve.

## 2.7 Bias and variance

For a scalar parameter estimate $\hat{\theta}$ of a true parameter $\theta$, the source defines bias as:

$$ \mathrm{Bias}(\hat{\theta}) = \mathbb{E}_D[\hat{\theta}] - \theta. $$

Its variance is:

$$ \mathrm{Var}(\hat{\theta}) = \mathbb{E}_D\left[\left(\hat{\theta}-\mathbb{E}_D[\hat{\theta}]\right)^2\right]. $$

The same ideas can be applied directly to predictions. Define the average predictor at input $x$ as:

$$ \bar{h}(x) = \mathbb{E}_D[h_D(x)]. $$

If $h^{\star}(x)$ denotes the underlying target function, the prediction bias at $x$ is:

$$ \mathrm{Bias}_h(x) = \bar{h}(x) - h^{\star}(x). $$

The prediction variance at $x$ is:

$$ \mathrm{Var}_h(x) = \mathbb{E}_D\left[\left(h_D(x)-\bar{h}(x)\right)^2\right]. $$

The source summarizes the classical trade-off as follows:

- More complex models generally make fewer restrictive assumptions and can have lower bias.
- The same flexibility can make the learned result depend more strongly on the sampled dataset, producing higher variance.

> [!NOTE]
> **Added clarification: parameter variance versus prediction variance**
>
> Page 9 introduces bias and variance using parameter estimates, while pages 10–11 decompose prediction error using fitted functions. These are related but distinct objects. The expected test-error decomposition below uses prediction bias and prediction variance.


## 2.8 Expected targets and expected predictors

The handwritten derivation begins from a joint data distribution:

$$ p(x,y) = p(y\mid x)p(x). $$

For a fixed input $x$, define the expected target:

$$ \bar{y}(x) = \mathbb{E}_{y\mid x}[y] = \int y\,p(y\mid x)\,dy. $$

The source also writes the target as:

$$ y = h^{\star}(x) + \epsilon. $$

In the decomposition, $h^{\star}(x)$ and $\bar{y}(x)$ play the role of the conditional mean target, while $\epsilon$ represents the remaining target noise.

The expected predictor averages the outputs of models learned from different possible training datasets:

$$ \bar{h}(x) = \mathbb{E}_{D\sim P^n}[h_D(x)]. $$

> [!NOTE]
> **Source terminology**
>
> The handwritten page sometimes calls $h_D$ a “classifier,” but the displayed derivation uses real-valued targets and squared error. Mathematically, it is a squared-error predictor or regressor in this section.


## 2.9 Expected error of a fitted model and of a learning algorithm

For one fixed learned model $h_D$, its expected squared test error is:

$$ R(h_D) = \mathbb{E}_{(x,y)\sim P}\left[\left(y-h_D(x)\right)^2\right]. $$

The handwritten notes also express this expectation as an integral:

$$ R(h_D) = \int_x\int_y \left(y-h_D(x)\right)^2p(x,y)\,dy\,dx. $$

However, a learning algorithm can output different models when it receives different training datasets. To evaluate the algorithm itself, the source also averages over possible datasets:

$$ R(\mathcal{A}) = \mathbb{E}_{D\sim P^n}\mathbb{E}_{(x,y)\sim P}\left[\left(y-h_D(x)\right)^2\right]. $$

Conceptually, this means repeating the following experiment:

1. Sample a training dataset $D$.
2. Train $h_D=\mathcal{A}(D)$.
3. Sample a new test pair $(x,y)$.
4. Measure squared prediction error.
5. Average over repetitions.

The expected error of the algorithm therefore describes the stability and accuracy of the learning procedure across possible training samples, not only the result of one observed run.

## 2.10 Decomposition of expected test error

The source decomposes expected squared test error into three components: noise, squared bias, and variance.

### Step 1: separate target noise from model error

Start with:

$$ \mathbb{E}_{D,(x,y)}\left[\left(y-h_D(x)\right)^2\right]. $$

Add and subtract the underlying target function $h^{\star}(x)$:

$$ y-h_D(x) = \left(y-h^{\star}(x)\right) + \left(h^{\star}(x)-h_D(x)\right). $$

After squaring and taking expectations, the source sets the cross term to zero because the test noise has conditional mean zero and the sampled test example is independent of the training dataset. This gives:

$$ \mathbb{E}_{D,(x,y)}\left[\left(y-h_D(x)\right)^2\right] = \mathbb{E}_{x,y}\left[\left(y-h^{\star}(x)\right)^2\right] + \mathbb{E}_{D,x}\left[\left(h^{\star}(x)-h_D(x)\right)^2\right]. $$

The first term is target noise. The second term is error caused by the learned predictor differing from the underlying target function.

### Step 2: decompose model error around the average predictor

Add and subtract $\bar{h}(x)=\mathbb{E}_D[h_D(x)]$:

$$ h^{\star}(x)-h_D(x) = \left(h^{\star}(x)-\bar{h}(x)\right) + \left(\bar{h}(x)-h_D(x)\right). $$

The cross term again vanishes because:

$$ \mathbb{E}_D\left[\bar{h}(x)-h_D(x)\right] = 0. $$

Therefore:

$$ \mathbb{E}_{D,x}\left[\left(h^{\star}(x)-h_D(x)\right)^2\right] = \mathbb{E}_x\left[\left(h^{\star}(x)-\bar{h}(x)\right)^2\right] + \mathbb{E}_{D,x}\left[\left(\bar{h}(x)-h_D(x)\right)^2\right]. $$

The first term is squared bias and the second term is variance.

### Final decomposition

Combining the two steps yields:

$$ \mathbb{E}_{D,(x,y)}\left[\left(y-h_D(x)\right)^2\right] = \mathbb{E}_{x,y}\left[\left(y-h^{\star}(x)\right)^2\right] + \mathbb{E}_x\left[\left(h^{\star}(x)-\bar{h}(x)\right)^2\right] + \mathbb{E}_{D,x}\left[\left(h_D(x)-\bar{h}(x)\right)^2\right]. $$

In compact form:

$$ \text{expected test error} = \text{noise} + \text{bias}^2 + \text{variance}. $$

The source marks the noise term as uncontrollable because it comes from variability in the data-generating process. Bias and variance arise from the model class and learning algorithm.

> [!NOTE]
> **Fixed-target version**
>
> The handwritten summary also gives “fixed target error = bias squared + variance.” This corresponds to measuring prediction error relative to the conditional mean target $h^{\star}(x)$ rather than relative to a noisy observed value $y$.


## 2.11 Why the cross terms disappear

The page 11 derivation explicitly verifies the zero-mean relationships used above.

For the learned predictor:

$$ \mathbb{E}_D\left[h_D(x)-\bar{h}(x)\right] = \mathbb{E}_D[h_D(x)]-\bar{h}(x)=0. $$

For the target:

$$ \mathbb{E}_{y\mid x}\left[y-\bar{y}(x)\right] = \mathbb{E}_{y\mid x}[y]-\bar{y}(x)=0. $$

These identities make the corresponding mixed terms vanish after taking expectations. The decomposition is therefore an algebraic consequence of adding and subtracting the appropriate mean predictor and mean target.

## 2.12 Interpreting the decomposition

The source uses the decomposition diagnostically:

| Component | Interpretation | Typical failure pattern |
|---|---|---|
| Noise | unavoidable variation in observed targets | remains even for the correct conditional mean |
| Squared bias | average prediction is systematically away from the target function | underfitting |
| Variance | fitted prediction changes substantially across training samples | overfitting |

The handwritten summary states:

- Underfitting is associated with high bias.
- Overfitting is associated with high variance.
- Model design involves a bias–variance trade-off.

Bias itself can be positive or negative. Squaring it makes both directions contribute positively to expected squared error.

The practical point of the decomposition is not only to name three terms. If a developed algorithm is not accurate enough, the dominant term suggests a different remedy: a high-bias problem and a high-variance problem should not be treated as the same failure.

## 2.13 Common mistakes

1. **Choosing the model using the final test set.** This mixes model selection with model assessment.
2. **Assuming the smallest training error gives the best model.** Training error generally falls with complexity, even after generalization begins to worsen.
3. **Treating the fitted model as deterministic.** The learned model depends on the randomly sampled training dataset.
4. **Confusing observation noise with model variance.** Noise comes from the conditional distribution of the target; variance comes from changes in the learned predictor across datasets.
5. **Ignoring the square on bias.** The error decomposition contains squared prediction bias, not signed bias.
6. **Assuming test error must always follow one U-shaped curve.** The handwritten overparameterization sketch records a possible second-descent regime.
7. **Calling the rotating cross-validation fold the final test set.** In model selection, it functions as a validation fold; final assessment requires separate untouched data.

## 2.14 Chapter summary

- Generalization measures prediction performance on unseen data.
- Model selection chooses among candidate models; model assessment estimates the selected model's final generalization error.
- Training, validation, and test sets have distinct roles.
- K-fold cross-validation averages held-out performance across multiple partitions.
- Underfitting is associated with insufficient flexibility and high bias.
- Overfitting is associated with sensitivity to the sampled data and high variance.
- A fitted model is random because it depends on the sampled training set.
- Expected squared test error decomposes into noise, squared bias, and variance.
- The source also notes that heavily overparameterized models may exhibit a second-descent pattern beyond the classical U-shaped curve.

## 2.15 Self-check questions

1. What is the distinction between model selection and model assessment?
2. Why are validation and test sets assigned different roles?
3. How is four-fold cross-validation performed?
4. Why can training error continue decreasing while test error increases?
5. Why should $h_D$ be treated as a random function?
6. How are $\bar{y}(x)$ and $\bar{h}(x)$ defined, and what is the difference between them?
7. Which expectation produces prediction variance?
8. Why do the cross terms vanish in the bias–variance derivation?
9. What part of expected test error is described as uncontrollable in the source?
10. How does the handwritten overparameterization sketch extend the classical complexity-error curve?
