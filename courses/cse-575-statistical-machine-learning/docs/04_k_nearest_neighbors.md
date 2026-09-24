# 4. K-Nearest Neighbors

K-nearest neighbors (KNN) is the simplest method that makes no assumption about the *shape* of the prediction function. There is no line, polynomial or sigmoid. It only assumes that nearby points have similar outputs. That makes it a good lens on four decisions every method has to make in some form: how to represent examples, how to measure closeness, how local to be, and how to scale features.

## 4.1 Classification by similarity

A classification problem needs quantifiable features, known labels, and a way to measure similarity. Two running examples:

- a flower shop predicts a customer's next purchase from what similar customers bought;
- a patient's survival is predicted from two numbers, age and the number of positive lymph nodes (this is the classic Haberman survival dataset).

> [!TIP]
> **Intuition**
>
> KNN's only assumption is that examples close together in feature space tend to have similar outputs. Everything else follows from what "close" means.

## 4.2 The classification rule

For a query $x$, let $\mathcal{N}_k(x)$ be the indices of the $k$ closest training examples. Count the votes for each class $c$,

$$ S_c(x)=\sum_{i\in\mathcal{N}_k(x)}\mathbf{1}\left[y^{(i)}=c\right], $$

and predict the majority:

$$ \hat{y}(x)=\mathrm{arg\,max}_{c}\,S_c(x). $$

Ties need a rule. Common choices are preferring the class of the single closest neighbor, weighting votes by $1/d$, or using an odd $k$ for two classes.

## 4.3 Measuring closeness

The default is Euclidean distance:

$$ d_2(x,z)=\left(\sum_{j=1}^{d}\left(x_j-z_j\right)^2\right)^{1/2}. $$

The distance is part of the model. Change it (Manhattan, cosine, a learned metric) and you change which examples count as neighbors, and therefore the predictions.

## 4.4 How k changes the model

**$k=1$.** The prediction is the label of the single nearest point. The decision boundary bends around individual examples and can leave small islands. It is very flexible and very sensitive to noise, so it tends to overfit.

**$k=n_{\text{train}}$.** Every query uses the whole training set, so the prediction is the global majority class everywhere. It ignores all local structure and underfits.

The useful $k$ lies in between and is chosen with validation or cross-validation (Chapter 2).

![KNN classification and regression intuition](assets/diagrams/04_knn_core_intuition.svg)

*Neighbor voting, boundaries that smooth out as $k$ grows, and KNN regression as local averaging.*

## 4.5 Feature scaling

Euclidean distance is dominated by whichever feature has the largest numeric range. If one feature spans 0–5 and another 10–60, a 1-unit difference in the first contributes $1^2$ to the squared distance while a 10-unit difference in the second contributes $100$. The second feature decides the neighborhood even if both matter equally.

Three common rescalings, each fitted on the **training** split only:

| Method | Formula | Result |
|---|---|---|
| Standardization | $x_j'=(x_j-\mu_j)/s_j$ | mean 0, variance 1 |
| Min–max | $x_j'=(x_j-x_{j,\min})/(x_{j,\max}-x_{j,\min})$ | range $[0,1]$ |
| Max-abs | $x_j'=x_j/\max_i\lvert x_j^{(i)}\rvert$ | range $[-1,1]$, keeps zeros (good for sparse data) |

with $\mu_j=\frac1n\sum_i x_j^{(i)}$ and $s_j^2=\frac1n\sum_i(x_j^{(i)}-\mu_j)^2$.

> [!WARNING]
> **Fit the scaler on training data only**
>
> Compute $\mu_j, s_j$, minima and maxima on the training split and reuse them for validation, test and production inputs. Fitting them on all the data leaks test information into the model. In scikit-learn, put the scaler inside a `Pipeline` so cross-validation refits it per fold.

## 4.6 Scaling usually helps, but not always

Scaling fixes unequal ranges. It doesn't fix redundancy. If $x_2=2x_1$, then after standardization the two columns are identical, and Euclidean distance counts the same information twice. Similarly, scaling makes a pure-noise feature just as influential as a signal feature. When features are correlated or irrelevant, what helps is feature selection, PCA (Chapter 11) or a learned metric, not more scaling.

## 4.7 KNN regression

For a numeric target, average the neighbors instead of voting:

$$ \hat{y}(x)=\frac{1}{k}\sum_{i\in\mathcal{N}_k(x)}y^{(i)}. $$

Small $k$ gives a jagged, step-like curve that follows individual points. Larger $k$ averages more points and smooths the curve. At $k=n_{\text{train}}$ every prediction equals the global mean $\bar y$, a constant.

## 4.8 k and effective complexity

**Increasing $k$ decreases the effective complexity of KNN.** This runs opposite to most hyperparameters we've seen, where a bigger number (polynomial degree, network width) means a more flexible model.

| $k$ | Neighborhood | Bias | Variance | Risk |
|---|---|---|---|---|
| small | very local | low | high | overfitting |
| large | broad average | high | low | underfitting |

A useful rule of thumb: KNN has roughly $n/k$ "effective parameters", since the training set is effectively split into $n/k$ neighborhoods. So when you draw the usual U-shaped validation curve, put $1/k$ (or $n/k$) on the complexity axis, not $k$.

## 4.9 Classification versus regression

| | KNN classification | KNN regression |
|---|---|---|
| Neighbor outputs | class labels | numeric targets |
| Aggregation | majority vote | mean |
| Very small $k$ | irregular regions | jagged steps |
| Very large $k$ | global majority | global mean |

## 4.10 Beyond the lecture: where KNN breaks, and where it shows up at scale

> [!NOTE]
> **The curse of dimensionality.** In high dimensions, distances concentrate: the nearest and farthest neighbors end up almost equally far away, so "nearest" carries little information. KNN works best on low-dimensional or well-embedded data. That is one reason to reduce dimension first (Chapter 11) or learn an embedding.
>
> **Cost.** KNN has no training step, but every prediction scans the training set, at $O(nd)$ per query. KD-trees help in low dimensions. At production scale we use **approximate nearest neighbor** (ANN) indexes such as HNSW, IVF-PQ or ScaNN, which give up a little recall for orders-of-magnitude speed. This is exactly the candidate-retrieval stage in my [ranking system design notes](https://github.com/hhsieh14/ml-system-design-learning-notes/blob/main/chapters/01_ranking_model/chapter_01_design_a_ranking_model.md): embed users and items, then retrieve the nearest items with ANN.

## 4.11 Common mistakes

1. **Thinking larger $k$ means a more complex model.** It is the opposite.
2. **Using unscaled features with very different ranges.**
3. **Fitting the scaler on validation/test data.**
4. **Assuming scaling removes redundant or irrelevant features.**
5. **Choosing $k=1$ because it fits the training set perfectly.** Training accuracy at $k=1$ is always 100% (each point is its own neighbor), so it tells you nothing.
6. **Forgetting that the distance function defines the model.**

## 4.12 Summary

- KNN predicts from the $k$ nearest training examples: a vote for classification, a mean for regression.
- Small $k$ means low bias and high variance; large $k$ the reverse.
- Distance-based methods need scaled features, with the scaler fitted on training data.
- KNN degrades in high dimensions and is served at scale with ANN indexes.

## 4.13 Self-check

1. What assumption makes KNN reasonable?
2. Why does $k=1$ have high variance?
3. What happens at $k=n_{\text{train}}$?
4. Why does Euclidean distance need scaled features?
5. Why doesn't scaling solve the $x_2=2x_1$ problem?
6. Why is 1-NN training accuracy meaningless?

<details>
<summary>Answers</summary>

1. Points close in feature space have similar outputs.
2. The prediction depends on one training point, so any noise in that point moves the prediction.
3. The prediction is the global majority class or global mean, constant for all inputs.
4. Otherwise the feature with the largest range dominates the distance.
5. Scaling changes units, not information; both columns still encode the same variable, so it is counted twice.
6. Every training point is its own nearest neighbor (distance 0), so it is always classified correctly.

</details>
