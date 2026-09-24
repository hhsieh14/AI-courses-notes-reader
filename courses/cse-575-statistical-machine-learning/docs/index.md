# CSE 575 Notes: Introduction

These notes follow the order of the course. Each chapter builds on the previous ones:

```text
1 linear regression ─► 2 generalization ─► 3 regularization + likelihood
                                              │
        ┌─────────────────────────────────────┴───────────────────────┐
        ▼                                                             ▼
4 KNN ─► 5 logistic/softmax ─► 6 evaluation ─► 7 generative ─► 8 SVM & kernels
                                                                      │
9 K-means ─► 10 GMM & EM ─► 11 PCA                                    │
                                                                      ▼
                                                  12 neural networks & backprop
```

Chapters 1–8 are supervised learning. Chapters 9–11 are unsupervised: clustering, latent-variable models and dimensionality reduction. Chapter 12 joins the two threads: a neural network is stacked logistic regressions trained by gradient descent, and it learns its own features instead of relying on hand-designed ones.

Three ideas recur throughout:

1. **Loss = negative log-likelihood.** Squared error (Gaussian), cross-entropy (Bernoulli/categorical) and even regularization (a prior) all come from one recipe.
2. **Bias–variance.** Polynomial degree, $\lambda$, $k$ in KNN, $C$ in SVMs and the number of PCA components are all knobs on the same trade-off.
3. **Linear in parameters, nonlinear in inputs.** Feature maps, kernels and hidden layers are three ways to get curved boundaries from linear machinery.

See the [chapter list](../README.md) and the [notation reference](notation.md).
