# Machine Learning Course Notes

Study notes from three graduate courses I took for my master's at Arizona State University. I rebuilt my handwritten notes into reading material for anyone learning this material. Each chapter explains the idea, derives the key results, works an example, runs code to check the numbers, and ends with answered questions.

| Course | What it covers | Start here |
|---|---|---|
| [**CSE 575: Statistical Machine Learning**](courses/cse-575-statistical-machine-learning/README.md) | the core toolbox: regression, regularization, k-NN, logistic/softmax, evaluation, generative classifiers, SVMs and kernels, k-means, GMM/EM, PCA, neural networks | [Chapter 1](courses/cse-575-statistical-machine-learning/docs/01_supervised_learning_and_linear_regression.md) |
| [**Statistical Learning Theory**](courses/statistical-learning-theory/README.md) | *why* learning works: margins and kernels, RKHS, concentration, Rademacher complexity, VC dimension, and convex, proximal and stochastic optimization | [Study guide](courses/statistical-learning-theory/docs/study_guide.md) |
| [**CSE 598: Modern Temporal Learning**](courses/modern-temporal-learning/README.md) | data ordered in time: ARIMA and exponential smoothing, wavelets, HMMs, RNN/LSTM/GRU, TCNs, contrastive learning, transformers | [Course map](courses/modern-temporal-learning/course_map.md) |

## How the three fit together

```text
CSE 575                    What models and algorithms do we have?
   ↓
Learning Theory            Why should a model trained on finite data generalize, and how do we optimize it?
   ↓
Modern Temporal Learning   What changes when the order and timing of observations matter?
```

The chapters link to each other across courses. For example, the EM derivation in CSE 575 is reused for HMMs in Temporal Learning, and the kernel trick connects SVMs (CSE 575), RKHS theory (Learning Theory) and kernel PCA (Temporal Learning).

## A few places to start

If you only have a few minutes, these pages show how the notes work:

- **[Bias–variance, simulated](courses/cse-575-statistical-machine-learning/docs/02_generalization_validation_bias_variance.md):** the decomposition derived, then checked by fitting polynomials of degree 1, 3 and 9 to 500 resampled datasets.
- **[Backpropagation with a gradient check](courses/cse-575-statistical-machine-learning/docs/12_neural_networks_and_backpropagation.md):** the backward pass derived by hand and verified against finite differences.
- **[VC dimension](courses/statistical-learning-theory/docs/chapters/08_vc_dimension.md):** why half-spaces in $\mathbb R^d$ shatter $d+1$ points but not $d+2$ (Radon's theorem), and the Sauer–Shelah bound.
- **[Rademacher generalization bound](courses/statistical-learning-theory/docs/chapters/07_generalization_and_rademacher.md):** a step-by-step proof, with the assumptions (bounded, Lipschitz losses) stated where each step uses them.
- **[Why EWMA works](courses/modern-temporal-learning/notes/12_handwritten_appendix.md):** differencing a noisy random walk gives MA(1), and the matching MA coefficient makes exponential smoothing the optimal forecaster. Derived, then simulated.
- **[LSTMs and vanishing gradients](courses/modern-temporal-learning/notes/08_rnn_lstm_gru_and_seq2seq.md):** the Jacobian-product argument, and exactly which part of the LSTM fixes it.
- **[Transformers for time series](courses/modern-temporal-learning/notes/11_transformers.md):** attention from first principles, why the $\sqrt{d_k}$ scaling is needed, and when a linear baseline still wins.

## Runnable notebooks

Each course has a companion notebook that re-derives its key results numerically. Outputs are saved, so they read fine on GitHub; open one in Colab to change the numbers and rerun. Everything runs on a CPU in about a minute.

| Course | Notebook | |
|---|---|---|
| CSE 575 | [cse575_companion.ipynb](courses/cse-575-statistical-machine-learning/notebooks/cse575_companion.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hhsieh14/AI-courses-notes-reader/blob/main/courses/cse-575-statistical-machine-learning/notebooks/cse575_companion.ipynb) |
| Learning Theory | [slt_companion.ipynb](courses/statistical-learning-theory/notebooks/slt_companion.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hhsieh14/AI-courses-notes-reader/blob/main/courses/statistical-learning-theory/notebooks/slt_companion.ipynb) |
| Temporal Learning | [mtl_companion.ipynb](courses/modern-temporal-learning/notebooks/mtl_companion.ipynb) | [![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hhsieh14/AI-courses-notes-reader/blob/main/courses/modern-temporal-learning/notebooks/mtl_companion.ipynb) |

To run locally: `pip install -r requirements.txt`, then open a notebook in Jupyter.

## How each chapter is organized

- **Intuition first**, then the math, then a worked example.
- **"Beyond the lecture" notes** mark material I added from papers and textbooks to fill gaps or connect ideas.
- **Warnings** flag common mistakes, including ones I made the first time.
- **Code snippets** are small and runnable. Their printed outputs are real results.
- **Questions and answers** at the end of each chapter, in collapsible blocks so you can test yourself first.
- **Consistent notation** within each course; each course has a notation page.

## About these notes

These are my independent study notes, not official course materials. They don't include the original course PDFs, slides, exams or assignments. Course titles are given for attribution, and rights in the original instructional materials remain with their authors and ASU.

I used AI tools (Claude) as an editor and checker: to turn handwritten notes into clean prose and LaTeX, redraw diagrams from my sketches, check derivations and code, and find inconsistencies. The selection of material, the explanations and the understanding are mine. Where the AI review found an error in my notes, I kept the corrected version and, when the mistake is a common one, turned it into a warning for readers.

## Related

- [ML System Design notes](https://github.com/hhsieh14/ml-system-design-learning-notes): how models like these are designed, evaluated and deployed in production systems.
