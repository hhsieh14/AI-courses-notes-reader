# Statistical Learning Theory: Introduction

CSE 575 (my other notes) asks *which models exist and how to fit them*. This course asks two harder questions:

1. **Why should a model that fits the training data work on new data?** Chapters 6–8 answer with concentration inequalities and complexity measures: Rademacher complexity and VC dimension.
2. **Why do our optimization algorithms find good solutions, and how fast?** Chapters 9–12 prove convergence rates for gradient descent, proximal methods and SGD, and end with why very wide neural networks train to zero error.

Kernel SVMs are the running example that ties the two together (Chapters 1–5). The final generalization bound for SVMs (Chapter 7) shows that the $\lVert w\rVert^2$ regularizer isn't a heuristic: it is exactly the quantity that controls the complexity term.

**How to read.** Each chapter states results as theorems and proves them. Where a proof is long and standard, I give its structure and a reference instead, and say so explicitly. "Beyond the lecture" boxes add results I found necessary to complete the picture (Radon's theorem for VC bounds, Sauer–Shelah, step-size schedules for SGD, the limits of the NTK regime).

Start with the [study guide](study_guide.md) for the map of results, or go straight to [Chapter 1](chapters/01_probabilistic_prediction.md). Symbols are listed in [notation](notation.md).
