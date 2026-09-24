# Statistical Learning Theory

My notes from the graduate Statistical Learning Theory course at Arizona State University: the theory behind *why* machine learning works. Every major result is stated as a theorem and proved, or explicitly cited when the proof is beyond the course.

## Start here

- [Introduction](docs/index.md): what questions the course answers
- [Study guide and theorem map](docs/study_guide.md): every result, where it's proved, and the recurring proof patterns
- [Course map](docs/course_map.md): chapter list
- [Questions and answers](docs/questions_and_discussion.md): 60 self-test questions with answers
- [Notation](docs/notation.md)

## Parts

| Part | Chapters | Big idea |
|---|---|---|
| Prediction, SVMs and kernels | [1](docs/chapters/01_probabilistic_prediction.md)–[5](docs/chapters/05_rkhs_and_representer.md) | Bayes rule, max margin, duality, RKHS, representer theorem |
| Concentration and generalization | [6](docs/chapters/06_concentration_inequalities.md)–[8](docs/chapters/08_vc_dimension.md) | Hoeffding/McDiarmid, Rademacher complexity, VC dimension |
| Optimization | [9](docs/chapters/09_smooth_convex_optimization.md)–[12](docs/chapters/12_neural_network_optimization.md) | GD, proximal GD, SGD, NTK analysis of wide networks |

## Runnable notebook

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/hhsieh14/AI-courses-notes-reader/blob/main/courses/statistical-learning-theory/notebooks/slt_companion.ipynb)

[`notebooks/slt_companion.ipynb`](notebooks/slt_companion.ipynb): each result checked numerically: Bayes risk, support vectors on the margin, the representer theorem, Hoeffding vs the exact tail, Rademacher complexity ∝ 1/√n, VC dimension of half-planes by linear programming, GD vs Nesterov rates, ISTA/FISTA, SGD step sizes, and lazy training of wide networks. Outputs are saved, so you can read it on GitHub without running anything.

Related: [CSE 575 notes](../cse-575-statistical-machine-learning/README.md) (the applied side of the same models) and [temporal learning notes](../modern-temporal-learning/README.md).
