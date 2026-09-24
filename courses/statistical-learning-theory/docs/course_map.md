# Course Map

| # | Chapter | Main results |
|---:|---|---|
| 1 | [Probabilistic Prediction and Bayes Classification](chapters/01_probabilistic_prediction.md) | Bayes classifier, excess-risk identity, plug-in bound |
| 2 | [Hard-Margin SVM and Duality](chapters/02_hard_margin_svm.md) | Lagrangian duality, KKT, support vectors |
| 3 | [Feature Maps and Kernels](chapters/03_feature_maps_and_kernels.md) | Hilbert spaces, kernel trick |
| 4 | [Soft-Margin SVM and Hinge Loss](chapters/04_soft_margin_svm.md) | slack = hinge loss |
| 5 | [RKHS and the Representer Theorem](chapters/05_rkhs_and_representer.md) | reproducing property, Moore–Aronszajn, representer theorem |
| 6 | [Concentration Inequalities](chapters/06_concentration_inequalities.md) | Hoeffding's lemma and inequality, McDiarmid |
| 7 | [Generalization Bounds and Rademacher Complexity](chapters/07_generalization_and_rademacher.md) | symmetrization, contraction, RKHS bound, SVM risk bound |
| 8 | [VC Dimension](chapters/08_vc_dimension.md) | shattering, half-spaces via Radon, Sauer–Shelah |
| 9 | [Smooth Convex Optimization](chapters/09_smooth_convex_optimization.md) | descent lemma, GD $O(1/k)$, AGD $O(1/k^2)$ |
| 10 | [Nonsmooth and Proximal Optimization](chapters/10_nonsmooth_proximal_optimization.md) | subgradients, soft thresholding, prox-GD $O(1/k)$ |
| 11 | [Stochastic Gradient Descent](chapters/11_stochastic_gradient_descent.md) | averaged SGD bound, noise floor, $O(1/\sqrt k)$ |
| 12 | [Optimization of Neural Networks](chapters/12_neural_network_optimization.md) | NTK, linear convergence, $y^\top K^{-1}y$ generalization |

Chapters 1–5 build kernel SVMs, 6–8 explain why they generalize, and 9–12 explain how they, and neural networks, are trained.
