---
course: "Statistical Learning Theory"
chapter: "00"
title: "Questions and Discussion"
source_pages: "598SLT.pdf, pp. 2-81"
status: "consolidated question index"
---

# Questions and Discussion

This page preserves questions written in the notebook rather than resolving them silently inside the course narrative.

## Probabilistic prediction

1. Why do we need learning methods if the Bayes decision function is already optimal?
2. Why does partitioning the input space into smaller cells help estimate a conditional probability?
3. How does cell size trade approximation accuracy against the number of observations per cell?
4. Why does disagreement with the Bayes classifier matter less near $\eta(x)=1/2$?

## SVM and duality

1. Why are binary labels encoded as $\{-1,1\}$?
2. Why is the maximum-margin problem equivalent to minimizing $\lVert w\rVert_{2}^{2}$ under canonical constraints?
3. Why solve the dual problem rather than the primal problem directly?
4. Which assumptions make the KKT conditions sufficient here?
5. Why does the final weight vector depend only on support vectors?

## Kernels and RKHS

1. Why must a kernel be positive definite?
2. What fails if the proposed inner product is only nonnegative but not definite?
3. What is the reproducing property, and why is it useful?
4. How is the pre-Hilbert space completed into an RKHS?
5. Why does the representer theorem restrict the optimizer to the span of training features?

## Concentration and generalization

1. Which assumptions are essential for each concentration inequality?
2. Why are moment-generating functions useful for probability tails?
3. Why does the proof of Hoeffding's lemma need only a second-order Taylor remainder rather than an infinite expansion?
4. Why can the standard hinge loss not be inserted directly into the bounded-loss Hoeffding application?
5. How does bounded coordinate sensitivity lead to a bound on the conditional martingale difference used in McDiarmid's proof?
6. Why does a union bound become ineffective for an infinite collection of sets?
7. What is the conceptual difference between a pointwise and a uniform generalization bound?
8. Why does the complexity of the function class appear through an expected supremum?
9. Why does taking a supremum make the bound valid for the data-dependent empirical-risk minimizer?
10. What role does the ghost sample play in symmetrization?
11. Why do random Rademacher signs measure a class's ability to fit noise?
12. Which assumptions are needed for the contraction principle used in the notes?
13. Why does an RKHS norm bound control Rademacher complexity?
14. Why does the SVM regularizer have both a maximum-margin interpretation and a generalization-bound interpretation?

## VC dimension

1. Why does shattering require all $2^{\lvert S\rvert}$ subsets rather than only many of them?
2. Why can intervals shatter two ordered points but not three?
3. What geometric assumption is needed for three points in $\mathbb{R}^{2}$ to be shattered by half-spaces?
4. Why does one non-shattered four-point example not by itself prove that no four-point set can be shattered?
5. How does the combinatorial interpretation of VC dimension differ from the random-sign interpretation of Rademacher complexity?

## Optimization

1. Why is a quadratic loss a convenient first objective for gradient descent?
2. Why does the source use an intercept feature $x_{0}=1$ in the vectorized regression model?
3. How does the Lipschitz constant $L$ constrain the usable learning rate?
4. Why does the descent lemma give an upper quadratic approximation, while convexity gives a tangent-hyperplane lower bound?
5. Why do the distance terms telescope in the gradient-descent proof?
6. Why is the final objective gap no larger than the average of the previous gaps?
7. What property gives accelerated gradient descent its improved rate?
8. Can accelerated gradient descent use a tolerance-based stopping rule rather than only a fixed iteration count?
9. Why is a composite objective split into a differentiable term $g$ and a possibly nondifferentiable term $h$?
10. Why does the proximal mapping solve an optimization problem instead of taking an ordinary gradient step on $h$?
11. Why does soft thresholding create exact zero coefficients rather than merely small coefficients?
12. Why is the subdifferential of $\lvert x\rvert$ at zero the entire interval $[-1,1]$?
13. How does the proximal optimality condition convert the update into a subgradient inequality?
14. Where is the condition $\eta\leq 1/L$ used in the proximal-gradient proof?
15. Why is the last-iterate objective gap bounded by the average of all previous gaps?
16. Which argument is missing from the source for the accelerated proximal-gradient rate $O(1/k^{2})$?

## Stochastic gradient descent

1. How should stochastic-gradient noise appear in the convergence analysis?
2. Why does uniform random sampling make the component gradient unbiased for the full gradient?
3. Why does the theorem use the averaged iterate $\bar{x}^{(k)}$ instead of only the final raw iterate $x^{(k)}$?
4. Which variance condition is actually required by the proof on page 74?
5. Where is the learning-rate condition $\eta\leq 1/L$ used?
6. Why does a fixed learning rate leave the residual term $\eta\sigma^{2}$ in the bound?
7. Why are the one-step expectations conditional on the earlier sampled indices?
8. How does Jensen's inequality connect the average of the objective values to the objective at the average iterate?
9. Which additional argument would be needed to obtain the almost-sure convergence result mentioned by the source?

## Neural-network optimization

1. Why are the output-layer signs $a_{r}$ fixed while only the hidden weights $w_{r}$ are trained in this lecture?
2. Why does the ReLU gradient introduce the activation indicator $I_{r,i}(k)$?
3. Which iteration index should appear inside the activation indicator when computing $\nabla_{w_{r}}\Phi(W(k-1))$?
4. How do the two normalization conventions for the tangent-feature matrix change the Gram matrix and effective learning rate?
5. Which part of the prediction-space recursion is exact, and which part is absorbed into the remainder $e(k)$?
6. Why does overparameterization help keep $H(k)$ close to $H(0)$?
7. How does Hoeffding's inequality produce entrywise concentration of $H(0)$ around $K$?
8. Why is the assumption $\lambda_{0}=\lambda_{\min}(K)>0$ essential for geometric convergence?
9. Where is empirical risk used, and where is population risk introduced?
10. Why does staying near the random initialization restrict the complexity of the reachable function class?
11. How does the eigendecomposition of $K$ explain the quantity $y^{\top}K^{-1}y$?
12. Why can alignment with leading NTK eigenvectors improve both optimization speed and the population-risk upper bound?
13. Which lemmas are proved in the notes, and which difficult steps are imported from the external reference?
14. Does the final generalization theorem mean that the network directly minimizes population risk?
