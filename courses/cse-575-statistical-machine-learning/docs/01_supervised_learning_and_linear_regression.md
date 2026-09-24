# 1. Supervised Learning and Linear Regression

**Source pages:** 2–6  
**Status:** reconstructed and equation-checked

This chapter introduces supervised learning, establishes the course notation, and develops linear regression through both the closed-form least-squares solution and gradient descent.

## 1.1 What does it mean for a machine to learn?

The source begins with two complementary definitions.

Arthur Samuel describes machine learning as the field that gives computers the ability to learn without being explicitly programmed. Tom Mitchell gives a more operational definition: a program learns from experience when its performance on a task improves according to a chosen performance measure.

Mitchell's definition can be organized using three objects:

- **Task, T:** the activity the system must perform, such as regression, classification, machine translation, anomaly detection, or density estimation.
- **Performance, P:** the quantity used to judge the system, such as prediction accuracy or another task-appropriate metric.
- **Experience, E:** the data or interactions from which the system learns.

For supervised learning, the experience contains input-output pairs. For unsupervised learning, the experience contains inputs without observed output labels.

> [!TIP]
> **Review intuition**
>
> The T-P-E framework prevents the phrase “the model learns” from being vague. A learning problem is not fully specified until the task, evidence, and success criterion are all defined.


## 1.2 Three broad modes of machine learning

The source distinguishes three modes:

1. **Supervised learning:** every training input has a known target.
2. **Unsupervised learning:** training inputs are available without target labels.
3. **Reinforcement learning:** listed in the handwritten annotation as another major learning mode, although it is not developed in these opening pages.

A supervised training set is:

$$ \mathcal{D}_{\mathrm{train}} = \left\lbrace \left(x^{(i)}, y^{(i)}\right)\right\rbrace_{i=1}^{n_{\mathrm{train}}}. $$

Here, $x^{(i)}$ is the input for training example $i$, and $y^{(i)}$ is its observed target.

An unsupervised dataset contains only the inputs:

$$ \mathcal{D}_{\mathrm{train}} = \left\lbrace x^{(i)}\right\rbrace_{i=1}^{n_{\mathrm{train}}}. $$

## 1.3 Regression and classification

The source divides supervised learning into two main task types.

| Task | Output type | Source examples |
|---|---|---|
| Regression | continuous numerical outcome | study hours to score; movie information to revenue |
| Classification | discrete category | spam versus not spam; animal category |

For regression, the output is real-valued:

$$ y^{(i)} \in \mathbb{R}. $$

For binary classification, the output can be encoded as zero or one:

$$ y^{(i)} \in \lbrace 0,1\rbrace. $$

For multiclass classification, the output belongs to a finite set of categories.

The supervised-learning workflow has two stages:

| Stage | What happens |
|---|---|
| **Training** | Combine data with known answers and a parameterized model. Fit the parameters using the training examples. |
| **Prediction** | Apply the fitted model to new data whose answer is unknown. Return a numerical value or category. |

## 1.4 Hypotheses and parameters

The model is written as $h_\theta(x)$ or $h(x;\theta)$, where $\theta$ denotes the learned parameters.

The source calls this object a **hypothesis**, a **parameterized model**, or a **parameterized function**. It represents an assumed relationship between the input and output.

Training chooses parameter values that minimize a loss over the training set:

$$ \theta^{\star} = \mathrm{arg\,min}_{\theta} L\left(\left\lbrace y^{(i)}, h_\theta\left(x^{(i)}\right)\right\rbrace_{i=1}^{n_{\mathrm{train}}}\right). $$

> [!NOTE]
> **Added clarification**
>
> The exact form of $L$ depends on the task and modeling assumptions. In the next sections, the course uses squared prediction error for linear regression.


## 1.5 The one-feature linear model

One of the simplest regression hypotheses is:

$$ h_\theta(x) = \theta_0 + \theta_1x. $$

The parameter vector is:

$$ \theta = \left[\theta_0,\theta_1\right]^\top. $$

The roles of the parameters are:

- $\theta_0$: intercept or bias.
- $\theta_1$: slope or coefficient of the input feature.

Different parameter values produce different candidate lines. Learning is the process of selecting the line that performs best according to the chosen objective.

## 1.6 Measuring prediction error

The source first reviews distance measures between two vectors $x^{(1)}$ and $x^{(2)}$.

The Euclidean or L2 distance is:

$$ d_2\left(x^{(1)},x^{(2)}\right) = \left(\sum_j \left|x_j^{(1)}-x_j^{(2)}\right|^2\right)^{1/2}. $$

The Manhattan or L1 distance is:

$$ d_1\left(x^{(1)},x^{(2)}\right) = \sum_j \left|x_j^{(1)}-x_j^{(2)}\right|. $$

The general Lp distance is:

$$ d_p\left(x^{(1)},x^{(2)}\right) = \left(\sum_j \left|x_j^{(1)}-x_j^{(2)}\right|^p\right)^{1/p}. $$

For linear regression, the course measures the vertical difference between the observed target and model prediction. The per-example residual is:

$$ r^{(i)} = y^{(i)} - h_\theta\left(x^{(i)}\right). $$

The course objective is the sum of squared residuals with a factor of one half:

$$ L(\theta) = \frac{1}{2}\sum_{i=1}^{n_{\mathrm{train}}}\left(y^{(i)}-h_\theta\left(x^{(i)}\right)\right)^2. $$

The source also presents the averaged version:

$$ L_{\mathrm{mean}}(\theta) = \frac{1}{n_{\mathrm{train}}}\sum_{i=1}^{n_{\mathrm{train}}}\left(y^{(i)}-h_\theta\left(x^{(i)}\right)\right)^2. $$

> [!NOTE]
> **Added clarification: RSS versus MSE**
>
> The slides use “RSS,” “loss,” and “mean squared error” near the same derivation. They differ only by constant scaling in this context: RSS sums squared residuals, while MSE divides by the number of examples. The factor $1/2$ is commonly inserted because it cancels the factor $2$ produced by differentiation. These constant factors do not change the minimizing parameter vector.


## 1.7 Vector representation

Introduce an augmented input vector that contains a constant coordinate:

$$ \tilde{x}^{(i)} = \left[1,x^{(i)}\right]^\top. $$

Then:

$$ h_\theta\left(x^{(i)}\right) = \left(\tilde{x}^{(i)}\right)^\top\theta. $$

Stack the training examples into a design matrix:

$$ X_{i,:} = \left(\tilde{x}^{(i)}\right)^\top, \qquad i=1,\ldots,n_{\mathrm{train}}. $$

Stack the observed targets into a vector:

$$ y = \left[y^{(1)},y^{(2)},\ldots,y^{(n_{\mathrm{train}})}\right]^\top. $$

All model predictions are then collected in $X\theta$, and all residuals are collected in $y-X\theta$.

## 1.8 Method of least squares

The scalar objective can be written in matrix form:

$$ L(\theta) = \frac{1}{2}(y-X\theta)^\top(y-X\theta). $$

This equality follows from the inner-product rule:

$$ (y-X\theta)^\top(y-X\theta) = \sum_{i=1}^{n_{\mathrm{train}}}\left(y^{(i)}-\left(\tilde{x}^{(i)}\right)^\top\theta\right)^2. $$

Expanding the quadratic objective gives:

$$ L(\theta) = \frac{1}{2}\left(y^\top y - 2y^\top X\theta + \theta^\top X^\top X\theta\right). $$

The gradient with respect to the parameter vector is:

$$ \nabla_\theta L(\theta) = X^\top X\theta - X^\top y. $$

At an optimum of this differentiable quadratic objective, the gradient is zero:

$$ X^\top X\theta - X^\top y = 0. $$

Therefore, the normal equations are:

$$ X^\top X\theta = X^\top y. $$

When $X^\top X$ is invertible, the least-squares solution is:

$$ \theta^{\star} = (X^\top X)^{-1}X^\top y. $$

> [!NOTE]
> **Technical note**
>
> The source presents the inverse form directly. The inverse requires $X^\top X$ to be nonsingular. Handling a singular design matrix is outside these pages and will not be developed here.


## 1.9 Gradient descent

The source next presents a generic iterative optimization method. Starting from an initial parameter vector $\theta^{(0)}$, gradient descent repeatedly moves in the negative-gradient direction:

$$ \theta^{(k+1)} = \theta^{(k)} - \eta\nabla_\theta L\left(\theta^{(k)}\right). $$

Here, $\eta$ is the learning rate.

For the simple function $f(x)=x^2$, the derivative is $f'(x)=2x$. A positive derivative means moving left lowers the function, while a negative derivative means moving right lowers it. The negative gradient therefore points locally toward decreasing objective values.

For linear regression with one input feature:

$$ L(\theta_0,\theta_1) = \frac{1}{2}\sum_{i=1}^{n_{\mathrm{train}}}\left(y^{(i)}-\theta_0-\theta_1x^{(i)}\right)^2. $$

The derivative with respect to the intercept is:

$$ \frac{\partial L}{\partial\theta_0} = -\sum_{i=1}^{n_{\mathrm{train}}}\left(y^{(i)}-\theta_0-\theta_1x^{(i)}\right). $$

The derivative with respect to the slope is:

$$ \frac{\partial L}{\partial\theta_1} = -\sum_{i=1}^{n_{\mathrm{train}}}\left(y^{(i)}-\theta_0-\theta_1x^{(i)}\right)x^{(i)}. $$

The simultaneous update is:

$$ \theta_0^{(k+1)} = \theta_0^{(k)} + \eta\sum_i\left(y^{(i)}-\theta_0^{(k)}-\theta_1^{(k)}x^{(i)}\right). $$

$$ \theta_1^{(k+1)} = \theta_1^{(k)} + \eta\sum_i\left(y^{(i)}-\theta_0^{(k)}-\theta_1^{(k)}x^{(i)}\right)x^{(i)}. $$

In matrix form, the same update is:

$$ \theta^{(k+1)} = \theta^{(k)} - \eta X^\top\left(X\theta^{(k)}-y\right). $$

> [!NOTE]
> **Closed-form least squares versus gradient descent**
>
> The handwritten note contrasts least squares as a specialized method that can produce the optimum directly with gradient descent as a generic optimization procedure that approaches an optimum iteratively.


## 1.10 Advanced linear regression through feature construction

Linear regression can include polynomial features:

$$ h_\theta(x) = \theta_0 + \theta_1x + \theta_2x^2 + \theta_3x^3 + \cdots. $$

It can also include other transformed features, such as trigonometric functions and logarithms.

For example, define a feature map:

$$ \phi(x) = \left[1,x,x^2,x^3\right]^\top. $$

The model becomes:

$$ h_\theta(x) = \theta^\top\phi(x). $$

Even though the prediction is nonlinear as a function of the raw input $x$, it is still a linear combination of the constructed features.

> [!NOTE]
> **Added clarification: what “linear” means**
>
> The source emphasizes that “linear” refers to taking a linear combination of features. Equivalently, the model is linear in its learned coefficients. The features themselves may be nonlinear transformations of the original input.


Increasing the polynomial degree makes the model more flexible. The source examples show that low-degree models may miss the shape of the data, while very high-degree models may produce unstable oscillations. The next chapter develops this issue as underfitting, overfitting, and generalization.

## 1.11 Common mistakes

1. **Confusing the fitted model with the learning algorithm.** The model is $h_\theta$; least squares or gradient descent is the procedure used to select $\theta$.
2. **Calling every squared-error expression MSE.** A sum, an average, and a half-scaled sum are related but not literally identical.
3. **Updating gradient-descent coordinates one at a time using already-updated values.** The displayed update is simultaneous: all coordinates on the right use iteration $k$.
4. **Thinking polynomial regression is nonlinear in its parameters.** It remains linear in the coefficients when polynomial terms are treated as features.
5. **Assuming the normal-equation inverse always exists.** The source solution is conditional on invertibility.

## 1.12 Chapter summary

- Supervised learning uses labeled input-output pairs.
- Regression predicts numerical targets; classification predicts categories.
- A hypothesis $h_\theta$ is a parameterized input-output mapping.
- Linear regression minimizes squared residuals.
- The least-squares solution follows from setting the matrix gradient to zero.
- Gradient descent optimizes the same objective iteratively.
- Nonlinear feature transformations can be used while retaining a model that is linear in its coefficients.

## 1.13 Self-check questions

1. In Mitchell's definition, what are T, P, and E for a spam classifier?
2. What is the distinction between a residual and the total squared-error objective?
3. Why does adding the factor $1/2$ not change the least-squares minimizer?
4. Starting from the matrix objective, how do the normal equations arise?
5. What condition is required for the displayed inverse solution?
6. Why can a cubic polynomial regression model still be called linear regression?
7. How do the specialized least-squares solution and gradient descent differ as optimization approaches?
