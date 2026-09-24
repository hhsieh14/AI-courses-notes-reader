# 12. Optimization of Neural Networks

Neural-network training objectives are nonconvex, so none of the convex results from Chapters 9–11 apply directly. Yet very wide networks trained by gradient descent reliably reach zero training error and often generalize. This chapter follows the **neural tangent kernel** (NTK) explanation for a two-layer ReLU network (Du et al., 2019; Arora et al., 2019). When the network is wide enough, its training dynamics are close to those of kernel regression with a fixed kernel. That brings back both the linear-convergence tools of Chapter 9 and the Rademacher bounds of Chapter 7.

## 1. Why neural networks?

Neural networks are multilayered prediction models whose output is computed from the input through a feed-forward sequence of transformations.

Different hidden layers can extract features at different levels of abstraction. In an image classifier, early layers detect low-level patterns, middle layers combine them into intermediate structures, and later layers form higher-level semantic features.

We won't analyze an arbitrary deep network. We study the training behavior of a **two-layer neural network** consisting of:

- one hidden layer;
- one output layer.

> [!TIP]
> **What does two-layer mean here?**
>
> The input coordinates are not counted as a layer. The two computational stages are the hidden layer and the output layer. In the special model analyzed later, only the hidden-layer weights are optimized while the output signs remain fixed.

## 2. A single neuron

Let the input be

$$x=[x_{1},x_{2},\ldots,x_{d}]^{\top}\in\mathbb{R}^{d}.$$

A hidden neuron has a weight vector

$$w=[w_{1},w_{2},\ldots,w_{d}]^{\top}\in\mathbb{R}^{d}$$

and an activation function $\sigma$.

The neuron first computes the weighted input

$$z=w^{\top}x,$$

then outputs

$$\sigma(z)=\sigma(w^{\top}x).$$

### 2.1 Sigmoid activation

One common activation is the sigmoid:

$$\sigma(z)=\frac{1}{1+e^{-z}}.$$

Its values lie in $(0,1)$.

### 2.2 ReLU activation

The analysis uses the rectified linear unit:

$$\sigma(z)=\max\{0,z\}.$$

For $z>0$, the derivative is $1$. For $z<0$, the derivative is $0$. The gradient calculations therefore use

$$\sigma'(z)=\mathbf{1}_{\{z\geq 0\}}.$$

> [!NOTE]
> **Derivative at zero**
>
> ReLU is not differentiable at $z=0$. We use the indicator convention above. Since a continuously distributed random initialization hits an exact zero inner product with probability zero under nondegenerate inputs, this convention does not affect the stated probabilistic argument.

## 3. The two-layer model

Suppose the hidden layer contains $m$ neurons. The $r$-th hidden neuron has weight

$$w_{r}\in\mathbb{R}^{d},\qquad 1\leq r\leq m,$$

and produces

$$\sigma(w_{r}^{\top}x).$$

The output layer associates a fixed sign

$$a_{r}\in\{-1,1\}$$

with the output of hidden neuron $r$.

The complete network is

$$f(W,x)=\frac{1}{\sqrt{m}}\sum_{r=1}^{m}a_{r}\sigma(w_{r}^{\top}x).$$

The collection of hidden-layer weights is

$$W=\{w_{r}\}_{r=1}^{m}.$$

Only $W$ is trained. The output signs $a_{1},\ldots,a_{m}$ are sampled once and then held fixed.

> [!TIP]
> **Why include the factor $1/\sqrt{m}$?**
>
> This width normalization is chosen so that the scale of the network output and its tangent-kernel matrix remains controlled as the number of hidden neurons grows.

## 4. Training data and empirical objective

The training data are

$$\{(x_{i},y_{i})\}_{i=1}^{n},$$

where

$$x_{i}\in\mathbb{R}^{d}$$

and

$$y_{i}\in\{-1,1\}.$$

The training objective is

$$\Phi(W)=\frac{1}{2n}\sum_{i=1}^{n}\left(y_{i}-f(W,x_{i})\right)^{2}.$$

Equivalently, if the vector of training predictions is

$$\widehat{y}(W)=[f(W,x_{1}),f(W,x_{2}),\ldots,f(W,x_{n})]^{\top},$$

and the label vector is

$$y=[y_{1},y_{2},\ldots,y_{n}]^{\top},$$

then

$$\Phi(W)=\frac{1}{2n}\lVert\widehat{y}(W)-y\rVert_{2}^{2}.$$

## 5. Random initialization

Initialize every hidden neuron as

$$w_{r}(0)\sim\mathcal{N}(0,\kappa^{2}I),\qquad 1\leq r\leq m,$$

where $\kappa\neq 0$ and $I$ is the $d\times d$ identity matrix.

The output-layer signs are independently sampled from the uniform distribution on $\{-1,1\}$:

$$P(a_{r}=1)=P(a_{r}=-1)=\frac{1}{2}.$$

All random variables in

$$\{w_{r}(0)\}_{r=1}^{m}$$

and

$$\{a_{r}\}_{r=1}^{m}$$

are mutually independent.

> [!TIP]
> **What is random after initialization?**
>
> The initialization is random, but once it has been sampled, ordinary full-batch gradient descent is deterministic. Probability statements later in the chapter are taken over the random training sample and the random initialization.

## 6. Gradient-descent training

Start from

$$W(0)=\{w_{r}(0)\}_{r=1}^{m}.$$

At iteration $k\geq 1$, update every hidden weight by

$$w_{r}(k)=w_{r}(k-1)-\eta\nabla_{w_{r}}\Phi(W(k-1)),$$

where $\eta>0$ is the learning rate.

### 6.1 Gradient with respect to one hidden neuron

For one input $x_{i}$,

$$\nabla_{w_{r}}f(W,x_{i})=\frac{a_{r}}{\sqrt{m}}\mathbf{1}_{\{w_{r}^{\top}x_{i}\geq 0\}}x_{i}.$$

Applying the chain rule to the squared empirical objective gives

$$\nabla_{w_{r}}\Phi(W)=\frac{a_{r}}{n\sqrt{m}}\sum_{i=1}^{n}\left(f(W,x_{i})-y_{i}\right)\mathbf{1}_{\{w_{r}^{\top}x_{i}\geq 0\}}x_{i}.$$

At iteration $k$, the proof-consistent form is therefore

$$\nabla_{w_{r}}\Phi(W(k-1))=\frac{a_{r}}{n\sqrt{m}}\sum_{i=1}^{n}\left(f(W(k-1),x_{i})-y_{i}\right)\mathbf{1}_{\{w_{r}(k-1)^{\top}x_{i}\geq 0\}}x_{i}.$$

> [!NOTE]
> **Iteration index**
>
> The indicator must use the *current* weights $w_r(k-1)$, the point where the gradient is evaluated. Writing $w_r(k)$ there is an easy slip to make.

## 7. Vectorized gradient dynamics

Stack the $m$ hidden-layer vectors into

$$\mathrm{vec}(W)\in\mathbb{R}^{md}.$$

Define the activation indicator

$$I_{r,i}(k)=\mathbf{1}_{\{w_{r}(k)^{\top}x_{i}\geq 0\}}.$$

Stack the per-example gradients into a block matrix

$$Z(k)\in\mathbb{R}^{md\times n}.$$

To avoid a fragile rendered matrix environment, define it columnwise. The $i$-th column of $Z(k)$ is

$$Z_{:,i}(k)=\frac{1}{n\sqrt{m}}[I_{1,i}(k)a_{1}x_{i}^{\top},\ldots,I_{m,i}(k)a_{m}x_{i}^{\top}]^{\top}.$$

Let

$$\widehat{y}(k)=[f(W(k),x_{1}),f(W(k),x_{2}),\ldots,f(W(k),x_{n})]^{\top}.$$

Then the vectorized parameter update is

$$\mathrm{vec}(W(k))=\mathrm{vec}(W(k-1))-\eta Z(k-1)(\widehat{y}(k-1)-y).$$

> [!NOTE]
> **Two normalizations**
>
> With the $\frac{1}{n}$ of the objective folded into $Z(k)$, the Gram matrix $Z^\top Z$ is $1/n^2$ times the "kernel-scale" matrix whose entries are $\frac{x_i^\top x_j}{m}\sum_rI_{r,i}I_{r,j}$. The two conventions differ only by rescaling $\eta$. Below I use the kernel scale, so $\mathbb E[H(0)]=K$ exactly.

## 8. Prediction-space dynamics

The change in the training predictions is approximately

$$\widehat{y}(k)-\widehat{y}(k-1)=-\eta Z(k-1)^{\top}Z(k-1)(\widehat{y}(k-1)-y)+e(k-1),$$

where the remainder satisfies the order statement

$$e(k)=O\left(\frac{1}{\sqrt{m}}\right)\lVert\widehat{y}(k)-y\rVert_{2}.$$

Define

$$H(k)=Z(k)^{\top}Z(k)\in\mathbb{R}^{n\times n}.$$

Subtracting the label vector gives the residual recursion

$$\widehat{y}(k)-y=(I-\eta H(k-1))(\widehat{y}(k-1)-y)+e(k-1).$$

> [!TIP]
> **Why is the width $m$ important?**
>
> The approximation error is stated as order $1/\sqrt{m}$. A wider network therefore makes the prediction dynamics closer to a linear recursion driven by the Gram matrix $H(k)$.

> [!NOTE]
> **What makes this rigorous**
>
> The Taylor remainder $e(k)$ comes from ReLU activation patterns flipping between steps. Du et al. show that if every $w_r$ stays within $O(1/\sqrt m)$ of its initialization, only a small fraction of patterns flip, which bounds $e(k)$ along the whole trajectory.

## 9. From the changing Gram matrix to the NTK matrix

The analysis uses two approximation facts.

### Fact 1: $H(k)$ stays close to $H(0)$

With probability at least $1-\delta$ over the random initialization (for fixed training inputs), Du et al. (2019) show

$$\lVert H(k)-H(0)\rVert_{F}=O\left(\frac{1}{\sqrt{m}\,\delta^{3/2}}\right).$$

Thus, for sufficiently large width, the Gram matrix changes little during training.

### Fact 2: $H(0)$ is close to a deterministic matrix $K$

For normalized inputs, define

$$K_{i,j}=\frac{x_{i}^{\top}x_{j}\left(\pi-\arccos(x_{i}^{\top}x_{j})\right)}{2\pi}.$$

and

$$K_{i,j}=\mathbb{E}[H_{i,j}(0)].$$

The matrix $K\in\mathbb{R}^{n\times n}$ is called the Gram matrix of the **Neural Tangent Kernel**, or NTK, for this two-layer model with fixed output signs.

> [!NOTE]
> **Input normalization**
>
> The displayed formula uses $\arccos(x_{i}^{\top}x_{j})$, so it implicitly requires inner products in $[-1,1]$. This is naturally satisfied when the inputs are normalized. The pages do not separately state that assumption.

> [!TIP]
> **Where does the angle formula come from?**
>
> For a Gaussian hidden weight, the two indicators are simultaneously active with probability equal to the fraction of directions lying in the intersection of two half-spaces. For unit vectors, that probability is $(\pi-\arccos(x_{i}^{\top}x_{j}))/(2\pi)$.

## 10. Concentration of the initial Gram matrix

### Lemma 1

With probability at least $1-\delta$ over the training data and random initialization,

$$\lVert H(0)-K\rVert_{F}=O\left(\frac{n\sqrt{\log(n/\delta)}}{\sqrt{m}}\right).$$

### Proof

Fix a pair $(i,j)$. At the kernel scale,

$$H_{i,j}(0)=\frac{x_{i}^{\top}x_{j}}{m}\sum_{r=1}^{m}I_{r,i}(0)I_{r,j}(0).$$

The terms

$$I_{r,i}(0)I_{r,j}(0)$$

are independent Bernoulli random variables over the hidden-neuron index $r$.

Since

$$\mathbb{E}[H_{i,j}(0)]=K_{i,j},$$

Hoeffding's inequality gives, with probability at least $1-\delta'$, the entrywise bound

$$\lvert H_{i,j}(0)-K_{i,j}\rvert\leq\sqrt{\frac{\log(2/\delta')}{2m}}.$$

Apply a union bound over all $n^{2}$ ordered pairs. With probability at least $1-n^{2}\delta'$, every pair satisfies the same bound. Therefore,

$$\lVert H(0)-K\rVert_{F}^{2}=\sum_{i=1}^{n}\sum_{j=1}^{n}(H_{i,j}(0)-K_{i,j})^{2}.$$

Using the entrywise estimate,

$$\lVert H(0)-K\rVert_{F}^{2}\leq\frac{n^{2}\log(2/\delta')}{2m}.$$

Set

$$\delta'=\frac{\delta}{n^{2}}.$$

Then

$$\lVert H(0)-K\rVert_{F}\leq n\sqrt{\frac{\log(2n^{2}/\delta)}{2m}}.$$

Hence,

$$\lVert H(0)-K\rVert_{F}=O\left(\frac{n\sqrt{\log(n/\delta)}}{\sqrt{m}}\right).$$

This proves the lemma.

> [!NOTE]
> **Bounded inner products**
>
> This Hoeffding constant treats the indicator average as the random part and assumes the multiplicative inner product is bounded in magnitude. Normalized inputs provide $\lvert x_{i}^{\top}x_{j}\rvert\leq 1$.

## 11. Approximate linear convergence in prediction space

Combining the three approximations gives

$$H(k)\approx H(0)\approx K$$

and treats $e(k)$ as negligible. The residual recursion becomes

$$\widehat{y}(k)-y\approx(I-\eta K)(\widehat{y}(k-1)-y).$$

Iterating,

$$\widehat{y}(k)-y\approx(I-\eta K)^{k}(\widehat{y}(0)-y).$$

The important simplification is that $K$ does not depend on the iteration number. It can be computed from the training inputs before running gradient descent.

### 11.1 Spectral contraction

For any matrix $A$ and vector $x$,

$$\lVert Ax\rVert_{2}\leq\lVert A\rVert_{2}\lVert x\rVert_{2}.$$

Assume that $K$ is positive definite and let

$$\lambda_{0}=\lambda_{\min}(K)>0.$$

Choose

$$\eta=O\left(\frac{\lambda_{0}}{n^{2}}\right)$$

so that $I-\eta K$ is positive semidefinite and

$$\lVert I-\eta K\rVert_{2}=1-\eta\lambda_{0}\in(0,1).$$

Therefore,

$$\lVert\widehat{y}(k)-y\rVert_{2}\lesssim(1-\eta\lambda_{0})^{k}\lVert\widehat{y}(0)-y\rVert_{2}.$$

> [!NOTE]
> **Positive definiteness is an assumption**
>
> A kernel Gram matrix is always positive semidefinite, but its minimum eigenvalue may be zero. The geometric convergence statement requires the stronger condition $\lambda_{0}>0$.

## 12. Informal optimization theorem

### Theorem 1

For sufficiently large $m$ and the learning-rate scale stated above, we get the informal bound

$$\Phi(W(k))\leq(1-\eta\lambda_{0})^{2k}\lVert\widehat{y}(0)-y\rVert_{2}^{2}$$

for every $k\geq 0$.

Since

$$0<1-\eta\lambda_{0}<1,$$

we have

$$\Phi(W(k))\longrightarrow 0$$

as $k\to\infty$.

Thus an overparameterized two-layer network can fit the training labels under the stated NTK-style conditions.

> [!NOTE]
> Here $\Phi$ is written without its $\frac{1}{2n}$ factor; the rate is unaffected.

> [!NOTE]
> **Why "informal"**
>
> Making this a theorem needs uniform control over all $k$ of the three approximations, $e(k)\approx0$, $H(k)\approx H(0)$ and $H(0)\approx K$, and an explicit width. For unit-variance initialization, Du et al. (2019) need $m=\Omega\!\left(n^6/(\lambda_0^4\delta^3)\right)$. Later work lowers the polynomial but not the idea.

## 13. Optimization is not generalization

Driving $\Phi(W(k))$ to zero controls the empirical training error. It does not by itself imply small population risk.

Now consider a loss satisfying the Lipschitz condition

$$\lvert\ell(f(W,x),y)-\ell(f(W,x'),y)\rvert\leq M\lvert f(W,x)-f(W,x')\rvert$$

and

$$\ell(y,y)=0$$

for all $y\in[-1,1]$.

From the earlier Rademacher-complexity chapter, with probability at least $1-\delta$, every $f\in\mathcal{F}$ satisfies

$$R(f)\leq\widehat{R}_{n}(f)+2M\mathfrak{R}_{n}(\mathcal{F})+\sqrt{\frac{\log(1/\delta)}{2n}}.$$

The empirical-risk term can become small during training, but a useful population-risk bound also requires the function class explored by gradient descent to have controlled complexity.

> [!TIP]
> **Do we get rid of empirical risk?**
>
> No. Gradient descent still minimizes the empirical squared objective. The generalization theorem adds an upper bound on population risk after training. The model does not directly minimize the displayed population-risk upper bound.

## 14. Distance from initialization and the reachable function class

Arora et al. (2019) show that, throughout training,

$$\lVert\mathrm{vec}(W(k))-\mathrm{vec}(W(0))\rVert_{2}\leq\sqrt{y^{\top}K^{-1}y}+\text{a small noise term}.$$

This motivates the function class

$$\mathcal{F}=\left\lbrace f(W,\cdot):\lVert\mathrm{vec}(W)-\mathrm{vec}(W(0))\rVert_{2}\leq\sqrt{y^{\top}K^{-1}y}+\text{a small noise term}\right\rbrace.$$

Every network encountered along the analyzed gradient-descent path belongs to this ball around the random initialization.

> [!TIP]
> **Why does staying near initialization matter?**
>
> A smaller parameter ball is a smaller function class. The Rademacher-complexity term can therefore be controlled using the maximum distance traveled from the initialization.

> [!NOTE]
> The proof (Arora et al., 2019) sums the per-step movements along the approximately linear trajectory, $\sum_k\eta\,Z(k)(\hat y(k)-y)$, and bounds the sum using $(I-\eta K)^k$ and $K^{-1}$.

## 15. Final population-risk bound

The final stated result is that, with probability at least $1-\delta$ over the training data and random initialization,

$$R(f)\leq M\sqrt{\frac{2y^{\top}K^{-1}y}{n}}+O\left(\sqrt{\frac{\log(n/(\lambda_{0}\delta))}{n}}\right),$$

provided that $m$ is sufficiently large and

$$\eta=O\left(\frac{\lambda_{0}}{n^{2}}\right).$$

This is the main generalization result of Arora, Du, Hu, Li & Wang, *Fine-Grained Analysis of Optimization and Generalization for Overparameterized Two-Layer Neural Networks* (ICML 2019).

The bound has two central messages:

1. the concentration remainder has the familiar order $\sqrt{\log(n/(\lambda_{0}\delta))/n}$;
2. the leading term depends on the data-dependent quantity $y^{\top}K^{-1}y$, whose scaling determines whether the complete bound is small.

## 16. Spectral interpretation of $y^{\top}K^{-1}y$

Suppose

$$K=U\Lambda U^{\top},$$

where the eigenvalues satisfy

$$\lambda_{1}\geq\lambda_{2}\geq\cdots\geq\lambda_{n}>0.$$

Then

$$y^{\top}K^{-1}y=\sum_{j=1}^{n}\frac{(u_{j}^{\top}y)^{2}}{\lambda_{j}}.$$

This quantity is small when most of the label vector lies in eigendirections associated with large eigenvalues of $K$.

That is what it means for the label vector is well represented by the top few eigenvectors of the NTK Gram matrix.

> [!TIP]
> **Why do top eigenvectors help both convergence and generalization?**
>
> Along an eigenvector with eigenvalue $\lambda_{j}$, the approximate residual is multiplied by $(1-\eta\lambda_{j})^{k}$. Large-eigenvalue components decay faster. The same components contribute $(u_{j}^{\top}y)^{2}/\lambda_{j}$ to the final complexity quantity, so they are also cheaper in the generalization bound.

## 17. Interpretation of the final figures

Arora et al. compare three label patterns on MNIST:

- a worst-case direction;
- random labels;
- the actual MNIST labels.

The actual-label loss falls rapidly, while the worst-case construction converges very slowly. The accompanying eigenvalue-projection plot indicates that the real labels have stronger alignment with the leading NTK eigendirections than random or adversarial label vectors.

The experiments support the main conclusion: overparameterization alone is not the entire explanation. The interaction between the data, labels, and NTK spectrum determines both the optimization speed and the population-risk upper bound.

## 18. What is proved here and what is cited

Proved in this chapter:

- the gradient formula for one hidden weight;
- the vectorized gradient update;
- concentration of $H(0)$ around $K$ (Hoeffding plus a union bound).

Cited from Du et al. (2019) and Arora et al. (2019):

- the prediction-space Taylor approximation and the remainder $e(k)$;
- stability of $H(k)$ around $H(0)$ throughout training;
- the width requirement;
- the distance-from-initialization bound;
- the final population-risk theorem.

> [!NOTE]
> **Beyond the lecture: the limits of the NTK picture**
>
> In the NTK regime the network barely moves from initialization, so it behaves like a fixed kernel method and does **not learn features**. Real networks at practical widths move much further, and their feature learning is part of why they beat kernels on images and text (Chizat, Oyallon & Bach, 2019, "lazy training"). The NTK story explains *why* gradient descent finds global minima of wide nonconvex networks, but not everything about why deep learning works.

## 19. Chapter summary

We analyzed a two-layer ReLU network with fixed random output signs and trainable hidden weights. In the overparameterized regime, the hidden activation patterns move little, so the changing tangent Gram matrix remains close to its initialization and to a deterministic NTK matrix $K$.

The resulting prediction dynamics are approximately linear:

$$\widehat{y}(k)-y\approx(I-\eta K)^{k}(\widehat{y}(0)-y).$$

A positive minimum eigenvalue of $K$ gives geometric decay of the training residual. Generalization is then controlled by combining a Rademacher-complexity bound with the fact that gradient descent remains in a parameter ball around initialization. The final data-dependent quantity

$$y^{\top}K^{-1}y$$

connects label alignment, optimization speed, and the population-risk upper bound.

---

[← Previous: Stochastic Gradient Descent](11_stochastic_gradient_descent.md) · [Course map](../course_map.md)
