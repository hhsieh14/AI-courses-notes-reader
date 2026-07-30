# Notation

This page consolidates notation introduced in the opening pages of the course notes.

| Symbol | Meaning |
|---|---|
| input x with sample superscript i | input variable or feature vector for sample i |
| output y with sample superscript i | target or output for sample i |
| theta | model-parameter vector |
| h sub theta | parameterized hypothesis or model |
| n train | number of training examples |
| X | design matrix containing the training inputs |
| y | vector containing all training targets |
| L | training objective or loss |
| eta | gradient-descent learning rate |

A supervised training set is written as:

$$ \mathcal{D}_{\mathrm{train}} = \left\lbrace \left(x^{(i)}, y^{(i)}\right)\right\rbrace_{i=1}^{n_{\mathrm{train}}}. $$

For one-dimensional linear regression, the feature vector includes a constant coordinate:

$$ \tilde{x}^{(i)} = \left[1,x^{(i)}\right]^\top, \qquad \theta = \left[\theta_0,\theta_1\right]^\top. $$

Then the model can be written as an inner product:

$$ h_\theta\left(x^{(i)}\right) = \left(\tilde{x}^{(i)}\right)^\top \theta. $$


## Generalization notation

| Symbol | Meaning |
|---|---|
| $D$ | randomly sampled training dataset |
| $P$ | underlying data distribution |
| $\mathcal{A}$ | learning algorithm |
| $h_D=\mathcal{A}(D)$ | predictor learned from dataset $D$ |
| $\bar{h}(x)$ | expected predictor across training datasets |
| $\bar{y}(x)$ | conditional mean target at input $x$ |
| $h^{\star}(x)$ | underlying target function used in the source derivation |
| $J_{\mathrm{CV}}$ | average cross-validation error |

The average learned predictor is:

$$ \bar{h}(x) = \mathbb{E}_D[h_D(x)]. $$

The expected target is:

$$ \bar{y}(x) = \mathbb{E}_{y\mid x}[y]. $$

## Regularization and probability notation

| Symbol | Meaning |
|---|---|
| $\lambda$ | regularization strength |
| $\Omega(\theta)$ | parameter-penalty function |
| $\phi_j(x)$ | transformed feature $j$ |
| $d_{\max}$ | number or maximum index of penalized features in the source formulas |
| $p_{\mathrm{data}}$ | unknown data-generating distribution |
| $p_{\mathrm{model}}(x;\theta)$ | parameterized model distribution |
| $L(\theta)$ | likelihood function in the probabilistic sections |
| $\ell(\theta)$ | log-likelihood |
| $D_{\mathrm{KL}}(p\parallel q)$ | Kullback–Leibler divergence from $p$ to $q$ |
| $\sigma^2$ | Gaussian noise variance |
| $z$ | latent representation in the page 15 generative-model sketch |

The general regularized objective is:

$$ L_{\mathrm{regularized}}(\theta)=L_{\mathrm{data}}(\theta)+\lambda\Omega(\theta). $$

The Gaussian-noise assumption for probabilistic linear regression is:

$$ y\mid x;\theta\sim\mathcal{N}\left(h_\theta(x),\sigma^2\right). $$


## K-nearest-neighbor notation

| Symbol | Meaning |
|---|---|
| $k$ | number of neighbors used for a prediction |
| $\mathcal{N}_k(x)$ | indices of the $k$ training examples closest to query $x$ |
| $d_2(x,z)$ | Euclidean distance between feature vectors $x$ and $z$ |
| $\hat{y}(x)$ | prediction for query input $x$ |
| $\mu_j$ | training-set mean of feature $j$ |
| $s_j$ | training-set standard deviation of feature $j$ |

KNN regression predicts by averaging neighboring targets:

$$ \hat{y}(x)=\frac{1}{k}\sum_{i\in\mathcal{N}_k(x)}y^{(i)}. $$


## Logistic, softmax, and GLM notation

| Symbol | Meaning |
|---|---|
| $\sigma(z)$ | logistic or sigmoid function |
| $h_\theta(x)$ | binary class-one probability in logistic regression |
| $\ell(\theta)$ | log-likelihood |
| $K$ | number of classes |
| $\theta_j$ | parameter vector for class $j$ |
| $\phi_j$ | probability of class $j$ |
| $\eta$ | natural or canonical parameter of an exponential-family distribution |
| $T(y)$ | sufficient statistic |
| $a(\eta)$ | log-partition function |
| $b(y)$ | observation-only factor in exponential-family form |

The sigmoid function is:

$$ \sigma(z)=\frac{1}{1+e^{-z}}. $$

Binary logistic regression uses:

$$ h_\theta(x)=p(y=1\mid x;\theta)=\sigma\left(\theta^\top x\right). $$

Softmax regression uses:

$$ p(y=j\mid x;\theta)=\frac{e^{\theta_j^\top x}}{\sum_{r=1}^{K}e^{\theta_r^\top x}}. $$

The exponential-family form used on source page 23 is:

$$ p(y;\eta)=b(y)\exp\left(\eta^\top T(y)-a(\eta)\right). $$


## Classification-evaluation notation

| Symbol | Meaning |
|---|---|
| $TP$ | true positives |
| $FN$ | false negatives |
| $FP$ | false positives |
| $TN$ | true negatives |
| $\mathrm{TPR}$ | true-positive rate or sensitivity |
| $\mathrm{FPR}$ | false-positive rate, equal to $1-\mathrm{Specificity}$ |
| $F_1$ | harmonic mean of precision and recall |
| $\tau$ | classification threshold |
| $\mathrm{AUC}$ | area under the ROC curve |

Recall or sensitivity is:

$$ \mathrm{Recall}=\mathrm{Sensitivity}=\frac{TP}{TP+FN}. $$

Precision is:

$$ \mathrm{Precision}=\frac{TP}{TP+FP}. $$

The ROC axes are:

$$ \mathrm{TPR}=\frac{TP}{TP+FN}, \qquad \mathrm{FPR}=\frac{FP}{FP+TN}. $$


## Generative-classification notation

| Symbol | Meaning |
|---|---|
| $p(y)$ | class-prior distribution |
| $p(x\mid y)$ | class-conditional input distribution |
| $p(y\mid x)$ | posterior class distribution |
| $\pi$ | Bernoulli probability for class $1$ in binary GDA |
| $\mu_0,\mu_1$ | class-conditional Gaussian means |
| $\Sigma$ | covariance shared by both classes in LDA |
| $\Sigma_0,\Sigma_1$ | class-specific covariance matrices in QDA |
| $\mathbb{1}[\cdot]$ | indicator function |
| $s_k(x)$ | unnormalized score for class $k$ |

Bayes' rule is:

$$ p(y\mid x)=\frac{p(x\mid y)p(y)}{p(x)}. $$

The GDA assumptions with shared covariance are:

$$ y\sim\mathrm{Bernoulli}(\pi), \qquad x\mid y=k\sim\mathcal{N}(\mu_k,\Sigma). $$

The Naive Bayes factorization is:

$$ p(x\mid y)=\prod_{i=1}^{n}p(x_i\mid y). $$

## Support-vector-machine and kernel notation

| Symbol | Meaning |
|---|---|
| $w$ | normal vector of the separating hyperplane |
| $b$ | hyperplane offset |
| $y^{(i)}\in\lbrace -1,+1\rbrace$ | SVM class label |
| $\delta_{\mathrm{func}}^{(i)}$ | functional margin of example $i$ |
| $\delta_{\mathrm{geo}}^{(i)}$ | geometric margin of example $i$ |
| $\xi_i$ | nonnegative slack variable for example $i$ |
| $C$ | soft-margin penalty weight |
| $\phi(x)$ | feature map |
| $K(x,z)$ | kernel or feature-space inner product |
| $\mathbf{K}$ | kernel Gram matrix |
| $\sigma$ | Gaussian-kernel bandwidth |

The SVM decision hyperplane is:

$$ w^\top x+b=0. $$

The geometric margin is:

$$ \delta_{\mathrm{geo}}^{(i)}=\frac{y^{(i)}\left(w^\top x^{(i)}+b\right)}{\lVert w\rVert_2}. $$

The soft-margin constraints are:

$$ y^{(i)}\left(w^\top x^{(i)}+b\right)\geq 1-\xi_i, \qquad \xi_i\geq 0. $$

A kernel is defined by:

$$ K(x,z)=\phi(x)^\top\phi(z). $$

## Unsupervised-learning and K-means notation

| Symbol | Meaning |
|---|---|
| $K$ | number of clusters |
| $c^{(i)}$ | cluster assignment of example $i$ |
| $\mu_j$ | centroid of cluster $j$ |
| $\mathcal{M}$ | set of centroids already selected during K-means++ |
| $D(x^{(i)})$ | distance from point $x^{(i)}$ to its nearest selected centroid |
| $J(c,\mu)$ | K-means inertia or distortion objective |

The K-means assignment rule is:

$$ c^{(i)}\leftarrow\underset{j\in\lbrace 1,\ldots,K\rbrace}{\mathrm{arg\,min}}\left\lVert x^{(i)}-\mu_j\right\rVert_2^2. $$

The centroid update is:

$$ \mu_j\leftarrow\frac{\sum_{i=1}^{n_{\mathrm{train}}}\mathbb{1}\left[c^{(i)}=j\right]x^{(i)}}{\sum_{i=1}^{n_{\mathrm{train}}}\mathbb{1}\left[c^{(i)}=j\right]}. $$

The inertia objective is:

$$ J(c,\mu)=\sum_{i=1}^{n_{\mathrm{train}}}\left\lVert x^{(i)}-\mu_{c^{(i)}}\right\rVert_2^2. $$

## Gaussian-mixture and EM notation

| Symbol | Meaning |
|---|---|
| $z^{(i)}$ | latent component assignment for example $i$ |
| $\phi_k$ | mixture weight or prior probability of component $k$ |
| $\mu_k$ | mean of Gaussian component $k$ |
| $\Sigma_k$ | covariance of Gaussian component $k$ |
| $w_k^{(i)}$ | responsibility of component $k$ for example $i$ |
| $N_k$ | effective number of observations assigned to component $k$ |
| $Q(z)$ | auxiliary or variational distribution over latent variables |
| $\Theta$ | collection of mixture parameters |
| $\mathrm{ELBO}$ | evidence lower bound |

The Gaussian-mixture density is:

$$ p(x;\Theta)=\sum_{k=1}^{K}\phi_k\mathcal{N}\left(x\mid\mu_k,\Sigma_k\right). $$

The E-step responsibility is:

$$ w_k^{(i)}=\frac{\phi_k\mathcal{N}\left(x^{(i)}\mid\mu_k,\Sigma_k\right)}{\sum_{j=1}^{K}\phi_j\mathcal{N}\left(x^{(i)}\mid\mu_j,\Sigma_j\right)}. $$

The ELBO decomposition is:

$$ \mathrm{ELBO}(x;Q,\theta)=\log p(x;\theta)-D_{\mathrm{KL}}\left(Q(z)\parallel p(z\mid x;\theta)\right). $$


## Dimensionality-reduction and PCA notation

| Symbol | Meaning |
|---|---|
| $N$ | number of original features |
| $k$ | number of retained dimensions |
| $U_k$ | matrix whose columns are the first $k$ principal directions |
| $u_j$ | principal direction $j$ |
| $y^{(i)}$ | lower-dimensional representation of example $i$ |
| $\Sigma$ | covariance matrix of centered observations |
| $\lambda_j$ | eigenvalue and variance captured by component $j$ |
| $A=USV^{\top}$ | singular value decomposition |
| $s_j$ | singular value $j$ |
| $r_j$ | explained-variance ratio of component $j$ |
| $R_k$ | cumulative explained variance through component $k$ |

The PCA projection is:

$$ y^{(i)}=U_k^{\top}x^{(i)}. $$

The principal directions satisfy:

$$ \Sigma u_j=\lambda_ju_j. $$

The truncated SVD is:

$$ A_k=U_kS_kV_k^{\top}=\sum_{j=1}^{k}s_ju_jv_j^{\top}. $$

The cumulative explained variance is:

$$ R_k=\sum_{j=1}^{k}r_j. $$


## Neural-network and backpropagation notation

| Symbol | Meaning |
|---|---|
| $z$ | a neuron's net input or pre-activation |
| $a$ | neuron or layer activation |
| $W^{(l)}$ | weight matrix connecting layer $l$ to layer $l+1$ under the source row-vector convention |
| $b^{(l)}$ | bias vector for layer $l$ |
| $z^{(l)}$ | pre-activation vector at layer $l$ |
| $a^{(l)}$ | activation vector at layer $l$ |
| $\sigma^{(l)}$ | activation function at layer $l$ |
| $\delta^{(l)}$ | error signal $\partial J/\partial z^{(l)}$ |
| $\odot$ | elementwise multiplication |
| $\eta$ | learning rate |
| $\alpha$ | negative-side slope of leaky ReLU |

A neuron computes:

$$ z=b+x^{\top}w, \qquad a=\sigma(z). $$

Forward propagation uses:

$$ z^{(l+1)}=a^{(l)}W^{(l)}+b^{(l+1)}, \qquad a^{(l+1)}=\sigma^{(l+1)}\left(z^{(l+1)}\right). $$

A hidden-layer error signal is:

$$ \delta^{(l)}=\delta^{(l+1)}\left(W^{(l)}\right)^{\top}\odot\sigma^{(l)\prime}\left(z^{(l)}\right). $$

The weight gradient is:

$$ \frac{\partial J}{\partial W^{(l)}}=\left(a^{(l)}\right)^{\top}\delta^{(l+1)}. $$
