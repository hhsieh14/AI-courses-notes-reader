# Equation Index

## Stationarity

### Strong stationarity

```math
F(x_{t_1},\ldots,x_{t_n})
=
F(x_{t_1+\tau},\ldots,x_{t_n+\tau})
```

**Source:** CSE598MTL.pdf, p. 5

### Weak-stationarity covariance invariance

```math
\mathrm{Cov}(x_i,x_j)
=
\mathrm{Cov}(x_{i+\tau},x_{j+\tau})
```

**Source:** CSE598MTL.pdf, p. 5

---

## Autocorrelation

### Population autocorrelation

```math
\rho_k =
\frac{\mathrm{Cov}(x_t,x_{t-k})}
{\mathrm{Var}(x_t)}
```

**Source:** CSE598MTL.pdf, p. 6

### Sample autocorrelation

```math
r_k =
\frac{
\sum_{t=1}^{n-k}(x_t-\bar{x})(x_{t-k}-\bar{x})
}{
\sum_{t=1}^{n}(x_t-\bar{x})^2
}
```

**Source:** CSE598MTL.pdf, p. 6

---

## Ideal stirred tank

### Time constant

```math
T=\frac{V}{f}
```

### Step response

```math
x_t=w_0(1-e^{-t/T})
```

### Sampled recurrence

```math
x_t=bw_t+(1-b)x_{t-1},
\qquad
b=1-e^{-\Delta t/T}
```

### Lag-one correlation

```math
\rho=e^{-\Delta t/T}
```

**Source:** CSE598MTL.pdf, p. 6

---

## Autoregressive models

### AR(1)

```math
x_t=c+\phi x_{t-1}+e_t
```

### AR(1) mean

```math
E[x_t]=\frac{c}{1-\phi}
```

### AR(1) standard deviation

```math
\mathrm{SD}(x_t)=
\frac{\sigma}{\sqrt{1-\phi^2}}
```

### AR(1) autocorrelation

```math
\rho_k=\phi^k
```

### AR(p)

```math
x_t=
c+\phi_1x_{t-1}+\cdots+\phi_px_{t-p}+e_t
```

**Source:** CSE598MTL.pdf, p. 7

---

## Moving-average models

### MA(1)

```math
x_t=\mu+e_t-\theta e_{t-1}
```

### MA(1) variance

```math
\mathrm{Var}(x_t)=\sigma^2(1+\theta^2)
```

### MA(1) autocorrelation

```math
\rho_1=\frac{-\theta}{1+\theta^2},
\qquad
\rho_k=0 \text{ for } k>1
```

### MA(q)

```math
x_t=
\mu+e_t-\theta_1e_{t-1}-\cdots-\theta_qe_{t-q}
```

**Source:** CSE598MTL.pdf, p. 8

---

## Integrated models and differencing

### Evolving mean

```math
x_t=\mu_t+e_t,
\qquad
\mu_t=\mu_{t-1}+\delta_t
```

### IMA(1,1)

```math
x_t=x_{t-1}+\epsilon_t-\theta\epsilon_{t-1}
```

### First difference

```math
\Delta x_t=x_t-x_{t-1}
```

**Sources:** CSE598MTL.pdf, pp. 9-10

---

## ARIMA and residuals

### ARIMA order

```math
\mathrm{ARIMA}(p,d,q)
```

### Residual

```math
r_t=x_t-\hat{x}_t
```

**Source:** CSE598MTL.pdf, p. 11

---

## Seasonal operations

### Seasonal backshift

```math
B^{12}x_t=x_{t-12}
```

### Seasonal difference

```math
(1-B^{12})x_t
```

### Combined regular and seasonal difference

```math
(1-B)(1-B^{12})x_t
```

**Source:** CSE598MTL.pdf, p. 13

---

## Temporal convolutional networks

### General temporal convolution

```math
F(t)
=
(x*f)(t)
=
\sum_{i=-K}^{K}
f(i)x_{t+i}
```

### Causal convolution

```math
F(t)
=
(x*f)(t)
=
\sum_{i=0}^{K}
f(i)x_{t-i}
```

**Source:** CSE598MTL.pdf, p. 79

### Ordinary receptive-field formula printed on the slide

```math
R=L+K-1
```

**Source:** CSE598MTL.pdf, p. 80

> Review note: the page's layer-by-layer counting suggests
> $R=1+L(K-1)$ for stride-one width $K$ layers.

### Dilated causal convolution

```math
F(t)
=
(x*_d f)(t)
=
\sum_{i=0}^{K}
f(i)x_{t-di}
```

**Source:** CSE598MTL.pdf, p. 81

### Residual-block form

```math
\mathbf{z}
=
\mathcal{F}(\mathbf{x})+\mathcal{I}(\mathbf{x})
```

where $\mathcal{I}$ is an identity map or a $1\times1$ projection.

**Source:** CSE598MTL.pdf, p. 82

---

## Kernel PCA

### Kernel inner product

```math
K(\mathbf{x}_i,\mathbf{x}_j)
=
\phi(\mathbf{x}_i)^\top\phi(\mathbf{x}_j)
```

### Polynomial kernel

```math
K(\mathbf{x}_i,\mathbf{x}_j)
=
(1+\mathbf{x}_i^\top\mathbf{x}_j)^d
```

### Gaussian/RBF kernel

```math
K(\mathbf{x}_i,\mathbf{x}_j)
=
\exp
\left(
-\gamma
\|\mathbf{x}_i-\mathbf{x}_j\|^2
\right)
```

### Hyperbolic-tangent kernel

```math
K(\mathbf{x}_i,\mathbf{x}_j)
=
\tanh
\left(
\beta\mathbf{x}_i^\top\mathbf{x}_j-\delta
\right)
```

**Source:** CSE598MTL.pdf, p. 84

---

## Autoencoders

### Reconstruction loss

```math
L
=
\sum_{i=1}^{N}
\|
\mathbf{x}_i-\hat{\mathbf{x}}_i
\|_2^2
```

### Encoder

```math
\mathbf{h}_i
=
\phi(W_1\mathbf{x}_i+\mathbf{b}_1)
```

### Decoder

```math
\hat{\mathbf{x}}_i
=
g(W_2\mathbf{h}_i+\mathbf{b}_2)
```

**Source:** CSE598MTL.pdf, p. 85

---

## Contrastive learning

### Cosine similarity

```math
\mathrm{sim}(\mathbf{u},\mathbf{v})
=
\frac{\mathbf{u}^\top\mathbf{v}}
{\|\mathbf{u}\|\|\mathbf{v}\|}
```

### Softmax-style contrastive loss

```math
\ell_i
=
-\log
\frac{
\exp(\mathrm{sim}(\mathbf{z}_i,\mathbf{z}_j)/\tau)
}{
\sum_k
\mathbf{1}_{[k\neq i]}
\exp(\mathrm{sim}(\mathbf{z}_i,\mathbf{z}_k)/\tau)
}
```

**Source:** CSE598MTL.pdf, p. 89

### Margin triplet loss

```math
\max
(
\mathbf{u}^\top\mathbf{v}^{-}
-
\mathbf{u}^\top\mathbf{v}^{+}
+
m,
0
)
```

**Source:** CSE598MTL.pdf, p. 90

---

## Mutual information

```math
I(X;Y)
=
H(X)-H(X\mid Y)
```

```math
I(X;Y)
=
H(Y)-H(Y\mid X)
```

```math
I(X;Y)
=
H(X)+H(Y)-H(X,Y)
```

```math
H(X)
=
-\sum_xp(x)\log p(x)
```

**Source:** CSE598MTL.pdf, p. 91

---

## Transformer attention

### Query, key, and value

```math
\mathbf{q}_i=W_Q\mathbf{x}_i
```

```math
\mathbf{k}_i=W_K\mathbf{x}_i
```

```math
\mathbf{v}_i=W_V\mathbf{x}_i
```

### Scaled dot-product similarity

```math
s_{ij}
=
\frac{
\mathbf{q}_i^\top\mathbf{k}_j
}{
\sqrt{d_k}
}
```

### Attention weights

```math
w_{ij}
=
\frac{
\exp(s_{ij})
}{
\sum_{\ell=1}^{T}\exp(s_{i\ell})
}
```

### Attention output

```math
\mathbf{z}_i
=
\sum_{j=1}^{T}
w_{ij}\mathbf{v}_j
```

**Source:** CSE598MTL.pdf, p. 93

### Per-head output

```math
\boldsymbol{\zeta}_i^{(h)}
=
\sum_{j=1}^{T}
w_{ij}^{(h)}
\mathbf{v}_j^{(h)}
```

### Multi-head concatenation and projection

```math
\boldsymbol{\zeta}_i
=
\mathrm{Concat}
(
\boldsymbol{\zeta}_i^{(1)},
\ldots,
\boldsymbol{\zeta}_i^{(H)}
)
```

```math
\mathbf{z}_i
=
W_O\boldsymbol{\zeta}_i
```

**Source:** CSE598MTL.pdf, p. 94

---

## Transformer feed-forward and residual layers

### Position-wise feed-forward network

```math
\mathrm{FFN}(\mathbf{x})
=
W_2
\mathrm{ReLU}
(
W_1\mathbf{x}+\mathbf{b}_1
)
+
\mathbf{b}_2
```

**Source:** CSE598MTL.pdf, p. 95

### Residual normalization

```math
\mathrm{LayerNorm}
(
\mathbf{x}
+
\mathrm{Sublayer}(\mathbf{x})
)
```

**Source:** CSE598MTL.pdf, p. 99

---

## Decoder output

```math
\mathbf{o}_j
=
W_O\mathbf{r}_j+\mathbf{b}_O
```

```math
\hat{y}_j
=
\mathrm{arg\,max}
[
\mathrm{softmax}(\mathbf{o}_{j-1})
]
```

**Source:** CSE598MTL.pdf, p. 97

---

## Handwritten local-level appendix

### Local-level model

```math
x_t=\mu_t+e_t
```

```math
\mu_t=\mu_{t-1}+\delta_t
```

### First difference

```math
d_t=x_t-x_{t-1}
=
\delta_t+e_t-e_{t-1}
```

**Sources:** CSE598MTL.pdf, pp. 9 and 103

### Variance decomposition written on the page

```math
\mathrm{Var}(d_t)
=
\mathrm{Var}(\delta_t)
+
\mathrm{Var}(e_t-e_{t-1})
+
2\mathrm{Cov}
(\delta_t,e_t-e_{t-1})
```

**Source:** CSE598MTL.pdf, p. 103

### Added consistency check under independent white-noise assumptions

```math
\mathrm{Var}(d_t)
=
\sigma_\delta^2+2\sigma_e^2
```

```math
\mathrm{Cov}(d_t,d_{t+1})
=
-\sigma_e^2
```

```math
\mathrm{Cov}(d_t,d_{t+k})
=
0,
\qquad k\geq2
```

```math
\rho_1
=
-
\frac{
\sigma_e^2
}{
\sigma_\delta^2+2\sigma_e^2
}
```

**Status:** Added algebraic consistency check based on the model assumptions visible on pages 9 and 103
