# Glossary

## Autocorrelation

Correlation between observations of the same process separated by a lag.

- Introduced: Chapter 2
- Source: CSE598MTL.pdf, p. 6
- Related: lag, ACF, PACF, stationarity

## Autoregressive model

A temporal model in which the current observation depends directly on
one or more previous observations.

- Notation: AR $(p)$
- Introduced: Chapter 2
- Source: CSE598MTL.pdf, p. 7

## Backshift operator

An operator that moves a time-series index backward:

```math
Bx_t=x_{t-1}.
```

- Seasonal form: $B^{12}x_t=x_{t-12}$
- Introduced: Chapter 2
- Source: CSE598MTL.pdf, p. 13

## Differencing

A transformation that subtracts a previous observation:

```math
\Delta x_t=x_t-x_{t-1}.
```

It is used in the notes to reduce changing levels or trends before fitting
AR and MA terms.

- Introduced: Chapter 2
- Sources: CSE598MTL.pdf, pp. 9-10

## Exogenous attribute

An external predictor added to a time-series model, analogous to an input
variable in regression.

- Introduced: Chapter 2
- Source: CSE598MTL.pdf, p. 13

## Forecast horizon

The number of future time points over which predictions are produced.

- Illustrated: one-step versus 20-step prediction
- Source: CSE598MTL.pdf, p. 7

## Moving-average model

A model in which the current observation depends on the current
disturbance and a finite number of previous disturbances.

- Notation: MA $(q)$
- Introduced: Chapter 2
- Source: CSE598MTL.pdf, p. 8

## Partial autocorrelation

The remaining relationship between two lagged observations after
adjusting for observations at the intermediate lags.

- Abbreviation: PACF
- Introduced: Chapter 2
- Source: CSE598MTL.pdf, p. 11

## Residual

The difference between an observed value and its model prediction:

```math
r_t=x_t-\hat{x}_t.
```

- Introduced: Chapter 2
- Source: CSE598MTL.pdf, p. 11

## Stationarity

Stability of the statistical properties of the process generating a time
series.

- Strong stationarity: invariance of finite-dimensional distributions
- Weak stationarity: constant mean and variance, with covariance depending
  only on lag
- Introduced: Chapter 2
- Source: CSE598MTL.pdf, p. 5

## White noise

A process described in the notes as having zero mean, constant variance,
and no temporal correlation.

- The notes emphasize that white noise need not be IID or normal.
- Introduced: Chapter 2
- Source: CSE598MTL.pdf, p. 5

## Additive seasonality

A seasonal effect expressed as an additive deviation from the level:

```math
x_t=L_t+S_t+N_t.
```

- Introduced: Chapter 3
- Source: CSE598MTL.pdf, p. 17

## Convolution

A time-invariant linear filtering operation:

```math
y_t=\sum_k\beta_kx_{t-k}.
```

- Introduced: Chapter 3
- Source: CSE598MTL.pdf, p. 14

## EWMA

Exponentially weighted moving average, a recursive smoother:

```math
y_t=\lambda x_t+(1-\lambda)y_{t-1}.
```

- Also called first-order exponential smoothing on the slide
- Introduced: Chapter 3
- Source: CSE598MTL.pdf, p. 14

## Holt's method

A smoothing method that maintains separate level and trend estimates.

- Introduced: Chapter 3
- Source: CSE598MTL.pdf, p. 16

## Holt-Winters method

A smoothing method that maintains level, trend, and seasonal components.
The notes show additive and multiplicative forms.

- Introduced: Chapter 3
- Source: CSE598MTL.pdf, p. 17

## Impulse response

The coefficient sequence of a linear filter, equal to the filter output
when the input is a single unit impulse.

- Introduced: Chapter 3
- Source: CSE598MTL.pdf, p. 14

## LOESS

Locally estimated scatterplot smoothing using locally weighted averages,
linear regression, or higher-order polynomial fits.

- Introduced: Chapter 3
- Source: CSE598MTL.pdf, p. 18

## Multiplicative seasonality

A seasonal effect that scales the level:

```math
x_t=L_tS_tN_t.
```

- Introduced: Chapter 3
- Source: CSE598MTL.pdf, p. 17

## STL

Seasonal and trend decomposition using LOESS.

- Introduced: Chapter 3
- Source: CSE598MTL.pdf, p. 18

## Transfer-function model

A dynamic regression model that combines current or lagged exogenous
inputs with temporally structured residual behavior.

- Introduced: Chapter 3
- Source: CSE598MTL.pdf, p. 19

## Approximation coefficient

A wavelet-transform coefficient representing coarse or low-resolution
signal structure at a selected scale.

- Introduced: Chapter 4
- Sources: CSE598MTL.pdf, pp. 23-28

## Detail coefficient

A wavelet-transform coefficient representing local differences or
higher-resolution structure.

- Introduced: Chapter 4
- Sources: CSE598MTL.pdf, pp. 23-31

## Discrete wavelet transform

A multiresolution representation that decomposes a sampled signal into
approximation and detail coefficients.

- Abbreviation: DWT
- Introduced: Chapter 4
- Source: CSE598MTL.pdf, p. 20

## Hard thresholding

A coefficient rule that retains $w$ when $|w|>\lambda$ and otherwise
sets it to zero.

- Introduced: Chapter 4
- Source: CSE598MTL.pdf, p. 32

## Multiresolution analysis

Representation of the same signal at several temporal resolutions,
combining coarse approximations and progressively finer details.

- Introduced: Chapter 4
- Sources: CSE598MTL.pdf, pp. 20-28

## Soft thresholding

A coefficient-shrinkage rule:

```math
S_\lambda(w)=\mathrm{sgn}(w)\max(|w|-\lambda,0).
```

- Introduced: Chapter 4
- Source: CSE598MTL.pdf, p. 32

## Support

The interval over which a wavelet function is nonzero. Shorter support is
associated on the course page with better temporal localization.

- Introduced: Chapter 4
- Source: CSE598MTL.pdf, p. 29

## Vanishing moment

A wavelet property under which polynomial components up to a specified
degree have zero detail coefficients.

- db $K$ is described as having $K$ vanishing moments.
- Introduced: Chapter 4
- Sources: CSE598MTL.pdf, pp. 29-31

## Wavelet coefficient

The inner product or local correlation between a signal and a wavelet at
a selected location and scale.

- Introduced: Chapter 4
- Source: CSE598MTL.pdf, p. 23

## Bias

The difference between the expected estimator and the target function.
The chapter uses squared bias as one component of expected squared error.

- Introduced: Chapter 5
- Source: CSE598MTL.pdf, p. 38

## Eigenvalue

For PCA, the variance associated with an eigenvector direction of the
covariance matrix.

- Introduced: Chapter 5
- Sources: CSE598MTL.pdf, pp. 34-35

## Eigenvector

A covariance-matrix direction used as a principal-component loading
vector.

- Introduced: Chapter 5
- Source: CSE598MTL.pdf, p. 34

## Lasso

Regression with an $L_1$ coefficient penalty. The notes emphasize
shrinkage and exact coefficient selection.

- Introduced: Chapter 5
- Source: CSE598MTL.pdf, p. 43

## Loading

An element of a principal-component eigenvector describing the
contribution of an original predictor to the component.

- Introduced: Chapter 5
- Source: CSE598MTL.pdf, p. 34

## Mahalanobis distance

A covariance-adjusted distance:

```math
(\mathbf{x}-\boldsymbol{\mu})^\top
\Sigma^{-1}
(\mathbf{x}-\boldsymbol{\mu}).
```

- Introduced: Chapter 5
- Source: CSE598MTL.pdf, p. 37

## Principal component analysis

A linear transformation that finds orthogonal directions of decreasing
predictor variance.

- Abbreviation: PCA
- Introduced: Chapter 5
- Sources: CSE598MTL.pdf, pp. 33-37

## Principal-component score

The projection of an observation onto a principal-component loading
vector:

```math
z_m=\mathbf{v}_m^\top\mathbf{x}.
```

- Introduced: Chapter 5
- Source: CSE598MTL.pdf, p. 34

## Regularization

Addition of a coefficient penalty to a loss function to control model
complexity.

- Introduced: Chapter 5
- Source: CSE598MTL.pdf, p. 40

## Ridge regression

Regression with an $L_2$-squared coefficient penalty.

- Introduced: Chapter 5
- Sources: CSE598MTL.pdf, pp. 40-42

## Scree plot

A plot of explained variance or eigenvalue against component index, used
in the notes to identify an elbow for selecting components.

- Introduced: Chapter 5
- Source: CSE598MTL.pdf, p. 36

## Variance

The variability of an estimator across repeated samples. In the
bias-variance decomposition, model complexity often increases this term.

- Introduced: Chapter 5
- Source: CSE598MTL.pdf, p. 38

## Absorbing state

A state that cannot be left once entered.

- Introduced: Chapter 6
- Source: CSE598MTL.pdf, p. 51

## Baum-Welch algorithm

An expectation-maximization algorithm for estimating HMM initial,
transition, and emission parameters from expected state and transition
counts.

- Introduced: Chapter 6
- Source: CSE598MTL.pdf, p. 50

## Emission distribution

The observation distribution associated with a hidden state in an HMM.

- Introduced: Chapter 6
- Sources: CSE598MTL.pdf, pp. 47 and 49

## Expectation-maximization

An iterative latent-variable estimation method alternating expected
latent assignments and parameter maximization.

- Abbreviation: EM
- Introduced: Chapter 6
- Sources: CSE598MTL.pdf, pp. 52-55

## Forward algorithm

Dynamic-programming recursion for computing the probability of an HMM
observation sequence.

- Introduced: Chapter 6
- Source: CSE598MTL.pdf, p. 48

## Hidden Markov model

A Markov state process whose states are not observed directly; each state
generates observations through an emission distribution.

- Abbreviation: HMM
- Introduced: Chapter 6
- Source: CSE598MTL.pdf, p. 47

## Markov property

The next state is conditionally independent of earlier states given the
current state.

- Introduced: Chapter 6
- Source: CSE598MTL.pdf, p. 45

## Responsibility

The posterior probability that an observation belongs to a mixture
component.

- Introduced: Chapter 6
- Source: CSE598MTL.pdf, p. 53

## Viterbi algorithm

A dynamic-programming method for finding the most probable complete HMM
state sequence.

- Introduced: Chapter 6
- Source: CSE598MTL.pdf, p. 49

## Activation function

A nonlinear transformation applied after a network's linear weighted sum.
Examples in the notes are sigmoid, tanh, and ReLU.

- Introduced: Chapter 7
- Source: CSE598MTL.pdf, p. 56

## Backpropagation

An efficient application of the chain rule for calculating neural-network
parameter derivatives.

- Introduced: Chapter 7
- Source: CSE598MTL.pdf, p. 61

## Cross-entropy

For one-hot multiclass targets, the negative logarithm of the probability
assigned to the true class, summed across observations.

- Introduced: Chapter 7
- Sources: CSE598MTL.pdf, pp. 58 and 60

## Dropout

A training regularizer that randomly omits network nodes during an update.

- Introduced: Chapter 7
- Source: CSE598MTL.pdf, p. 64

## Logit

The pre-output linear value $\mathbf{o}_i$ transformed by sigmoid,
softmax, or another output function.

- Introduced: Chapter 7
- Source: CSE598MTL.pdf, p. 56

## Minibatch

A subset of training instances used for one gradient update.

- Introduced: Chapter 7
- Source: CSE598MTL.pdf, p. 62

## ReLU

Rectified linear unit:

```math
\mathrm{ReLU}(u)=\max(0,u).
```

- Introduced: Chapter 7
- Source: CSE598MTL.pdf, p. 56

## Softmax

A vector output function converting logits into nonnegative values that
sum to one.

- Introduced: Chapter 7
- Source: CSE598MTL.pdf, p. 56

## Stochastic gradient descent

Gradient training that updates parameters using one training instance at
a time in the course's definition.

- Abbreviation: SGD
- Introduced: Chapter 7
- Source: CSE598MTL.pdf, p. 61

## Weight decay

A squared-weight penalty added to a neural-network loss to shrink
parameter magnitudes.

- Introduced: Chapter 7
- Source: CSE598MTL.pdf, p. 64

## Backpropagation through time

Gradient computation through an RNN unrolled across time steps.

- Abbreviation: BPTT
- Introduced: Chapter 8
- Source: CSE598MTL.pdf, p. 70

## Cell state

The LSTM memory vector updated through forget and input gates and passed
to the next time step.

- Introduced: Chapter 8
- Sources: CSE598MTL.pdf, pp. 74-76

## Encoder-decoder

A sequence architecture in which an encoder represents the input sequence
and a decoder generates an output sequence.

- Introduced: Chapter 8
- Source: CSE598MTL.pdf, p. 78

## Gated recurrent unit

A recurrent unit using reset and update gates to combine the previous
hidden state and a candidate state.

- Abbreviation: GRU
- Introduced: Chapter 8
- Source: CSE598MTL.pdf, p. 77

## Hidden state

An internal recurrent representation carrying information from previous
time steps.

- Introduced: Chapter 8
- Sources: CSE598MTL.pdf, pp. 68-73

## Long short-term memory

A gated recurrent unit with separate hidden and cell states and forget,
input, and output gates.

- Abbreviation: LSTM
- Introduced: Chapter 8
- Sources: CSE598MTL.pdf, pp. 74-76

## Recurrent neural network

A neural model whose hidden state is updated using the previous hidden
state and current input.

- Abbreviation: RNN
- Introduced: Chapter 8
- Source: CSE598MTL.pdf, p. 68

## Stateful training

RNN training in which hidden states are carried from one ordered batch to
the corresponding sequence continuation in the next batch.

- Introduced: Chapter 8
- Source: CSE598MTL.pdf, p. 73

## Truncated BPTT

Backpropagation through only a limited number of recurrent time steps.

- Introduced: Chapter 8
- Source: CSE598MTL.pdf, p. 70

## Vanishing gradient

A gradient that becomes very small after repeated multiplication through
many recurrent steps, limiting long-range learning.

- Introduced: Chapter 8
- Source: CSE598MTL.pdf, p. 74

## Causal convolution

A convolution whose output at time $t$ uses only current and earlier
input positions.

- Introduced: Chapter 9
- Source: CSE598MTL.pdf, p. 79

## Dilated convolution

A convolution whose filter taps are separated by a dilation factor $d$.

- Introduced: Chapter 9
- Source: CSE598MTL.pdf, p. 81

## Receptive field

The temporal input range capable of influencing a selected network
output.

- Introduced: Chapter 9
- Sources: CSE598MTL.pdf, pp. 80-82

## Residual block

A neural-network block that adds a transformed branch to an identity or
projected input branch.

- Introduced for TCNs: Chapter 9
- Source: CSE598MTL.pdf, p. 82

## Temporal convolutional network

A one-dimensional fully-convolutional temporal network based on causal
convolution, dilation, and often residual blocks.

- Abbreviation: TCN
- Introduced: Chapter 9
- Sources: CSE598MTL.pdf, pp. 79-82

## Autoencoder

An encoder-bottleneck-decoder model trained to reconstruct its input.

- Introduced: Chapter 10
- Source: CSE598MTL.pdf, p. 85

## Contrastive learning

A representation-learning method that trains similar pairs to be close
and dissimilar pairs to be far apart.

- Introduced: Chapter 10
- Sources: CSE598MTL.pdf, pp. 87-90

## Kernel PCA

A nonlinear PCA method expressed through inner products in an implicit
feature space.

- Introduced: Chapter 10
- Source: CSE598MTL.pdf, p. 84

## Pretext task

A task constructed from unlabeled data to provide a self-supervised
training signal.

- Introduced: Chapter 10
- Source: CSE598MTL.pdf, p. 87

## Projection head

A training-time mapping $z=g(h)$ applied to an encoder representation
before the contrastive loss.

- Introduced: Chapter 10
- Sources: CSE598MTL.pdf, pp. 88-90

## Self-supervised learning

Learning from targets or relationships created from the data rather than
externally supplied labels.

- Introduced: Chapter 10
- Source: CSE598MTL.pdf, p. 87

## Slowness assumption

The temporal assumption that nearby segments tend to have similar
underlying representations.

- Introduced: Chapter 10
- Source: CSE598MTL.pdf, p. 88

## Temperature

A contrastive-softmax parameter controlling how sharply similarities are
converted into probabilities.

- Symbol: $\tau$
- Introduced: Chapter 10
- Source: CSE598MTL.pdf, p. 89

## Temporal representation learning

Mapping a temporal object into a lower-dimensional embedding for later
analysis.

- Abbreviation: TRL
- Introduced: Chapter 10
- Source: CSE598MTL.pdf, p. 83

## Attention

A mechanism that forms a weighted combination of value vectors using
query-key similarities.

- Introduced: Chapter 11
- Source: CSE598MTL.pdf, p. 93

## Cross-attention

Decoder attention using decoder queries and encoder keys and values.

- Introduced: Chapter 11
- Source: CSE598MTL.pdf, p. 97

## Key

A learned vector compared with a query to calculate an attention score.

- Introduced: Chapter 11
- Source: CSE598MTL.pdf, p. 93

## Masked attention

Decoder self-attention that prevents access to future output positions.

- Introduced: Chapter 11
- Source: CSE598MTL.pdf, p. 96

## Multi-head attention

Parallel attention using several learned query, key, and value
transformations, followed by concatenation and output projection.

- Introduced: Chapter 11
- Sources: CSE598MTL.pdf, pp. 94-95

## Positional encoding

A fixed or learned representation of sequence position added to content
embeddings.

- Introduced: Chapter 11
- Source: CSE598MTL.pdf, p. 99

## Query

A learned vector representing what a sequence element seeks from other
elements.

- Introduced: Chapter 11
- Source: CSE598MTL.pdf, p. 93

## Self-attention

Attention in which query, key, and value vectors originate from the same
sequence.

- Introduced: Chapter 11
- Source: CSE598MTL.pdf, p. 93

## Teacher forcing

Training a decoder with the known target sequence rather than its own
previous generated outputs.

- Introduced: Chapter 11
- Source: CSE598MTL.pdf, p. 98

## Transformer

A sequence architecture based on attention, position-wise feed-forward
layers, residual connections, normalization, and positional information.

- Introduced: Chapter 11
- Sources: CSE598MTL.pdf, pp. 92-100

## Value

The learned information vector combined according to attention weights.

- Introduced: Chapter 11
- Source: CSE598MTL.pdf, p. 93

## Local-level model

A model with an observed series around an evolving latent level:

```math
x_t=\mu_t+e_t,
\qquad
\mu_t=\mu_{t-1}+\delta_t.
```

The handwritten appendix studies the first difference of this model.

- Introduced: Chapter 2
- Revisited: Chapter 12
- Sources: CSE598MTL.pdf, pp. 9 and 103
