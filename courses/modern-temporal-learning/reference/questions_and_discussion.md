# Self-Test Questions

Questions for checking your understanding, chapter by chapter. Try each one before opening the answer. Every chapter also ends with its own practical Q&A; these questions are more conceptual, and several are common interview questions.

## 1. Temporal data

<details><summary>Why is random K-fold cross-validation wrong for most forecasting problems?</summary>

It trains on the future to predict the past, and overlapping windows put near-duplicate rows on both sides of the split. The error estimate comes out optimistic. Use expanding or sliding windows with a gap ([Ch 8, §2](../notes/08_rnn_lstm_gru_and_seq2seq.md)).
</details>

<details><summary>Give an example of a feature that leaks the future.</summary>

A centered moving average (it uses $x_{t+1},\ldots$), a series standardized with statistics from the whole dataset, or a "days until failure" field. Each is fine for analysis but not available at prediction time.
</details>

<details><summary>Why can Euclidean distance mislead when comparing two sequences?</summary>

A small time shift of the same shape produces a large pointwise distance. Dynamic time warping aligns the sequences first. Distances on features or learned embeddings compare shapes rather than positions.
</details>

## 2. Stationarity and ARIMA

<details><summary>A series has an ACF that decays very slowly and stays high out to large lags. What does that suggest?</summary>

Non-stationarity: a trend or unit root. Difference it (or detrend it) and look at the ACF again before choosing AR/MA orders.
</details>

<details><summary>The PACF cuts off after lag 2 and the ACF tails off. Which model?</summary>

AR(2).
</details>

<details><summary>Why does the lag-1 ACF of an over-differenced series tend toward −0.5?</summary>

Differencing white noise gives $e_t-e_{t-1}$, whose lag-1 autocorrelation is $-\sigma^2/(2\sigma^2)=-0.5$. Differencing a series that was already stationary injects exactly this structure.
</details>

<details><summary>Residuals pass the Ljung–Box test, but the model is still biased. How is that possible?</summary>

Ljung–Box checks for autocorrelation, not for the mean. A constant offset has no autocorrelation. Check the residual mean and plot separately.
</details>

<details><summary>What does the "I" in ARIMA do, and why is it separate from AR?</summary>

It differences the series $d$ times before fitting ARMA. That's equivalent to an AR polynomial with a unit root, $(1-B)^d$, factored out because that root makes the process non-stationary.
</details>

## 3. Filters and decomposition

<details><summary>What is the effective memory of an EWMA with weight λ on the newest point?</summary>

The weights $\lambda(1-\lambda)^k$ have mean lag $(1-\lambda)/\lambda$. For example, $\lambda=0.1$ averages over roughly the last 10 points.
</details>

<details><summary>Why does a simple EWMA lag behind a linear trend, and what fixes it?</summary>

It estimates a level, so on a slope $\beta_1$ it trails by $\beta_1(1-\lambda)/\lambda$. Holt's method adds a trend state; double exponential smoothing corrects the bias.
</details>

<details><summary>When is EWMA the optimal forecaster?</summary>

When the series is IMA(1,1), e.g. a random-walk level plus noise, with $\lambda=1-\theta$. See the derivation in [Ch 12](../notes/12_handwritten_appendix.md).
</details>

<details><summary>Additive or multiplicative seasonality: how do you tell?</summary>

If the seasonal swing grows with the level, it's multiplicative. Taking logs turns it additive.
</details>

## 4. Fourier and wavelets

<details><summary>What can wavelets tell you that the Fourier transform can't?</summary>

*When* a frequency is present. Fourier basis functions extend over all time, so a short burst is spread across the whole spectrum. Wavelets are localized in time and scale.
</details>

<details><summary>Why do wavelets with more vanishing moments make changes easier to spot?</summary>

A wavelet with $p$ vanishing moments is orthogonal to polynomials of degree $<p$, so smooth stretches give near-zero detail coefficients and only breaks produce large ones.
</details>

<details><summary>Why does the pyramid algorithm halve the length at each level?</summary>

Each level splits the signal into a low-pass (approximation) half-band and a high-pass (detail) half-band. Each half needs only half the samples (downsampling by 2), so the total number of coefficients stays equal to the signal length.
</details>

## 5. PCA and regularization

<details><summary>Why standardize before PCA?</summary>

PCA maximizes variance, so a variable measured in larger units dominates the first component. Standardizing (PCA on the correlation matrix) puts variables on an equal footing, unless the units are already comparable and their scale is meaningful.
</details>

<details><summary>What are Hotelling's T² and the Q (SPE) statistic, and why use both?</summary>

$T^2$ measures unusual variation *inside* the PC subspace (a Mahalanobis distance on the scores). $Q$ measures the residual distance *off* the subspace. A new kind of fault often shows up only in $Q$.
</details>

<details><summary>Why does lasso give exact zeros while ridge doesn't?</summary>

The L1 ball has corners on the axes, so the loss contours usually first touch it at a corner. The L2 ball is round, so the solution only shrinks toward zero. In an orthonormal basis lasso is soft thresholding, which sets small coefficients exactly to 0.
</details>

<details><summary>Which directions does ridge shrink most?</summary>

Those with small singular values. The fitted value along direction $m$ is scaled by $d_m^2/(d_m^2+\lambda)$, so low-variance directions, where estimates are noisiest, get shrunk the most.
</details>

## 6. Markov models, HMMs and EM

<details><summary>Why is the forward algorithm O(Q²T) instead of O(Q^T)?</summary>

$\alpha_t(j)$ summarizes every path that ends in $j$ at time $t$. The Markov property means the future only needs that summary, so each step combines $Q$ states into $Q$ states.
</details>

<details><summary>When can the sequence of per-step most-likely states be an impossible path?</summary>

Whenever it uses a transition with $a_{ij}=0$. Per-step decoding optimizes each position separately; Viterbi optimizes the path as a whole.
</details>

<details><summary>Does EM always increase the likelihood?</summary>

It never decreases it. It may stay flat at convergence, and it finds a local optimum only.
</details>

<details><summary>How does a GMM with shared spherical covariance relate to K-means?</summary>

As the shared variance goes to 0, the responsibilities become hard 0/1 assignments to the nearest mean, and EM becomes K-means.
</details>

## 7. Neural-network foundations

<details><summary>Show that minimizing cross-entropy is maximum likelihood.</summary>

With one-hot $\mathbf y_i$, $P(\mathbf y_i)=\prod_kp_{ik}^{y_{ik}}$, so $-\log L=-\sum_i\sum_ky_{ik}\log p_{ik}$, which is the cross-entropy.
</details>

<details><summary>Why can't all weights be initialized to the same value?</summary>

All hidden units would compute the same output and get the same gradient, and they would stay identical forever. Random initialization breaks the symmetry.
</details>

<details><summary>How many parameters are in a 70→20→20→10→4 network?</summary>

$1420+420+210+44=2094$ (weights plus biases).
</details>

## 8. RNNs, LSTMs, GRUs

<details><summary>Why do RNN gradients vanish or explode?</summary>

The gradient from step $k$ to step $t$ is a product of $t-k$ Jacobians $\mathrm{diag}(\phi')U$. Its size scales like $(\gamma\|U\|)^{t-k}$, which shrinks or grows geometrically.
</details>

<details><summary>Which part of the LSTM fixes vanishing gradients, and how?</summary>

The additive cell update $\mathbf c_t=\mathbf f_t\odot\mathbf c_{t-1}+\mathbf i_t\odot\tilde{\mathbf c}_t$. Along it, $\partial\mathbf c_t/\partial\mathbf c_{t-1}\approx\mathrm{diag}(\mathbf f_t)$, with no weight matrix and no squashing, so with $f\approx1$ the gradient passes through largely unchanged.
</details>

<details><summary>What does "stateful" change, and what does it not change?</summary>

It changes where each batch's initial hidden state comes from (the previous batch instead of zeros). It doesn't change the weights or how they're updated. It does require unshuffled, aligned batches and detaching the state between batches.
</details>

<details><summary>What problem did attention originally solve?</summary>

The encoder–decoder bottleneck: squeezing a whole input sequence into one fixed vector. Attention lets each decoder step build its own weighted summary of all encoder states.
</details>

## 9. TCNs

<details><summary>What's the receptive field of a TCN with kernel size 3 and dilations 1, 2, 4, 8 (one conv per layer)?</summary>

$1+(3-1)(1+2+4+8)=31$.
</details>

<details><summary>How does dilation differ from stride?</summary>

Dilation spaces out the filter's *inputs* and keeps an output at every time step. Stride skips *outputs* and shortens the sequence.
</details>

<details><summary>Why is a TCN faster to train than an RNN on long sequences?</summary>

All time positions in a layer are computed in parallel. An RNN has to finish step $t-1$ before starting step $t$.
</details>

## 10. Representation learning

<details><summary>Why can't the Gaussian-kernel feature map be written out?</summary>

Its expansion contains monomials of every degree, so the feature space is infinite-dimensional. Only the kernel (inner product) can be computed.
</details>

<details><summary>Is a linear autoencoder the same as PCA?</summary>

It spans the same optimal subspace, but its code is an arbitrary invertible transform of the PC scores, not the orthonormal, variance-ordered components.
</details>

<details><summary>Why is the projection head thrown away after contrastive pretraining?</summary>

The loss makes the head's output invariant to the augmentations, which throws information away. The encoder output before the head keeps more, and probes on it do better.
</details>

<details><summary>What does lowering the temperature do in NT-Xent?</summary>

It sharpens the softmax, so the loss concentrates on the hardest negatives. That gives stronger separation, but false negatives (true positives treated as negatives) are punished harder.
</details>

<details><summary>In the real-vs-artificial anomaly detector, which probability is the anomaly score?</summary>

The probability of the *artificial* class. A high probability of "real" means the series looks normal.
</details>

## 11. Transformers

<details><summary>Why scale dot products by √d_k?</summary>

With unit-variance components, $\mathbf q^\top\mathbf k$ has variance $d_k$. Unscaled scores saturate the softmax and kill its gradients. Scaling restores unit variance.
</details>

<details><summary>Is self-attention aware of order?</summary>

No. It's permutation-equivariant. Order comes only from positional (or timestamp) encodings added to the inputs.
</details>

<details><summary>Why can the decoder be trained in parallel but not run in parallel?</summary>

During training the whole true target sequence is known (teacher forcing) and the causal mask hides the future. At inference each output depends on the previous generated one.
</details>

<details><summary>What's the memory cost of full attention on a 10,000-step series, and two ways to reduce it?</summary>

$10^8$ scores per head per layer. Patching (shorter token sequence) and sparse or low-rank attention reduce it.
</details>

## 12. Appendix

<details><summary>For x_t = μ_t + e_t with a random-walk μ_t, what is the lag-1 autocorrelation of the first difference?</summary>

$\rho_1=-\sigma_e^2/(\sigma_\delta^2+2\sigma_e^2)$, and $\rho_k=0$ for $k\ge2$, i.e. MA(1).
</details>
