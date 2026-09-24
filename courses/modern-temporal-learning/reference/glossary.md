# Glossary

Terms in alphabetical order, each with a one-line definition and the chapter where it's explained.

| Term | Meaning | Ch |
|---|---|:---:|
| **Absorbing state** | a Markov state that is never left once entered ($a_{ii}=1$) | [6](../notes/06_markov_models_hmm_and_em.md) |
| **ACF / autocorrelation** | correlation of a series with itself at lag $k$: $\rho_k=\gamma(k)/\gamma(0)$ | [2](../notes/02_classical_time_series_models.md) |
| **Activation function** | the nonlinearity between layers (sigmoid, tanh, ReLU); without it, stacked layers collapse to one linear map | [7](../notes/07_neural_network_foundations.md) |
| **Additive seasonality** | a seasonal swing of constant size, added to the level: $x_t=L_t+S_t+N_t$ | [3](../notes/03_filters_smoothing_and_decomposition.md) |
| **AIC / BIC** | $-2\log L$ plus a complexity penalty ($2p$ or $p\log T$); lower is better | [3](../notes/03_filters_smoothing_and_decomposition.md) |
| **Approximation coefficient** | the low-pass (smooth) output of a wavelet level | [4](../notes/04_wavelets.md) |
| **AR($p$) model** | $x_t$ regressed on its own $p$ previous values plus a shock | [2](../notes/02_classical_time_series_models.md) |
| **ARIMA($p,d,q$)** | AR($p$) + MA($q$) fitted after $d$ differences | [2](../notes/02_classical_time_series_models.md) |
| **Attention** | a content-dependent weighted average of value vectors, with weights from query–key similarity | [8](../notes/08_rnn_lstm_gru_and_seq2seq.md), [11](../notes/11_transformers.md) |
| **Autoencoder** | a network trained to reconstruct its input through a narrow bottleneck; the bottleneck is the representation | [10](../notes/10_representation_learning.md) |
| **Backpropagation** | the chain rule applied backward through a network to get every parameter's gradient efficiently | [7](../notes/07_neural_network_foundations.md) |
| **BPTT** | backpropagation through the time-unrolled RNN, summing gradients for the shared weights | [8](../notes/08_rnn_lstm_gru_and_seq2seq.md) |
| **Backshift operator** | $Bx_t=x_{t-1}$; $B^{s}x_t=x_{t-s}$ | [2](../notes/02_classical_time_series_models.md) |
| **Baum–Welch** | EM for HMMs: forward–backward expected counts (E-step), then ratio re-estimates (M-step) | [6](../notes/06_markov_models_hmm_and_em.md) |
| **BERT** | encoder-only transformer pretrained by masked-token prediction; bidirectional context | [11](../notes/11_transformers.md) |
| **Bias (statistical)** | $\mathbb E[\hat f]-f$: systematic error of an estimator | [5](../notes/05_pca_and_regularization.md) |
| **Causal convolution** | a convolution whose output at $t$ uses only inputs at $t$ and earlier | [9](../notes/09_temporal_convolutional_networks.md) |
| **Cell state** | the LSTM's additively updated memory line $\mathbf c_t$ | [8](../notes/08_rnn_lstm_gru_and_seq2seq.md) |
| **Contrastive learning** | learning embeddings where positive pairs are close and negatives are far apart | [10](../notes/10_representation_learning.md) |
| **Convolution (filter)** | $y_t=\sum_k\beta_kx_{t-k}$: a weighted sum over a window, with the same weights at every position | [3](../notes/03_filters_smoothing_and_decomposition.md), [9](../notes/09_temporal_convolutional_networks.md) |
| **Cross-attention** | decoder queries attending to encoder keys and values | [11](../notes/11_transformers.md) |
| **Cross-entropy** | $-\sum_ky_k\log p_k$; the negative log-likelihood of a categorical model | [7](../notes/07_neural_network_foundations.md) |
| **Detail coefficient** | the high-pass output of a wavelet level: local change at that scale | [4](../notes/04_wavelets.md) |
| **Differencing** | $\Delta x_t=x_t-x_{t-1}$; removes a random-walk level or a linear trend | [2](../notes/02_classical_time_series_models.md) |
| **Dilation** | spacing of $d$ steps between a convolution's taps | [9](../notes/09_temporal_convolutional_networks.md) |
| **Discrete wavelet transform** | repeated low/high-pass filtering and downsampling (the pyramid algorithm) | [4](../notes/04_wavelets.md) |
| **Dropout** | randomly zeroing units during training to prevent co-adaptation | [7](../notes/07_neural_network_foundations.md) |
| **Early stopping** | stopping training at the minimum of the validation loss | [7](../notes/07_neural_network_foundations.md) |
| **Eigenvalue / eigenvector** | $\Sigma\mathbf v=\lambda\mathbf v$; in PCA, the variance along a direction and the direction itself | [5](../notes/05_pca_and_regularization.md) |
| **EM** | expectation–maximization: alternate expected latent statistics and parameter re-estimation; the likelihood never decreases | [6](../notes/06_markov_models_hmm_and_em.md) |
| **Emission distribution** | $F_j(y)=P(y_t\mid x_t=j)$ in an HMM | [6](../notes/06_markov_models_hmm_and_em.md) |
| **Encoder–decoder** | an encoder summarizes the input; a decoder generates the output sequence from it | [8](../notes/08_rnn_lstm_gru_and_seq2seq.md) |
| **EWMA** | exponentially weighted moving average $y_t=\lambda x_t+(1-\lambda)y_{t-1}$ | [3](../notes/03_filters_smoothing_and_decomposition.md) |
| **Exogenous variable** | an external input to a time-series model (the X in SARIMAX) | [2](../notes/02_classical_time_series_models.md) |
| **Exposure bias** | the train–test mismatch from teacher forcing: the model never trains on its own mistakes | [8](../notes/08_rnn_lstm_gru_and_seq2seq.md), [11](../notes/11_transformers.md) |
| **Forecast horizon** | how many steps ahead we predict, $H$ | [2](../notes/02_classical_time_series_models.md), [8](../notes/08_rnn_lstm_gru_and_seq2seq.md) |
| **Forget gate** | LSTM gate that scales the old cell state; 1 = keep, 0 = forget | [8](../notes/08_rnn_lstm_gru_and_seq2seq.md) |
| **Forward algorithm** | computes $P(\mathbf y)$ for an HMM in $O(Q^2T)$ by summing over paths recursively | [6](../notes/06_markov_models_hmm_and_em.md) |
| **GRU** | gated recurrent unit: reset and update gates, one state; a lighter LSTM | [8](../notes/08_rnn_lstm_gru_and_seq2seq.md) |
| **Hard thresholding** | keep a coefficient if $\lvert w\rvert>\lambda$, otherwise set it to 0 | [4](../notes/04_wavelets.md) |
| **Hidden Markov model** | a hidden Markov chain of states, each emitting an observation | [6](../notes/06_markov_models_hmm_and_em.md) |
| **Hidden state (RNN)** | the per-step activation $\mathbf h_t$ that summarizes the past; not a weight | [8](../notes/08_rnn_lstm_gru_and_seq2seq.md) |
| **Holt / Holt–Winters** | exponential smoothing with a trend state / with trend and seasonal states | [3](../notes/03_filters_smoothing_and_decomposition.md) |
| **Hotelling's $T^2$** | a Mahalanobis distance on PC scores, $\sum_mz_m^2/\lambda_m$ | [5](../notes/05_pca_and_regularization.md) |
| **IID** | independent and identically distributed | [2](../notes/02_classical_time_series_models.md) |
| **IMA(1,1)** | a series whose first difference is MA(1); EWMA is its optimal forecaster | [2](../notes/02_classical_time_series_models.md), [12](../notes/12_handwritten_appendix.md) |
| **Impulse response** | a filter's output to a single unit spike; its weights $\beta_k$ | [3](../notes/03_filters_smoothing_and_decomposition.md) |
| **InfoNCE / NT-Xent** | the softmax contrastive loss: pick the positive among $N$ candidates | [10](../notes/10_representation_learning.md) |
| **Kernel PCA** | PCA in an implicit feature space, computed from the kernel (Gram) matrix | [10](../notes/10_representation_learning.md) |
| **Key / query / value** | attention's projections: what a position offers / what it seeks / what it hands over | [11](../notes/11_transformers.md) |
| **KV cache** | stored keys and values of past positions, reused during autoregressive decoding | [11](../notes/11_transformers.md) |
| **Lasso** | least squares with an $\ell_1$ penalty; shrinks and sets coefficients exactly to 0 | [5](../notes/05_pca_and_regularization.md) |
| **Layer normalization** | normalizing each position's feature vector to zero mean and unit variance | [11](../notes/11_transformers.md) |
| **Ljung–Box test** | a joint test that residual autocorrelations up to lag $h$ are all zero | [2](../notes/02_classical_time_series_models.md) |
| **Loading** | the weight of an original variable in a principal component | [5](../notes/05_pca_and_regularization.md) |
| **Local-level model** | $x_t=\mu_t+e_t$, $\mu_t=\mu_{t-1}+\delta_t$: a random-walk level plus noise | [2](../notes/02_classical_time_series_models.md), [12](../notes/12_handwritten_appendix.md) |
| **LOESS** | locally weighted regression smoothing | [3](../notes/03_filters_smoothing_and_decomposition.md) |
| **Logit** | the unnormalized output $\mathbf o$ before sigmoid or softmax | [7](../notes/07_neural_network_foundations.md) |
| **LSTM** | long short-term memory: forget, input and output gates around an additive cell state | [8](../notes/08_rnn_lstm_gru_and_seq2seq.md) |
| **MA($q$) model** | $x_t$ as the current shock plus $q$ past shocks | [2](../notes/02_classical_time_series_models.md) |
| **Mahalanobis distance** | distance scaled by the covariance: $(\mathbf x-\boldsymbol\mu)^\top\Sigma^{-1}(\mathbf x-\boldsymbol\mu)$ | [5](../notes/05_pca_and_regularization.md) |
| **Markov property** | the future depends on the past only through the present state | [6](../notes/06_markov_models_hmm_and_em.md) |
| **Masked attention** | attention with future positions' scores set to $-\infty$ | [11](../notes/11_transformers.md) |
| **Minibatch** | the subset of examples used for one gradient step | [7](../notes/07_neural_network_foundations.md) |
| **Momentum** | an EWMA of gradients used as the update direction | [7](../notes/07_neural_network_foundations.md) |
| **Multi-head attention** | several attention heads in parallel in lower-dimensional subspaces, concatenated and projected by $W_O$ | [11](../notes/11_transformers.md) |
| **Multiplicative seasonality** | a seasonal swing proportional to the level: $x_t=L_tS_tN_t$ | [3](../notes/03_filters_smoothing_and_decomposition.md) |
| **Multiresolution analysis** | describing a signal at a nested set of scales | [4](../notes/04_wavelets.md) |
| **Mutual information** | $I(X;Y)=H(X)-H(X\mid Y)$: the reduction in uncertainty about one variable from knowing the other | [10](../notes/10_representation_learning.md) |
| **PACF** | correlation at lag $k$ after removing the effect of the intermediate lags | [2](../notes/02_classical_time_series_models.md) |
| **Positional encoding** | a vector added to embeddings to tell attention where each element is | [11](../notes/11_transformers.md) |
| **Pretext task** | a label-free task invented from the data to train a representation | [10](../notes/10_representation_learning.md) |
| **Principal component** | a direction of maximal variance (a covariance eigenvector); its projections are the scores | [5](../notes/05_pca_and_regularization.md) |
| **Projection head** | a small MLP $g$ on top of the encoder where the contrastive loss is applied; discarded afterwards | [10](../notes/10_representation_learning.md) |
| **Q statistic (SPE)** | squared residual distance from the PCA subspace | [5](../notes/05_pca_and_regularization.md) |
| **Receptive field** | how many past inputs can influence one output | [9](../notes/09_temporal_convolutional_networks.md) |
| **Recurrent neural network** | a network that updates a hidden state with shared weights at each time step | [8](../notes/08_rnn_lstm_gru_and_seq2seq.md) |
| **Regularization** | constraining a model (penalty, early stopping, dropout) to reduce variance | [5](../notes/05_pca_and_regularization.md), [7](../notes/07_neural_network_foundations.md) |
| **ReLU** | $\max(0,u)$ | [7](../notes/07_neural_network_foundations.md) |
| **Residual** | $r_t=x_t-\hat x_t$; should look like white noise for a good model | [2](../notes/02_classical_time_series_models.md) |
| **Residual connection** | $\mathbf y=\mathbf x+\mathcal F(\mathbf x)$; skips layers, not time | [9](../notes/09_temporal_convolutional_networks.md), [11](../notes/11_transformers.md) |
| **Responsibility** | $\gamma(z_{nk})$: the posterior probability that component $k$ generated point $n$ | [6](../notes/06_markov_models_hmm_and_em.md) |
| **Ridge regression** | least squares with an $\ell_2$ penalty; shrinks low-variance directions most | [5](../notes/05_pca_and_regularization.md) |
| **SARIMAX** | ARIMA with seasonal terms and exogenous inputs | [2](../notes/02_classical_time_series_models.md) |
| **Scree plot** | eigenvalues against component index, used to choose how many PCs to keep | [5](../notes/05_pca_and_regularization.md) |
| **Self-attention** | attention where queries, keys and values all come from the same sequence | [11](../notes/11_transformers.md) |
| **Self-supervised learning** | supervision generated from the data itself | [10](../notes/10_representation_learning.md) |
| **Slowness assumption** | nearby time segments share the same underlying state | [10](../notes/10_representation_learning.md) |
| **Soft thresholding** | $\mathrm{sgn}(w)(\lvert w\rvert-\lambda)_+$; the lasso solution in an orthonormal basis | [4](../notes/04_wavelets.md), [5](../notes/05_pca_and_regularization.md) |
| **Softmax** | $e^{o_k}/\sum_je^{o_j}$: turns logits into a probability vector | [7](../notes/07_neural_network_foundations.md) |
| **Stateful training** | carrying an RNN's hidden state across batches (aligned, unshuffled, detached) | [8](../notes/08_rnn_lstm_gru_and_seq2seq.md) |
| **Stationarity (strict / weak)** | the whole distribution / the first two moments are invariant to time shifts | [2](../notes/02_classical_time_series_models.md), [12](../notes/12_handwritten_appendix.md) |
| **STL** | seasonal–trend decomposition using LOESS | [3](../notes/03_filters_smoothing_and_decomposition.md) |
| **Stochastic gradient descent** | updating with the gradient of one example or a minibatch | [7](../notes/07_neural_network_foundations.md) |
| **Support (of a wavelet)** | the interval where the wavelet is nonzero; shorter support localizes better | [4](../notes/04_wavelets.md) |
| **Teacher forcing** | training a decoder on the true previous outputs rather than its own | [8](../notes/08_rnn_lstm_gru_and_seq2seq.md), [11](../notes/11_transformers.md) |
| **Temperature** | $\tau$ dividing the similarities in a softmax; small = sharp, large = flat | [10](../notes/10_representation_learning.md) |
| **Temporal convolutional network** | a stack of causal, dilated, residual 1-D convolutions | [9](../notes/09_temporal_convolutional_networks.md) |
| **Transfer-function model** | a regression of $y_t$ on lagged inputs $x_t$ with ARMA noise | [3](../notes/03_filters_smoothing_and_decomposition.md) |
| **Transformer** | an encoder–decoder built from attention, feed-forward layers, residuals and normalization; no recurrence | [11](../notes/11_transformers.md) |
| **Truncated BPTT** | backpropagating only $k$ steps back in time | [8](../notes/08_rnn_lstm_gru_and_seq2seq.md) |
| **Vanishing / exploding gradient** | geometric shrinkage or growth of gradients through many layers or time steps | [8](../notes/08_rnn_lstm_gru_and_seq2seq.md) |
| **Vanishing moments** | a wavelet with $p$ vanishing moments is orthogonal to polynomials of degree $<p$ | [4](../notes/04_wavelets.md) |
| **Viterbi algorithm** | dynamic programming for the single most probable hidden-state path | [6](../notes/06_markov_models_hmm_and_em.md) |
| **Weight decay** | an $\ell_2$ penalty on network weights; each step shrinks them by a factor | [7](../notes/07_neural_network_foundations.md) |
| **White noise** | uncorrelated, constant mean and variance; need not be independent or Gaussian | [2](../notes/02_classical_time_series_models.md) |
