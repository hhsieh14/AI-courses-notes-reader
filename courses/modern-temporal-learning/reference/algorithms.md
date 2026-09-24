# Algorithms

Step-by-step procedures from the notes, in the order the chapters introduce them. Each one links to the chapter with the explanation.

## Box–Jenkins model refinement ([Ch 2](../notes/02_classical_time_series_models.md))

```text
input: series x (optionally exogenous inputs, candidate seasonal period s)
1. plot x; look for trend, changing variance, seasonality
2. stabilize variance (log / Box–Cox) if the spread grows with the level
3. difference: regular (1−B)^d and/or seasonal (1−B^s)^D until a unit-root test and the ACF say stationary
4. read the ACF / PACF for small candidate orders p, q (P, Q seasonal)
5. fit candidates; compare AIC / BIC
6. check residuals: plot, mean ≈ 0, ACF + Ljung–Box, histogram / Q–Q
7. if structure remains, add the missing term and refit; otherwise forecast on a time-ordered holdout
```

Worked path: ARIMA(1,0,0) under-predicts the trend → ARIMA(1,1,0) fixes the level but leaves lag-1 residual correlation → ARIMA(1,1,1).

## Tuning an EWMA ([Ch 3](../notes/03_filters_smoothing_and_decomposition.md))

```text
for λ in grid (e.g. 0.05 … 0.95):
    y_0 ← mean of the first few observations
    for t = 1..n:  forecast_t ← y_{t−1};  y_t ← λ·x_t + (1−λ)·y_{t−1}
    score(λ) ← MSE / MAE / MAPE of forecast_t vs x_t   (skip a short burn-in)
return argmin score
```

## Haar discrete wavelet transform ([Ch 4](../notes/04_wavelets.md))

```text
input: x of length 2^J;  a ← x
for level = 1..K:
    approx_i ← (a_{2i} + a_{2i+1}) / √2
    detail_i ← (a_{2i} − a_{2i+1}) / √2
    store detail at this level;  a ← approx
return (a, details for levels 1..K)      # total coefficients = len(x)
denoise: soft-threshold the details at σ̂·√(2 log N), σ̂ = median(|finest details|)/0.6745, then invert
```

## PCA monitoring ([Ch 5](../notes/05_pca_and_regularization.md))

```text
fit on normal data: standardize with training mean/std; eigendecompose the covariance; keep K components (≈ 90% variance)
for a new observation x:
    z ← V_Kᵀ x_std
    T² ← Σ_k z_k² / λ_k                       # unusual inside the model
    Q  ← ‖x_std − V_K z‖²                      # unusual outside the model
    alarm if T² or Q exceeds its control limit (set from the training distribution)
```

## Forward–backward ([Ch 6](../notes/06_markov_models_hmm_and_em.md))

```text
α_1(i) = π_i F_i(y_1);          α_{t+1}(j) = [Σ_i α_t(i) a_ij] F_j(y_{t+1})     (rescale each step by c_t = Σ_i α_t(i))
β_T(i) = 1;                     β_t(i) = Σ_j a_ij F_j(y_{t+1}) β_{t+1}(j)        (rescale with the same c_{t+1})
P(y) = Σ_i α_T(i)   (log P(y) = Σ_t log c_t when scaling)
γ_t(i) = α_t(i) β_t(i) / P(y)
```

## Viterbi ([Ch 6](../notes/06_markov_models_hmm_and_em.md))

```text
δ_1(i) = log π_i + log F_i(y_1)
for t = 1..T−1, each j:
    ψ_{t+1}(j) = argmax_i [δ_t(i) + log a_ij]
    δ_{t+1}(j) = δ_t(ψ_{t+1}(j)) + log a_{ψ_{t+1}(j) j} + log F_j(y_{t+1})
x*_T = argmax_i δ_T(i);   for t = T−1..1:  x*_t = ψ_{t+1}(x*_{t+1})
```

## EM for HMMs (Baum–Welch) and GMMs ([Ch 6](../notes/06_markov_models_hmm_and_em.md))

```text
initialize parameters (K-means for means; uniform or random transitions)
repeat until the change in log-likelihood < tol:
    E-step:  HMM → γ_t(i), ξ_t(i,j) from forward–backward
             GMM → responsibilities r_nk ∝ π_k N(x_n | μ_k, Σ_k)
    M-step:  HMM → π_i = γ_1(i);  a_ij = Σ_t ξ_t(i,j) / Σ_t γ_t(i);  emissions from γ-weighted counts
             GMM → N_k = Σ_n r_nk;  μ_k, Σ_k = r-weighted mean / covariance (+ small ridge);  π_k = N_k / N
run from several starts; keep the best likelihood
```

## Training a feed-forward network ([Ch 7](../notes/07_neural_network_foundations.md))

```text
encode (one-hot) and standardize inputs using training statistics
choose architecture, activation, output function + matching loss
initialize weights randomly (He / Glorot scale); biases 0
for epoch = 1..E:
    shuffle; for each minibatch b:
        forward pass → loss (+ λ‖w‖²) → backprop gradients
        v ← β v + g;  θ ← θ − η v            # momentum SGD (or Adam)
    evaluate validation loss; keep the best weights; stop after `patience` epochs without improvement
    decay η
```

## Truncated BPTT with stateful batches ([Ch 8](../notes/08_rnn_lstm_gru_and_seq2seq.md))

```text
split each long sequence into consecutive chunks of length k; align row i of every batch to the same sequence
h ← 0
for each batch in order (no shuffling):
    h ← h.detach()                          # keep the value, cut the gradient
    outputs, h ← RNN(batch, h)
    loss ← Σ_t L(y_t, ŷ_t);  backprop through these k steps only
    clip ‖grad‖ to c;  optimizer step
at the end of the epoch: h ← 0
```

## Dilated TCN ([Ch 9](../notes/09_temporal_convolutional_networks.md))

```text
choose kernel k and dilation levels L so that R = 1 + 2(k−1)·Σ_blocks Σ_ℓ 2^ℓ ≥ needed history
for each block, for d in 1, 2, 4, …, 2^{L−1}:
    y ← Dropout(ReLU(WeightNorm(CausalConv(x, k, d))))   # left-pad by (k−1)d
    y ← Dropout(ReLU(WeightNorm(CausalConv(y, k, d))))
    x ← ReLU(y + (Conv1×1(x) if channels differ else x))
head: linear map on the last time step → 1 step (recursive) or H steps (direct)
```

## Contrastive pretraining (SimCLR-style) ([Ch 10](../notes/10_representation_learning.md))

```text
for each minibatch of K series:
    make two random views of each (crop, jitter, scale) → 2K inputs
    h ← f(x);  z ← g(h);  z ← z / ‖z‖
    S ← z zᵀ / τ;  set S_ii = −∞
    loss ← mean over i of cross-entropy(S_i, partner(i))    # NT-Xent
    update f, g
after training: discard g; use h = f(x) with a linear probe / k-NN / clustering
```

## Real-vs-artificial anomaly scoring ([Ch 10](../notes/10_representation_learning.md))

```text
generate artificial series from a reference distribution (shuffle in time, independent marginals, uniform range)
label real = 0, artificial = 1; train any probabilistic classifier
anomaly score(x) = P(y = 1 | x) = 1 − P(real | x)
```

## Multi-head self-attention block ([Ch 11](../notes/11_transformers.md))

```text
X ← embeddings + positional encoding                     # (T, d_model)
for each head h:  Q_h, K_h, V_h = X W_Q^h, X W_K^h, X W_V^h     # (T, d_model/H)
    A_h = softmax(Q_h K_hᵀ / √d_k + mask) V_h            # mask = −∞ above the diagonal in decoders
Z = Concat(A_1..A_H) W_O
X ← LayerNorm(X + Z);   X ← LayerNorm(X + FFN(X))       # post-LN (pre-LN: normalize before each sublayer)
```

## Autoregressive decoding ([Ch 11](../notes/11_transformers.md))

```text
memory ← Encoder(source);  out ← [BOS];  cache ← empty
repeat:
    logits ← Decoder(out[-1], memory, cache)       # KV cache: compute only the newest position
    next ← argmax / sample / beam search over softmax(logits)
    out.append(next)
until next = EOS or length = max (for forecasting: until H steps)
```

Training uses **teacher forcing** instead: feed the shifted true targets, compute every position in parallel under the causal mask, and minimize the summed next-step loss.
