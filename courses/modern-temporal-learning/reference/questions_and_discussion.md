# Questions and Discussion

This file consolidates unresolved questions preserved from the source notes.
They are intentionally not resolved without additional course material,
implementation code, instructor clarification, or outside research.

## Temporal Data and Temporal Learning

1. When should a raw sequence be used instead of engineered features?
2. Should temporal segments be mutually exclusive or overlapping?
3. When does a symbolic representation discard too much numerical detail?
4. How should landmarks be selected when no natural event boundary exists?
5. How can we distinguish a short anomalous interval from a persistent
   distributional change?
6. Which distances remain meaningful when sequences have unequal length
   or shifted timing?

These questions are motivated by the slides and annotations but are not
fully answered on pages 2-4.

**Chapter source:** `../notes/01_temporal_data_and_learning.md`

## Stationarity and Classical Time-Series Models

1. Why does the note describe integrated moving average as a first
   practical choice?
2. How should the order of differencing be selected without removing too
   much structure?
3. How far can an AR model forecast before recursive error dominates?
4. When should a visible ACF pattern be attributed to trend rather than
   an AR or MA term?
5. How should the seasonal period be estimated when it is not known?
6. Why do the handwritten notes associate PACF with a cycle on page 13?
7. What formal test, if any, was intended for stationarity in this course?
8. How should candidate models be compared when several leave
   approximately white residuals?

The source pages raise these issues but do not fully resolve them.

**Chapter source:** `../notes/02_classical_time_series_models.md`

## Filters, Smoothing, Decomposition, and Transfer Functions

1. How should the effective memory of EWMA be quantified from $\lambda$?
2. Why is EWMA the minimum-MSE predictor for the stated IMA $(1,1)$
   model?
3. How should $y_0$ be selected for a short series?
4. When should MSE, MAE, or MAPE be preferred?
5. How should additive versus multiplicative seasonality be diagnosed?
6. How are Holt-Winters parameters estimated in the software used in the
   course?
7. How should STL window sizes be selected beyond the qualitative rules
   given on page 18?
8. How robust is STL to prolonged anomalies?
9. Which Prophet terms were actually used in the course implementation?
10. How should transfer-function lag orders be selected in practice?

These questions are motivated by the source pages but are not fully
answered there.

**Chapter source:** `../notes/03_filters_smoothing_and_decomposition.md`

## Fourier Analysis and Wavelets

1. Which boundary-extension method was used in the db2 edge example?
2. How does the course define increasing versus decreasing "wavelet level"
   in the software output?
3. How should the decomposition depth $K$ be selected?
4. How should a wavelet family be selected for anomaly detection?
5. What threshold rule produced the 51 selected CO2 coefficients?
6. When should hard thresholding be preferred to soft thresholding?
7. How does coefficient magnitude change with scale for the same type of
   discontinuity?
8. What is the intended formal relationship between matching pursuit,
   basis pursuit, ridge, and lasso in the page-30 bullet list?
9. How should edge coefficients be interpreted when the signal length is
   not a power of two?
10. How are wavelet features eventually used by the learning models later
    in the course?

These questions arise from the source pages and remain unresolved there.

**Chapter source:** `../notes/04_wavelets.md`

## PCA, Bias-Variance, and Regularization

1. What row/column construction was used when applying PCA to the temporal
   traces on page 33?
2. Were predictors standardized in every PCA example?
3. How were the class labels in the page-36 score plots generated?
4. Which distance or statistic was used for actual PCA anomaly detection?
5. What exact threshold distribution was intended for $T^2$?
6. Why is the direction of greatest variance useful when the target may
   depend on a low-variance direction?
7. Was page 39 intended to fit 99 of 999 predictors?
8. How was $\lambda\approx4$ chosen in the ridge example?
9. Did the course use cross-validation for ridge and lasso?
10. Is page 42 displaying fitted values while labeling them as
    coefficient estimates?
11. How is $\sigma$ estimated for the wavelet threshold
    $C=\sigma\sqrt{2\log N}$?
12. When should wavelet coefficients be thresholded before versus during
    supervised model fitting?

These questions are supported by ambiguities or open points in the source
pages.

**Chapter source:** `../notes/05_pca_and_regularization.md`

## Markov Chains, Hidden Markov Models, and EM

1. Which exact recurrent-state definition was intended on page 51?
2. What smoothing or prior was intended to prevent zero-probability HMM
   estimates for short sequences?
3. Which HMM applications in the handwritten list were emphasized by the
   professor?
4. Did the course implement scaling or log-space calculations for long
   forward-backward sequences?
5. How was the number of hidden states $Q$ selected?
6. How were Gaussian-emission covariance structures selected?
7. Which definition of “optimal state sequence” was used in assignments?
8. Did the Baum-Welch implementation use multiple random initializations?
9. How was $K$ selected for Gaussian mixtures?
10. Did the Gaussian-mixture implementation permit singular covariance
    matrices?
11. What convergence threshold was used for likelihood or parameters?
12. Does the strict likelihood inequality on page 50 allow equal
    likelihood at convergence?

These questions arise from the source pages and are not fully resolved
there.

**Chapter source:** `../notes/06_markov_models_hmm_and_em.md`

## Neural-Network Foundations and Training

1. Which output/loss combinations were actually used in the course code?
2. What exact theorem and assumptions support the approximation statements
   on page 57?
3. Does the course use averaged or summed losses over $N$?
4. Which logarithm base was used in implementations?
5. Did training use full-batch, stochastic, or minibatch updates?
6. What minibatch size was used?
7. Was the learning-rate schedule really $\eta_r=1/r$, or was that only
   an example?
8. What initialization distribution and scaling were used for deeper
   networks?
9. How was early stopping patience selected?
10. Were biases excluded from weight decay?
11. Was dropout rescaled during training or prediction?
12. How was momentum parameterized?
13. Why does the page-65 parameter example use 70 rather than 80 encoded
    inputs?
14. Were categorical predictors encoded with a reference category instead
    of full one-hot encoding?
15. How does this feed-forward setup change when the inputs and outputs are
    temporal sequences?

These questions come directly from open or inconsistent points in the
source pages.

**Chapter source:** `../notes/07_neural_network_foundations.md`

## RNNs, Stateful Training, LSTMs, GRUs, and Encoder-Decoder Models

1. What formal answer was intended for the page-66 permutation and
   uniqueness question?
2. Which direct versus recursive multi-horizon strategy was implemented?
3. What exact blocked-validation method was used in assignments?
4. Was a gap placed between temporal training and test blocks?
5. Did the RNN tensor use fixed $T$, padding, masking, or variable
   sequence lengths?
6. Which activation function was used in the basic RNN examples?
7. What truncation length was used for BPTT?
8. How was exploding-gradient behavior handled?
9. Were gradients clipped?
10. How were long-sequence windows constructed at the dataset boundaries?
11. Which framework's definition of `stateful=True` is represented on
    page 73?
12. Were states detached from the gradient graph between batches?
13. Were LSTM and GRU gates implemented with concatenated matrices or
    separate input/recurrent matrices?
14. Which activation function is denoted by $\phi$ in the LSTM and GRU
    candidate equations?
15. How were encoder and decoder states initialized?
16. Did the page-78 handwritten weighted-state idea correspond to an
    implemented attention model?

These questions come from open or framework-dependent points in the source
pages.

**Chapter source:** `../notes/08_rnn_lstm_gru_and_seq2seq.md`

## Temporal Convolutional Networks

1. Which exact TCN paper or implementation supplied the architecture
   figures?
2. What padding convention was used at the beginning of each sequence?
3. Does the model trim padded outputs before calculating loss?
4. Which filter-width convention should replace the conflicting uses of
   $K$ on page 79?
5. Is the page-80 printed formula $R=L+K-1$ a typo for
   $R=1+L(K-1)$?
6. What receptive-field formula was used for stacked dilation blocks?
7. Did dilation reset at the beginning of each residual block?
8. What channel counts were used in each block?
9. Was weight normalization used in the course implementation?
10. What dropout rate was used?
11. Were predictions for $H>1$ generated recursively or in parallel in
    assignments?
12. Were gradients or activations affected by long left-padding regions?
13. Which Sequential MNIST and P-MNIST numerical results were considered
    most important?
14. Under what datasets did the course observe TCN performance better than
    RNN or LSTM?
15. How should receptive field be selected relative to the actual
    dependency length in the data?

These questions arise from source ambiguities or omitted implementation
details.

**Chapter source:** `../notes/09_temporal_convolutional_networks.md`

## Temporal Representation and Contrastive Learning

1. What downstream metric was used to compare PCA, LSTM, and other
   temporal representations?
2. What exact generative/discriminative probability factorization was
   intended on page 83?
3. Were kernel matrices centered before kernel PCA?
4. How was the kernel bandwidth selected?
5. Was the autoencoder bottleneck deterministic?
6. How were variable-length temporal sequences reconstructed?
7. Was anomaly detection based on latent distance, reconstruction error,
   or both?
8. How was local anomaly timing recovered from one global embedding?
9. Which transformations were valid positive-pair augmentations for the
   course's time-series datasets?
10. How were cyclic processes handled under the slowness assumption?
11. Were false negatives present when different series had the same
    underlying state?
12. Which encoder and projection-head dimensions were used?
13. What temperature value was used?
14. Did the loss use all other augmented instances as negatives?
15. Were representations normalized before cosine similarity?
16. Why did the projection head improve the retained encoder
    representation?
17. Was mutual information estimated explicitly?
18. Should the page-91 anomaly score use class-1 probability rather than
    class-0 probability?
19. How were artificial time series generated for the anomaly classifier?

These questions follow directly from omitted details or ambiguities in the
source pages.

**Chapter source:** `../notes/10_representation_learning.md`

## Transformers and Temporal Applications

1. Which exact transformer variant was implemented in the course?
2. Was attention normalized by $\sqrt{d_k}$ in all code examples?
3. How were query, key, and value dimensions divided across heads?
4. Does page 94 use $d_k=d_v=d_{\text{model}}/H$, or different
   dimensions?
5. Were encoder and decoder weights shared anywhere besides embeddings?
6. Was layer normalization applied before or after each sublayer?
7. What masking value was used before softmax?
8. Did decoder cross-attention use every encoder block or only the final
   block?
9. How was sequence termination handled for numerical time-series output?
10. Was teacher forcing used with a schedule or always applied?
11. Which positional encoding—fixed sinusoidal, learned, or timestamp
    encoding—was used for time series?
12. Were embeddings tied in the course implementation?
13. How was inference caching handled?
14. Did the time-series transformer use an encoder-only, decoder-only, or
    encoder-decoder architecture?
15. Which low-rank attention approximation was intended on page 100?
16. What multiresolution hierarchy was used for long time series?
17. How were missing values and irregular timestamps represented?
18. Which BERT pretraining details were included only as historical
    context versus used in assignments?

These questions arise from implementation details or notation not fully
specified in the source pages.

**Chapter source:** `../notes/11_transformers.md`

## Handwritten Appendix: Stationarity and Covariance Exercise

1. Was page 103 intended to calculate lag-one autocorrelation only, or a
   general lag $k$ autocorrelation?
2. Were $e_t$ and $e_{t-1}$ intended to be independent?
3. Is the covariance substitution
   $E[e_te_{t-1}]=E[e_t^2]$ a transcription mistake made during class?
4. Why does the final denominator contain $4\sigma_e^2$?
5. Was the numerator intended to be
   $-\sigma_e^2$, $\sigma_\delta^2$, or
   $\sigma_\delta^2-2\sigma_e^2$?
6. Was the exercise intended to derive the MA(1) autocorrelation of the
   differenced local-level model?
7. Did the instructor correct this exercise verbally after the handwritten
   derivation?
8. What exact lag notation is written in the highlighted heading?

These questions cannot be resolved from the PDF alone.

**Chapter source:** `../notes/12_handwritten_appendix.md`
