# Visual Index

Every diagram in the notes, grouped by chapter. Click a chapter title to read the diagram in context.

## [2. Stationarity and Classical Time-Series Models](../notes/02_classical_time_series_models.md)

![Stationarity patterns](../assets/clean_diagrams/stationarity_patterns.png)

**Strict (strong) stationarity:**

![AR and MA memory](../assets/clean_diagrams/ar_ma_memory.png)

![ARIMA refinement loop](../assets/clean_diagrams/arima_refinement_loop.png)

## [4. Fourier Analysis and Wavelets](../notes/04_wavelets.md)

![Wavelet family comparison](../assets/clean_diagrams/wavelet_family_comparison.png)

*Haar, Daubechies, Symlet and Coiflet families differ in support length, smoothness, symmetry and number of vanishing moments.*

![Wavelet pyramid](../assets/clean_diagrams/wavelet_pyramid.png)

![Plateau localization](../assets/clean_diagrams/plateau_wavelet_localization.png)

## [5. PCA, Bias–Variance, and Regularization](../notes/05_pca_and_regularization.md)

![PCA as a rotation](../assets/clean_diagrams/pca_coordinate_rotation.png)

*PCA rotates the axes to line up with the directions of greatest spread.*

![Mahalanobis geometry](../assets/clean_diagrams/mahalanobis_geometry.png)

![Ridge and lasso paths](../assets/clean_diagrams/ridge_lasso_paths.png)

## [6. Markov Chains, Hidden Markov Models, and EM](../notes/06_markov_models_hmm_and_em.md)

![Hidden and observed sequences](../assets/clean_diagrams/hmm_hidden_observed_sequences.png)

*A piecewise-constant hidden state generates noisy observations.*

![Soft versus hard assignment](../assets/clean_diagrams/em_soft_vs_hard_assignments.png)

## [7. Neural-Network Foundations and Training](../notes/07_neural_network_foundations.md)

![Width and depth](../assets/clean_diagrams/feedforward_depth_width.png)

*Width is the number of units in a layer; depth is the number of successive transformations.*

## [8. RNNs, LSTMs, GRUs and Encoder–Decoder Models](../notes/08_rnn_lstm_gru_and_seq2seq.md)

![Temporal cross-validation](../assets/clean_diagrams/temporal_cross_validation.png)

*Random, expanding-window and sliding-window validation.*

![Basic and stacked RNNs](../assets/clean_diagrams/rnn_basic_and_stacked.png)

*Left: the recurrence unrolled through time. Right: a stacked RNN, where state flows in time within each layer and upward between layers.*

![LSTM cell](../assets/clean_diagrams/lstm_memory_gates.png)

*The cell-state line along the top is only scaled by $\mathbf f_t$ and added to. The tanh is applied only on the branch that produces $\mathbf h_t$.*

![Encoder–decoder](../assets/clean_diagrams/encoder_decoder_sequence_forecast.png)

*Top: translation ("I am tired" → "Estoy cansado/a"). Bottom: the same structure for forecasting, a history window in and all $H$ horizon values out.*

## [9. Temporal Convolutional Networks](../notes/09_temporal_convolutional_networks.md)

![TCN overview](../assets/clean_diagrams/tcn_overview.png)

*Causal paths only point forward in time; dilation doubles the spacing at each layer; blocks are wrapped in residual connections.*

## [10. Temporal Representation and Contrastive Learning](../notes/10_representation_learning.md)

![Temporal autoencoder](../assets/clean_diagrams/temporal_autoencoder.png)

*Train the encoder–bottleneck–decoder, then discard the decoder and reuse $\mathbf h$ downstream.*

![Temporal contrastive pipeline](../assets/clean_diagrams/contrastive_temporal_pipeline.png)

*Positive and negative temporal pairs, a shared encoder, a projection head and the contrastive loss.*

![Projection-head comparison](../assets/clean_diagrams/projection_head_design.png)

*The head shapes the contrastive objective. The representation before it is what gets reused.*

## [11. Transformers and Temporal Applications](../notes/11_transformers.md)

![Transformer overview](../assets/clean_diagrams/transformer_overview.png)

*Encoder stack (left) and decoder stack (right). Every decoder block cross-attends to the output of the final encoder block.*

![Multi-head attention](../assets/clean_diagrams/multihead_attention.png)

*Left: attention from "it" in the example sentence. Right: $H$ parallel query/key/value projections, concatenated and projected by $W_O$.*

![Residual learning block](../assets/clean_diagrams/residual_learning_block.png)

*A residual block outputs $\mathcal F(\mathbf x)+\mathbf x$.*

![Decoder and generation](../assets/clean_diagrams/transformer_decoder_generation.png)

*Masked self-attention, cross-attention to the encoder and autoregressive generation.*

![Time-series transformer taxonomy](../assets/clean_diagrams/time_series_transformer_taxonomy.png)

*Time-series transformers vary by positional encoding, attention module and architecture, and are applied to forecasting, anomaly detection and classification.*

## [12. Appendix: Stationarity Recap and the Differenced Local-Level Model](../notes/12_handwritten_appendix.md)

![Stationarity recap](../assets/clean_diagrams/stationarity_recap.png)

*Strict stationarity, weak stationarity, IID and the sampling-rate intuition on one page.*

![First-difference covariance](../assets/clean_diagrams/first_difference_covariance.png)

*Consecutive differences share exactly one noise term, $e_t$, with opposite signs.*
