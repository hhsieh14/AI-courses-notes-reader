# Visual Index

This page is generated from the final chapter files.

Redrawn diagrams provide stable, reader-friendly explanations in the same
style used throughout the repository. Source figures are retained only when
they preserve exact results, handwriting, taxonomies, or details that should
not be simplified.

## Fourier Analysis and Wavelets

[Open chapter](../notes/04_wavelets.md)

![Qualitative comparison of Haar, Daubechies, Symlet, and Coiflet scaling and wavelet characteristics](../assets/clean_diagrams/wavelet_family_comparison.png)

*Redrawn course diagram — Qualitative comparison of Haar, Daubechies, Symlet, and Coiflet scaling and wavelet characteristics.*


![Wavelet pyramid with repeated decomposition of the approximation branch](../assets/clean_diagrams/wavelet_pyramid.png)

*Redrawn course diagram — Wavelet pyramid with repeated decomposition of the approximation branch.*


![db2 and db7 plateau-detail comparison](../assets/clean_diagrams/plateau_wavelet_localization.png)

*Redrawn course diagram — Short-support wavelets localize plateau boundaries more sharply than longer-support wavelets.*


## PCA, Bias-Variance, and Regularization

[Open chapter](../notes/05_pca_and_regularization.md)

![Correlated data and PCA directions from the course page](../assets/clean_diagrams/pca_coordinate_rotation.png)

*Redrawn course diagram — PCA rotates correlated observations onto orthogonal directions ordered by variance.*


![Euclidean circles versus covariance-aligned Mahalanobis ellipses and the stronger penalty along low-variance directions](../assets/clean_diagrams/mahalanobis_geometry.png)

*Redrawn course diagram — Euclidean circles versus covariance-aligned Mahalanobis ellipses and the stronger penalty along low-variance directions.*


![Schematic coefficient paths showing smooth ridge shrinkage and lasso coefficients reaching exact zero](../assets/clean_diagrams/ridge_lasso_paths.png)

*Redrawn course diagram — Schematic coefficient paths showing smooth ridge shrinkage and lasso coefficients reaching exact zero.*


## Markov Chains, Hidden Markov Models, and EM

[Open chapter](../notes/06_markov_models_hmm_and_em.md)

![Hidden-state and observed-sequence example from the course page](../assets/clean_diagrams/hmm_hidden_observed_sequences.png)

*Redrawn course diagram — A piecewise latent state sequence generates noisy observations through the emission model.*


![Soft responsibilities preserve uncertainty, while K-means makes one hard assignment](../assets/clean_diagrams/em_soft_vs_hard_assignments.png)

*Redrawn course diagram — Soft responsibilities preserve uncertainty, while K-means makes one hard assignment.*


## Neural-Network Foundations and Training

[Open chapter](../notes/07_neural_network_foundations.md)

![Multilayer feed-forward network from the course page](../assets/clean_diagrams/feedforward_depth_width.png)

*Redrawn course diagram — Network width counts units within a layer, while depth counts successive transformations.*


## RNNs, Stateful Training, LSTMs, GRUs, and Encoder-Decoder Models

[Open chapter](../notes/08_rnn_lstm_gru_and_seq2seq.md)

![Random, expanding-window, and sliding-window temporal validation](../assets/clean_diagrams/temporal_cross_validation.png)

*Redrawn course diagram — Random, expanding-window, and sliding-window temporal validation.*


![Basic and stacked RNN architectures from the course page](../assets/clean_diagrams/rnn_basic_and_stacked.png)

*Redrawn course diagram — Stacked RNNs combine recurrent state propagation through time with vertical depth.*


![LSTM cell diagrams from the course page](../assets/clean_diagrams/lstm_memory_gates.png)

*Redrawn course diagram — Forget, input, candidate, and output gates regulate the LSTM cell-state pathway.*


![Encoder-decoder sequence and forecast diagrams from the course page](../assets/clean_diagrams/encoder_decoder_sequence_forecast.png)

*Redrawn course diagram — Encoder-decoder models map an input sequence to a target sequence or forecast horizon.*


## Temporal Convolutional Networks

[Open chapter](../notes/09_temporal_convolutional_networks.md)

![TCN causality, exponentially increasing dilation, and residual-block structure](../assets/clean_diagrams/tcn_overview.png)

*Redrawn course diagram — TCN causality, exponentially increasing dilation, and residual-block structure.*


## Temporal Representation and Contrastive Learning

[Open chapter](../notes/10_representation_learning.md)

![Temporal encoder-bottleneck-decoder architecture and downstream reuse of the learned representation](../assets/clean_diagrams/temporal_autoencoder.png)

*Redrawn course diagram — Temporal encoder-bottleneck-decoder architecture and downstream reuse of the learned representation.*


![Temporal positive and negative pair construction, shared encoder, projection head, and contrastive loss](../assets/clean_diagrams/contrastive_temporal_pipeline.png)

*Redrawn course diagram — Temporal positive and negative pair construction, shared encoder, projection head, and contrastive loss.*


![Projection-head comparison from the course page](../assets/clean_diagrams/projection_head_design.png)

*Redrawn course diagram — Projection heads shape the contrastive objective while the encoder representation is retained downstream.*


## Transformers and Temporal Applications

[Open chapter](../notes/11_transformers.md)

![Simplified transformer encoder-decoder stack with self-attention, masked attention, cross-attention, feed-forward layers, and residual normalization](../assets/clean_diagrams/transformer_overview.png)

*Redrawn course diagram — Simplified transformer encoder-decoder stack with self-attention, masked attention, cross-attention, feed-forward layers, and residual normalization.*


![Sentence-level attention interpretation and parallel multi-head query-key-value projections](../assets/clean_diagrams/multihead_attention.png)

*Redrawn course diagram — Sentence-level attention interpretation and parallel multi-head query-key-value projections.*


![Transformer decoder block with masked self-attention, encoder-decoder attention, and autoregressive generation](../assets/clean_diagrams/transformer_decoder_generation.png)

*Redrawn course diagram — Transformer decoder block with masked self-attention, encoder-decoder attention, and autoregressive generation.*


![Residual learning block from the course page](../assets/clean_diagrams/residual_learning_block.png)

*Redrawn course diagram — A residual block adds the learned correction $F(x)$ to the identity path $x$.*


![Taxonomy of transformer methods for time-series modeling](../assets/clean_diagrams/time_series_transformer_taxonomy.png)

*Redrawn course diagram — Time-series transformers vary in representation, attention, architecture, and application.*


## Handwritten Appendix: Stationarity and Covariance Exercise

[Open chapter](../notes/12_handwritten_appendix.md)

![Full handwritten stationarity page](../assets/clean_diagrams/stationarity_recap.png)

*Redrawn appendix diagram — Strict stationarity, weak stationarity, IID assumptions, and sampling-rate intuition.*


![Full handwritten covariance exercise](../assets/clean_diagrams/first_difference_covariance.png)

*Redrawn appendix diagram — First differencing separates process innovations and observation-noise covariance.*


