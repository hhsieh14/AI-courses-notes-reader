# CSE 598: Modern Temporal Learning

My study notes for ASU's graduate course on learning from time-ordered data. It starts with stationarity and ARIMA and ends with transformers. I rebuilt my handwritten lecture notes into chapters that read on their own. Each one has worked examples, redrawn diagrams, runnable snippets whose outputs I verified, and answered review questions.

## Who this is for

- **Students** taking a time-series or sequence-modeling course who want one path from classical statistics to deep learning.
- **Practitioners** who know deep learning but want the classical models (ARIMA, exponential smoothing, HMMs) that are still hard baselines to beat.
- **Anyone preparing for interviews** on RNNs, attention or forecasting evaluation. The Q&A sections are written for that.

## Start here

1. Skim the [course map](course_map.md): one row per chapter, the question it answers and the takeaways.
2. Read [Chapter 1](notes/01_temporal_data_and_learning.md). It sets up the ideas the rest depends on (dependence, leakage, temporal validation).
3. Go in order, or jump. Every chapter links back to what it builds on.

## Chapters

| Part | Chapters |
|---|---|
| Classical time series | [1 Temporal data](notes/01_temporal_data_and_learning.md) · [2 Stationarity & ARIMA](notes/02_classical_time_series_models.md) · [3 Filters & decomposition](notes/03_filters_smoothing_and_decomposition.md) · [4 Fourier & wavelets](notes/04_wavelets.md) |
| Statistical learning | [5 PCA & regularization](notes/05_pca_and_regularization.md) · [6 Markov models, HMMs & EM](notes/06_markov_models_hmm_and_em.md) |
| Deep sequence models | [7 Neural-network foundations](notes/07_neural_network_foundations.md) · [8 RNN / LSTM / GRU / seq2seq](notes/08_rnn_lstm_gru_and_seq2seq.md) · [9 TCNs](notes/09_temporal_convolutional_networks.md) · [10 Representation & contrastive learning](notes/10_representation_learning.md) · [11 Transformers](notes/11_transformers.md) |
| Appendix | [12 Stationarity recap & differencing derivation](notes/12_handwritten_appendix.md) |

## Reference pages

| Page | Use it to… |
|---|---|
| [Equations](reference/equations.md) | find any formula fast |
| [Algorithms](reference/algorithms.md) | see step-by-step procedures (forward–backward, Viterbi, EM, BPTT, TCN, attention) |
| [Model comparisons](reference/model_comparisons.md) | choose between models |
| [Glossary](reference/glossary.md) | look up a term |
| [Concept dependencies](reference/concept_dependencies.md) | see what to learn before what |
| [Questions & answers](reference/questions_and_discussion.md) | test yourself |
| [Visual index](reference/visual_index.md) | browse every diagram |

## Related notes

- [CSE 575 Statistical Machine Learning](../cse-575-statistical-machine-learning/README.md): the non-temporal foundations (regression, SVMs, GMM/EM, PCA, backprop).
- [Statistical Learning Theory](../statistical-learning-theory/README.md): why these models generalize.
- [ML System Design notes](https://github.com/hhsieh14/ml-system-design-learning-notes): how models like these are deployed.
