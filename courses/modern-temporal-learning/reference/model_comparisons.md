# Model Comparisons

Side-by-side tables for the choices that come up most often. Each links to the chapter with the details.

## Picking a forecasting model

| Situation | Start with | Why |
|---|---|---|
| short series, one variable, clear trend/season | ETS / Holt–Winters, SARIMA ([Ch 2–3](../notes/02_classical_time_series_models.md)) | few parameters, strong baselines, prediction intervals |
| known external drivers | SARIMAX / transfer function ([Ch 3](../notes/03_filters_smoothing_and_decomposition.md)) | explicit input → output dynamics |
| hidden regimes | HMM ([Ch 6](../notes/06_markov_models_hmm_and_em.md)) | discrete states with interpretable transitions |
| many related series, nonlinear, lots of data | TCN or LSTM ([Ch 8–9](../notes/08_rnn_lstm_gru_and_seq2seq.md)) | shared weights across series, learned features |
| very long context, irregular timestamps, huge data | transformer ([Ch 11](../notes/11_transformers.md)) | direct long-range links, flexible inputs |
| always | a naive / seasonal-naive and a linear baseline | if you can't beat these, the fancy model isn't helping |

## Classical models

| | AR($p$) | MA($q$) |
|---|---|---|
| built from | past observations | current and past shocks |
| a shock's effect | decays geometrically, forever | lasts exactly $q$ steps |
| ACF | tails off | cuts off after $q$ |
| PACF | cuts off after $p$ | tails off |
| physical example | stirred tank, tanks in series | a disturbance with a finite effect |

| | ARIMA | SARIMAX |
|---|---|---|
| regular AR / differencing / MA | ✓ | ✓ |
| seasonal AR / differencing / MA | – | ✓ |
| exogenous regressors | – | ✓ |
| example | tank-level forecasting | Mauna Loa CO₂ (trend + annual cycle) |

| | IID | White noise | Weakly stationary |
|---|---|---|---|
| constant mean & variance | ✓ | ✓ | ✓ |
| uncorrelated across time | ✓ | ✓ | not required |
| independent | ✓ | not required | not required |
| identical distributions | ✓ | not required (only first two moments) | not required |
| role | simplest case | what good residuals look like | what AR/MA models describe |

| | One-step prediction | Recursive multi-step |
|---|---|---|
| uses | the latest *observed* value | its own earlier predictions |
| errors | don't accumulate | compound; the forecast reverts to the mean |
| a model can look | excellent | much worse at 20 steps |

## Hidden-state models

| | Markov chain | HMM | GMM |
|---|---|---|---|
| states observed? | yes | no | no (component labels) |
| links between time steps | transitions | transitions | none, each draw independent |
| learning | count transitions | Baum–Welch (EM) | EM |
| inference | – | forward–backward, Viterbi | responsibilities |

| | Per-step argmax of $\gamma_t$ | Viterbi |
|---|---|---|
| maximizes | expected number of correct states | probability of the whole path |
| output always a valid path? | no | yes |

## Sequence networks

| | RNN | LSTM | GRU | TCN | Transformer |
|---|---|---|---|---|---|
| time mechanism | recurrent state | state + gated cell | gated state | causal dilated convs | attention + positions |
| parallel over time | no | no | no | yes | yes |
| path between distant steps | $O(T)$ | $O(T)$, additive cell path | $O(T)$, gated | $O(\log T)$ | $O(1)$ |
| memory length | fades | long | long | exactly $R$ | whole window |
| cost per layer | $O(Td^2)$ | $4\times$ RNN | $3\times$ RNN | $O(Tkd^2)$ | $O(T^2d+Td^2)$ |
| streaming inference | cheap | cheap | cheap | window recompute | KV cache, grows with $T$ |

| | Stateless training | Stateful training |
|---|---|---|
| $\mathbf h_0$ at a batch | zeros | last state of the matching row in the previous batch |
| shuffling | fine | not allowed |
| reset | every batch | end of each epoch |

| | Ordinary causal conv | Dilated causal conv |
|---|---|---|
| taps | consecutive | spaced $d$ apart |
| receptive field | $1+L(k-1)$, linear | $1+(k-1)\sum d_\ell$, exponential with doubling $d$ |

| | Recurrent connection | Residual connection |
|---|---|---|
| skips across | time | layers |
| carries | hidden state | the layer's input, added to its output |
| used by | RNN / LSTM / GRU | TCN blocks, transformers, ResNets |

## Representation learning

| | PCA | Kernel PCA | Autoencoder | Contrastive |
|---|---|---|---|---|
| mapping | linear | nonlinear via kernel | learned | learned |
| objective | variance | variance in feature space | reconstruction | pick the positive |
| invariances | none by design | none | none by design | set by the pair rule |
| keep for downstream | scores | scores | bottleneck $\mathbf h$ | encoder $\mathbf h$ (drop the head) |
| main risk | misses nonlinearity | bandwidth, $O(N^2)$ memory | reconstructs irrelevant detail | pair rule removes useful info, false negatives |

| | Small temperature $\tau$ | Large $\tau$ |
|---|---|---|
| softmax | sharp, near one-hot | flat, near uniform |
| focus | hardest negatives | all negatives about equally |
| risk | punishes false negatives | weak learning signal |

## Attention

| Layer | Queries | Keys / values | Visible positions |
|---|---|---|---|
| encoder self-attention | encoder | encoder | all inputs |
| decoder masked self-attention | decoder | decoder | positions $\le i$ |
| cross-attention | decoder | final encoder output | all inputs |

| | Training | Inference |
|---|---|---|
| decoder input | true targets, shifted right (teacher forcing) | the model's own outputs |
| parallel over output positions | yes (the mask enforces causality) | no |
| mistakes | never fed back | can compound |

| | Self-attention | TCN |
|---|---|---|
| context selection | learned, content-dependent weights | fixed connectivity pattern |
| order | positional encoding | built into the causal structure |
| long-sequence cost | quadratic | linear |
| data needed | more | less |
