# Model Comparisons

## AR versus MA

| Dimension | AR $p$ | MA $q$ |
|---|---|---|
| Direct inputs | Previous observations | Current and previous disturbances |
| Disturbance effect | Propagates through future values | Directly persists for only $q$ periods |
| ACF behavior emphasized in course | Gradual decay | Cutoff beyond lag $q$ |
| Order-identification cue | PACF used to identify $p$ | ACF cutoff used to identify $q$ |
| Example in notes | Stirred tank / tanks in series | Finite disturbance propagation |

**Sources:** CSE598MTL.pdf, pp. 6-8 and 11

## ARIMA versus SARIMAX

| Dimension | ARIMA | SARIMAX |
|---|---|---|
| Regular AR terms | Yes | Yes |
| Regular differencing | Yes | Yes |
| Regular MA terms | Yes | Yes |
| Seasonal terms | Not in basic form | Yes |
| Exogenous predictors | Not in basic form | Yes |
| Example in notes | Tank-model forecasting | Monthly CO2 trend and cycle |

**Sources:** CSE598MTL.pdf, pp. 11-13

## IID versus white noise

| Dimension | IID | White noise as defined in notes |
|---|---|---|
| Identical marginal distributions | Yes | Not explicitly required by the note |
| Independence | Yes | Not required |
| No temporal correlation | Yes | Yes |
| Normality | Not required by IID itself | Explicitly not required |
| Role in chapter | Stationary extreme | Desired residual behavior |

**Source:** CSE598MTL.pdf, p. 5

## One-step versus long-horizon prediction

| Dimension | One-step prediction | Long-horizon recursive prediction |
|---|---|---|
| Recent true values available | Typically yes | Not for every future step |
| Error accumulation | Limited | Can compound |
| Course illustration | Close training fit | Weaker 20-point test prediction |

**Source:** CSE598MTL.pdf, p. 7

## RNN versus TCN

| Dimension | RNN | TCN |
|---|---|---|
| Temporal mechanism | Recurrent hidden state | Causal convolution |
| Position computation | Sequential through hidden states | Parallel within each layer |
| Context control | Learned state memory | Explicit receptive field |
| Long-context design | LSTM/GRU gates and BPTT | Dilation, width, depth |
| Course regularization | Dropout, state handling | Dropout, residual blocks |

**Sources:** CSE598MTL.pdf, pp. 68-82

## Ordinary versus dilated causal convolution

| Dimension | Ordinary causal | Dilated causal |
|---|---|---|
| Tap spacing | Consecutive | $d$ time steps |
| Long receptive field | More layers or larger filters | Increasing dilation |
| Equation index | $x_{t-i}$ | $x_{t-di}$ |
| Course purpose | Preserve causality | Efficient long-history coverage |

**Sources:** CSE598MTL.pdf, pp. 79-81

## Recurrent connection versus residual connection

| Dimension | Recurrent | Residual |
|---|---|---|
| Skips across | Time | Network depth |
| Carries | Hidden state | Earlier representation |
| Main purpose in notes | Temporal memory | Deep-network training and identity path |
| Used by | RNN/LSTM/GRU | TCN residual block |

**Sources:** CSE598MTL.pdf, pp. 68-82

## PCA versus kernel PCA versus autoencoder

| Dimension | PCA | Kernel PCA | Autoencoder |
|---|---|---|---|
| Representation mapping | Linear | Implicit nonlinear feature space | Learned neural mapping |
| Objective | Variance | Variance in kernel space | Reconstruction |
| Explicit decoder | No | No | Yes during training |
| Temporal architecture | No | No | RNN/LSTM possible |
| Main limitation in notes | Linear/time-order agnostic | Kernel choice | Reconstruction may not help downstream task |

**Sources:** CSE598MTL.pdf, pp. 83-86

## Autoencoder versus contrastive learning

| Dimension | Autoencoder | Contrastive learning |
|---|---|---|
| Target | Original input | Positive instance/view |
| Main loss | Reconstruction distance | Pairwise similarity classification |
| Invariance | Emerges indirectly | Defined by augmentations |
| Representation retained | Bottleneck | Encoder output before head |
| Evaluation concern | Reconstruction/downstream mismatch | Pair-definition/downstream mismatch |

**Sources:** CSE598MTL.pdf, pp. 85-90

## Encoder representation versus projection head

| Dimension | Encoder $h$ | Head output $z$ |
|---|---|---|
| Intended downstream reuse | Yes | Usually no |
| Loss applied directly | Not necessarily | Yes |
| Information role | Preserve broadly useful features | Satisfy contrastive invariance |
| Course example | 2048-dimensional representation | Variable lower dimension |

**Sources:** CSE598MTL.pdf, pp. 88-90

## Small versus large temperature

| Dimension | Small $\tau$ | Large $\tau$ |
|---|---|---|
| Softmax shape | Sharp | Flat |
| Largest similarity probability | Near one | Closer to others |
| Negative separation pressure | Concentrated on hardest comparisons | More evenly distributed |
| Limiting intuition | One-hot-like | Uniform-like |

**Source:** CSE598MTL.pdf, p. 89

## RNN versus transformer

| Dimension | RNN/LSTM/GRU | Transformer |
|---|---|---|
| Sequence interaction | Recurrent state | Attention |
| Training across positions | Sequential | Parallel within layer |
| Order representation | Recurrence | Positional encoding |
| Long-range connection | Through many steps | Direct query-key relation |
| Main long-sequence challenge | Gradient/state memory | Attention computation |

**Sources:** CSE598MTL.pdf, pp. 68-100

## Encoder self-attention versus masked decoder attention versus cross-attention

| Type | Queries | Keys and values | Allowed positions |
|---|---|---|---|
| Encoder self-attention | Encoder | Encoder | All input positions |
| Decoder masked self-attention | Decoder | Decoder | Current prefix only |
| Encoder-decoder attention | Decoder | Encoder | All encoded input positions |

**Sources:** CSE598MTL.pdf, pp. 93 and 96-97

## Training versus inference decoding

| Dimension | Training | Inference |
|---|---|---|
| Decoder input | Known shifted target | Previous generated output |
| Parallel target computation | Yes | No |
| Masking | Prevent future target access | Future tokens do not exist |
| Course term | Teacher forcing | Test mode |

**Source:** CSE598MTL.pdf, p. 98

## Self-attention versus TCN

| Dimension | Self-attention | TCN |
|---|---|---|
| Context selection | Learned pairwise weights | Fixed receptive-field connectivity |
| Long context | Direct global attention | Dilation and depth |
| Position order | Positional encoding | Causal convolution structure |
| Parallel computation | Yes within layer | Yes within layer |
| Long-sequence cost | Dense attention can be high | Controlled by convolution design |

**Sources:** CSE598MTL.pdf, pp. 79-100
