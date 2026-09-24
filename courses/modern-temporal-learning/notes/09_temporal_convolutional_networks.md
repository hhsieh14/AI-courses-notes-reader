# 9. Temporal Convolutional Networks

A temporal convolutional network (TCN) handles sequences with ordinary 1-D convolutions instead of recurrence. Three ideas make it work:

1. **causality:** the output at time $t$ sees only inputs at $t$ and earlier;
2. **dilation:** filters skip inputs at growing intervals, so coverage grows exponentially with depth;
3. **residual blocks:** deep stacks train reliably with skip connections, weight normalization and dropout.

The payoff is a long, explicitly controlled memory **without** an RNN's step-by-step computation: every time position in a layer can be computed in parallel. The reference architecture is Bai, Kolter and Koltun (2018), which built on WaveNet's dilated causal convolutions (van den Oord et al., 2016).

```text
causal conv → local past only
stack of causal convs → receptive field grows linearly with depth
dilations 1, 2, 4, 8, … → receptive field grows exponentially
residual blocks → deep, trainable
```

## 1. What a TCN is

A TCN is a 1-D **fully convolutional** network: no dense layers over time, and every hidden layer has the same length $T$ as the input (left zero-padding keeps it that way). Why use one:

- it's a plain feed-forward network, so all of [Chapter 7](07_neural_network_foundations.md) applies (backprop, dropout, residual connections), with no BPTT;
- positions are computed in parallel, so training is much faster than an RNN when $T$ is large;
- the memory length is a design choice, not something learned (or lost) through gates;
- it often matches or beats LSTMs on sequence benchmarks.

## 2. Causal convolution

The filters from [Chapter 3](03_filters_smoothing_and_decomposition.md) were two-sided: $F(t)=\sum_{i=-K}^{K}f(i)\,x_{t+i}$, a width-$(2K+1)$ window that looks into the future. That's fine for smoothing a finished series and cheating for forecasting. A **causal** convolution only looks back:

```math
F(t)=(x*f)(t)=\sum_{i=0}^{k-1}f(i)\,x_{t-i},
```

where $k$ is the **filter width** (number of taps). So $\hat y_{t+1}$ can depend on $x_0,\ldots,x_t$ and nothing later.

**Example.** The filter $f=(0.1,-0.8,0.3)$ applied with stride 1 gives $F(t)=0.1x_t-0.8x_{t-1}+0.3x_{t-2}$. On $x=(1,2,3,4,5,6)$ with two zeros padded on the left:

```text
padded input   0    0    1    2    3    4    5    6
output              0.1 -0.6 -1.0 -1.4 -1.8 -2.2
```

Two properties carry over from image CNNs:

- **local connectivity:** each output depends on a small neighborhood;
- **weight sharing:** the same 3 weights are used at every position. A dense layer over the same window would need a separate weight for every input–output pair.

**Padding vs causality.** These solve different problems. Causality decides *which* inputs an output may use; left padding with $(k-1)d$ zeros just gives the first few outputs something to read, so the layer length stays $T$. Those early outputs are computed partly from zeros, so it's common to exclude the first receptive-field-worth of positions from the loss.

**Channels.** A layer has many filters. Each produces its own output channel, and the next layer convolves across all channels jointly (a filter there has shape $k\times C_{\text{in}}$).

## 3. Receptive field

The **receptive field** $R$ is how many input steps can influence one output. With $L$ stacked stride-1 layers of width $k$, each layer adds $k-1$ steps of history:

```math
R=1+L(k-1).
```

Growth is linear: a width-3 filter needs 50 layers to see 101 steps. Getting a long memory from ordinary convolutions means many layers or very wide filters, both expensive.

![TCN overview](../assets/clean_diagrams/tcn_overview.png)

*Causal paths only point forward in time; dilation doubles the spacing at each layer; blocks are wrapped in residual connections.*

## 4. Dilated causal convolution

A **dilated** convolution with dilation factor $d$ spaces its taps $d$ steps apart:

```math
F(t)=(x*_df)(t)=\sum_{i=0}^{k-1}f(i)\,x_{t-d\,i},
```

so with $d=4$ the taps are $x_t,x_{t-4},x_{t-8},\ldots$. It's equivalent to a wider filter with zeros at the skipped positions, and $d=1$ is ordinary convolution. **Dilation isn't stride:** stride skips *outputs* (and shortens the sequence); dilation skips *inputs* and still produces an output at every step.

With dilations $d_1,\ldots,d_L$, each layer adds $(k-1)d_\ell$ steps of history:

```math
R=1+(k-1)\sum_{\ell=1}^{L}d_\ell .
```

Doubling the dilation at each layer, $d=1,2,4,\ldots,2^{L-1}$, gives $R=1+(k-1)(2^L-1)$: **exponential** growth. With $k=2$ and $d=1,2,\ldots,512$ (10 layers), $R=1+1023=1024=2^{10}$.

| Stack (width $k=3$, 4 layers) | $R$ |
|---|---:|
| dilations 1, 1, 1, 1 | 9 |
| dilations 1, 2, 4, 8 | 31 |

```python
def receptive_field(k, dilations, convs_per_block=1):
    return 1 + convs_per_block * (k - 1) * sum(dilations)

print(receptive_field(2, [2**i for i in range(10)]))                  # 1024
print(receptive_field(3, [1, 2, 4, 8], convs_per_block=2))            # 61 (Bai et al. block)
```

**Why not one filter of width 1024?** A dilation stack reaches the same span with $10\times k$ weights per channel instead of 1024, and it composes ten **nonlinear** transformations instead of one linear one. It's cheaper and more expressive.

**Sizing it.** Pick $R$ to cover the longest dependency the data actually has, e.g. at least one seasonal period plus the forecast horizon. To reach a target like "50 steps of history," choose $k$ and the number of dilation levels so that $1+(k-1)(2^L-1)\ge50$, e.g. $k=3,L=5$ gives 63.

To go further, **stack blocks**: repeat the $1,2,4,\ldots$ dilation cycle in several blocks. Each block adds its own span, and the sum formula still applies.

## 5. The residual block

In the Bai et al. TCN, each block with width $k$ and dilation $d$ is:

```text
input ─┬─► dilated causal conv → weight norm → ReLU → dropout
       │   → dilated causal conv → weight norm → ReLU → dropout ─┐
       └─► (1×1 conv if channel counts differ) ─────────────────(+)─► output
```

The output is $\mathbf x+\mathcal F(\mathbf x)$: the branch learns a **correction** to its input, and the identity path gives gradients a direct route through deep stacks (the ResNet idea; [Chapter 10](10_representation_learning.md) covers residual learning in more detail). The $1\times1$ convolution is a per-time-step linear map that matches channel counts so the sum is defined. Since each block has two convolutions, its receptive-field contribution is $2(k-1)d$.

## 6. Training and forecasting

Training works like any feed-forward network: backprop, Adam, dropout, and early stopping on a chronological validation block ([Chapter 8, §2](08_rnn_lstm_gru_and_seq2seq.md)).

For horizon $H>1$ there are two options:

- **recursive:** predict $\hat y_{t+1}$, append it to the input, predict $\hat y_{t+2}$, and so on. Simple, but errors compound;
- **direct / multi-output:** let the last layer emit $H$ channels, one per horizon, from the representation at time $t$. One forward pass and no feedback errors.

## 7. How it compares

On benchmarks such as **sequential MNIST** (an image read one pixel at a time, 784 steps) and **permuted MNIST** (the same with a fixed random pixel order, which destroys locality and forces long-range memory), Bai et al. report a generic TCN matching or beating LSTMs and GRUs of similar size. That's strong evidence for the approach, though not a guarantee for every dataset.

| | RNN / LSTM | TCN |
|---|---|---|
| temporal mechanism | recurrent hidden state | causal (dilated) convolution |
| computation over positions | sequential, $t$ waits for $t-1$ | parallel within a layer (layers still run in order) |
| memory | learned, in principle unbounded, in practice fades | exactly $R$ steps, no more |
| gradient path | through $T$ steps (vanish/explode) | through depth $\approx\log R$ layers + residuals |
| streaming inference | $O(1)$ per new step | recompute over the window (or cache per layer) |
| variable-length input | natural | natural (fully convolutional) |

**Ways to enlarge $R$:** more dilation levels, larger dilation factors, wider filters, or more blocks.

## 8. Common confusions

- **Causality vs padding:** which inputs are allowed vs keeping the length at $T$.
- **Filter width vs max lag:** width $k$ means lags $0..k-1$. Watch whether a formula uses $k$ or $K=k-1$.
- **Receptive field vs parameter count:** a dilated stack has a huge $R$ with few parameters.
- **Dilation vs stride:** spaced taps with full-length output vs skipped outputs.
- **Residual vs recurrent connection:** a residual skips *layers* at the same time step; a recurrent connection carries state across *time*.

## 9. Questions and answers

<details><summary>Should dilation reset at the start of each block?</summary>

Yes, in the standard designs (WaveNet, Bai et al.). Each block cycles $1,2,4,\ldots$ again, so fine-grained and coarse patterns are both reprocessed at every depth.
</details>

<details><summary>Is weight normalization necessary?</summary>

No. It reparameterizes $\mathbf w=g\,\mathbf v/\|\mathbf v\|$ to stabilize training. Batch norm is awkward with causal padding and small batches, and layer norm or no normalization with a good initialization often works just as well.
</details>

<details><summary>What if the receptive field is shorter than the true dependency?</summary>

The model cannot see the relevant past, no matter how long it trains. Symptoms are residual autocorrelation at the missing lag and a validation loss that stops improving. Add a dilation level, or give lagged or seasonal features as extra input channels.
</details>

<details><summary>What if it is far longer than needed?</summary>

The early outputs are dominated by padding, there are more parameters than needed, and there's some overfitting risk. Trim it to about 1.5× the longest real dependency.
</details>

<details><summary>How does a TCN compare with a transformer for forecasting?</summary>

Both are parallel over time. A TCN has a fixed, local-to-global inductive bias and linear cost in $T$. Attention can link any two positions directly but costs $O(T^2)$ and needs more data. For moderate data and clear local structure, TCNs are a strong, cheap baseline (Chapter 11).
</details>

---

[← Previous: RNNs, LSTMs and Seq2Seq](08_rnn_lstm_gru_and_seq2seq.md) · [Course map](../course_map.md) · [Next: Representation Learning →](10_representation_learning.md)
