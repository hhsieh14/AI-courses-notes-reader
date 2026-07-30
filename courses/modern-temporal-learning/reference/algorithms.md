# Algorithm Index

## ARIMA/SARIMAX model-refinement workflow

### Input

A time series, optionally with exogenous attributes and a known or
candidate seasonal period.

### Output

A candidate ARIMA or SARIMAX model whose residuals contain little
remaining systematic temporal structure.

### Procedure reconstructed from the notes

1. Plot the original series.
2. Look for changing mean, trend, and seasonality.
3. Apply regular or seasonal differencing when needed.
4. Inspect the autocorrelation and partial-autocorrelation plots.
5. Select small candidate AR and MA orders.
6. Fit the model.
7. Inspect training and test predictions.
8. Inspect whether residuals are centered around zero.
9. Inspect residual autocorrelation.
10. Inspect the residual histogram and normal Q-Q plot.
11. Add or remove terms and refit when structure remains.

### Course example

- ARIMA $(1,0,0)$: systematic underprediction because the trend was not
  differenced.
- ARIMA $(1,1,0)$: forecast improves, but a lag-one residual correlation
  remains.
- Proposed next model: ARIMA $(1,1,1)$.

**Sources:** CSE598MTL.pdf, pp. 11-13

---

## EWMA parameter-selection workflow

### Input

A time series and candidate values of $\lambda$.

### Output

A smoothing parameter selected by forecast-error performance.

### Procedure reconstructed from the notes

1. Select an initial value $y_0$.
2. Compute:

   ```math
   y_t=\lambda x_t+(1-\lambda)y_{t-1}.
   ```

3. Use the previous smoothed value as the next short-horizon forecast.
4. Compute MSE, MAE, or MAPE.
5. Repeat for candidate values of $\lambda$.
6. Select the parameter with the preferred error measure.

**Source:** CSE598MTL.pdf, p. 15

---

## STL component-tuning workflow

### Input

A time series, a trend-window size $w_t$, and a seasonal-window size
$w_s$.

### Output

Estimated trend, seasonal, and residual components.

### Procedure at the level presented in the notes

1. Select a local neighborhood around each time point.
2. Fit a locally weighted trend estimate.
3. Fit a locally weighted seasonal estimate.
4. Compute the residual from the observed value and estimated components.
5. Reduce $w_t$ or $w_s$ to allow faster component changes.
6. Increase $w_t$ or $w_s$ to enforce more stable components.
7. Inspect whether anomalies are being absorbed into trend or seasonality.

**Source:** CSE598MTL.pdf, p. 18

---

## Causal-convolution layer

### Input

A temporal sequence, a set of filters, and left padding.

### Output

One or more length-preserving feature channels.

### Procedure

1. Pad the beginning of the sequence.
2. At output time $t$, select only $x_t,x_{t-1},\ldots$.
3. Apply the same filter weights at every time position.
4. Repeat for every filter to produce multiple channels.
5. Pass all channels to the next layer.

**Source:** CSE598MTL.pdf, p. 79

---

## Dilated TCN construction

### Input

Temporal sequence, filter width, dilation schedule, channel counts, and
number of blocks.

### Output

A temporal network with a controlled receptive field.

### Procedure reconstructed from the notes

1. Begin with a causal convolution.
2. Use dilation $d=1$ in the first layer.
3. Increase dilation across layers, commonly
   $1,2,4,8,\ldots$.
4. Apply nonlinear activation and regularization.
5. Add residual or projected identity connections.
6. Stack more dilation blocks when a larger receptive field is required.
7. Produce outputs at every time point or at selected final positions.

**Sources:** CSE598MTL.pdf, pp. 81-82

---

## Recursive multi-horizon TCN forecast

### Input

A fitted TCN, observed input window, and horizon $H$.

### Output

Predictions through horizon $H$.

### Procedure described on page 81

1. Predict the first future value.
2. Append or substitute that predicted value into the next model input.
3. Shift the temporal input window.
4. Predict the next future value.
5. Repeat until $H$ values are produced.

**Source:** CSE598MTL.pdf, p. 81

---

## Temporal autoencoder workflow

### Input

A collection of temporal sequences.

### Output

A low-dimensional representation for each sequence.

### Procedure

1. Choose an RNN- or LSTM-based encoder.
2. Encode each time series into a bottleneck state $h_i$.
3. Decode the bottleneck into a reconstructed sequence.
4. Minimize MSE, SSE, or MAE reconstruction loss.
5. Retain the trained encoder.
6. Discard the decoder for downstream tasks.
7. Use $h_i$ for clustering, anomaly detection, or supervised learning.
8. Evaluate downstream usefulness separately from reconstruction error.

**Source:** CSE598MTL.pdf, p. 86

---

## Minibatch contrastive-learning workflow

### Input

A minibatch of $K$ original instances and an augmentation policy.

### Output

A trained encoder representation.

### Procedure

1. Create two transformed views of each original instance, producing
   $2K$ examples.
2. Encode every transformed example:

   ```math
   h_i=f(x_i).
   ```

3. Apply the projection head:

   ```math
   z_i=g(h_i).
   ```

4. Treat the two views of the same original instance as a positive pair.
5. Treat other minibatch views as negatives.
6. Compute cosine similarities.
7. Apply the temperature-scaled contrastive loss.
8. Sum losses over anchors.
9. Update encoder and head parameters.
10. After pretraining, discard the head and retain the encoder.

**Sources:** CSE598MTL.pdf, pp. 88-90

---

## Artificial-contrast anomaly workflow

### Input

Actual temporal sequences and a method for generating artificial similar
sequences.

### Output

A classifier-derived score for a new series.

### Procedure shown on page 91

1. Label actual series as class 0.
2. Label artificial series as class 1.
3. Train a supervised classifier.
4. Calculate a class-probability score for a test series.

### Review warning

The slide calls class-0 probability an anomaly measure, although its own
label definition makes class-0 probability more naturally an actual-data
or normality score. The score direction must be resolved before
implementation.

**Source:** CSE598MTL.pdf, p. 91

---

## Encoder self-attention

### Input

Embedded sequence elements with positional information.

### Output

Contextualized representation for every sequence position.

### Procedure

1. Transform every element into query, key, and value vectors.
2. For each query, compute scaled dot products with all keys.
3. Apply softmax over candidate positions.
4. Form a weighted sum of value vectors.
5. Repeat independently for each attention head.
6. Concatenate head outputs.
7. Apply the output projection.
8. Add the residual input and normalize.
9. Apply the position-wise feed-forward layer.
10. Add the residual and normalize again.

**Sources:** CSE598MTL.pdf, pp. 93-95 and 99

---

## Autoregressive transformer decoding

### Input

Final encoder representation and beginning-of-sequence token.

### Output

Generated output sequence ending in an EOS token.

### Procedure

1. Embed the known decoder prefix.
2. Add positional encoding.
3. Apply masked decoder self-attention.
4. Use decoder representations as queries over encoder keys and values.
5. Apply decoder feed-forward layers.
6. Transform the final decoder representation to output logits.
7. Apply softmax and select or sample the next element.
8. Append the generated element to the decoder input.
9. Repeat until EOS or another stopping rule.

**Sources:** CSE598MTL.pdf, pp. 96-98

---

## Teacher-forced transformer training

### Input

Source sequence and complete target sequence.

### Output

Parallel next-element training losses.

### Procedure

1. Shift the target sequence and prepend BOS.
2. Input the known shifted target sequence to the decoder.
3. Mask future target positions.
4. Compute decoder positions in parallel within each layer.
5. Predict the next target at each position.
6. Compute the sequence loss against known targets.
7. Update encoder and decoder parameters.

**Source:** CSE598MTL.pdf, p. 98

---

## Time-series transformer adaptation checklist

### Input

A temporal task and sequence characteristics.

### Output

A transformer architecture suitable for the temporal data.

### Questions reconstructed from page 100

1. How should time position be encoded?
2. Should timestamps or intervals be embedded explicitly?
3. Is full attention computationally feasible?
4. Should attention be sparse or low-rank?
5. Does the application require forecasting, anomaly detection, or
   classification?
6. Are several temporal resolutions required?
7. Should the model be encoder-only, decoder-only, or encoder-decoder?
8. How should numerical multivariate outputs be represented?

**Source:** CSE598MTL.pdf, p. 100
