# Concept Dependencies

What to understand before what. An arrow $A\to B$ means "$B$ uses $A$." If a chapter feels hard, walk its arrows backward.

## The whole course

```mermaid
flowchart TD
    subgraph Classical
        S[Stationarity & autocovariance] --> AR[AR / MA / ARIMA]
        AR --> DIFF[Differencing & IMA]
        DIFF --> EWMA[EWMA / Holt–Winters]
        S --> FFT[Fourier / spectrum]
        FFT --> WAV[Wavelets]
    end
    subgraph Statistical learning
        COV[Covariance & eigenvectors] --> PCA
        BV[Bias–variance] --> REG[Ridge / lasso]
        WAV --> THR[Wavelet thresholding]
        REG --> THR
        MLE[Maximum likelihood] --> EM[EM]
        MC[Markov chains] --> HMM[HMM: forward–backward, Viterbi]
        EM --> BW[Baum–Welch]
        HMM --> BW
    end
    subgraph Deep sequence models
        MLE --> CE[Cross-entropy]
        CE --> NN[Feed-forward nets + SGD]
        REG --> NN
        EWMA --> MOM[Momentum]
        MOM --> NN
        NN --> RNN[RNN + BPTT]
        HMM -.state idea.-> RNN
        RNN --> LSTM[LSTM / GRU]
        EWMA -.gated average.-> LSTM
        NN --> TCN[Causal & dilated convs]
        WAV -.multiresolution.-> TCN
        PCA --> AE[Autoencoders]
        NN --> AE
        LSTM --> AE
        AE --> CL[Contrastive learning]
        LSTM --> ATT[Encoder–decoder attention]
        ATT --> TR[Transformer]
        TCN --> TR
    end
```

## Branch details

### Filters, regularization and wavelets (Chapters 3–5)

```mermaid
flowchart TD
    A[Covariance matrix] --> B[PCA eigenvectors]
    B --> C[PC scores: reduction, T² / Q monitoring]
    C --> D[Supervised model on scores]
    D --> E[Bias–variance]
    E --> F[Ridge: shrink all directions]
    E --> G[Lasso: sparsity]
    H[Orthonormal wavelet basis] --> I[Wavelet coefficients]
    G --> J[Lasso in an orthonormal basis]
    I --> J
    J --> K[Soft thresholding]
```

Soft thresholding *equals* the lasso solution only when the basis is orthonormal, as with Haar or Daubechies wavelets. In a general basis the lasso has no closed form.

### Recurrent and convolutional (Chapters 8–9)

```mermaid
flowchart TD
    A[Shared weights over time] --> B[RNN hidden state]
    B --> C[BPTT]
    C --> D[Vanishing / exploding gradients]
    D --> E[Clipping, truncation]
    D --> F[Additive cell path: LSTM / GRU]
    F --> G[Encoder–decoder]
    G --> H[Bottleneck → attention]
    I[Causal convolution] --> J[Receptive field 1 + L(k−1)]
    J --> K[Dilation: 1 + (k−1)Σd]
    K --> L[Residual TCN block]
```

### Representation learning (Chapter 10)

```mermaid
flowchart TD
    A[High-dimensional series] --> B[PCA]
    B --> C[Kernel trick → kernel PCA]
    A --> D[Temporal autoencoder → bottleneck h]
    A --> E[Pretext task]
    E --> F[Positive / negative pairs]
    F --> G[Encoder h → projection head z]
    G --> H[NT-Xent with temperature τ]
    B & C & D & G --> I[Downstream: clustering, anomaly detection, classification]
```

### Transformer (Chapter 11)

```mermaid
flowchart TD
    A[Embeddings + positional encoding] --> B[Q, K, V projections]
    B --> C[Scaled dot-product attention]
    C --> D[Multi-head + W_O]
    D --> E[Add & norm → FFN → add & norm]
    E --> F[Encoder output]
    G[Shifted targets] --> H[Masked self-attention]
    H --> I[Cross-attention]
    F --> I
    I --> J[Linear + softmax]
    J --> K[Teacher forcing / autoregressive inference]
    C --> L[Sparse, low-rank, patched attention for long series]
```
