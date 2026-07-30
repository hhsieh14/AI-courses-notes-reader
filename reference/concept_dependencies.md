# Concept Dependencies

```mermaid
flowchart TD
    A[Temporal data representation] --> B[Wavelet basis]
    A --> C[Covariance structure]
    C --> D[PCA eigenvectors]
    D --> E[Principal-component scores]
    E --> F[Dimensionality reduction]
    E --> G[Monitoring and anomaly distance]

    F --> H[Supervised predictive model]
    H --> I[Bias-variance tradeoff]
    I --> J[Penalized regression]
    J --> K[Ridge]
    J --> L[Lasso]

    B --> M[Wavelet coefficients]
    L --> N[Orthonormal wavelet lasso]
    M --> N
    N --> O[Soft thresholding]
```

## Key dependency notes

- Covariance and eigendecomposition are prerequisites for PCA.
- PCA scores can reduce representation size before supervised learning.
- Bias-variance motivates regularization.
- Ridge and lasso differ through their penalty geometry.
- Soft wavelet thresholding is derived from lasso only in the orthonormal
  basis case shown on page 44.

**Sources:** CSE598MTL.pdf, pp. 21-44

## Temporal-convolution branch

```mermaid
flowchart TD
    A[Temporal convolution] --> B[Causal convolution]
    B --> C[Stacked causal layers]
    C --> D[Receptive field]
    D --> E[Dilated causal convolution]
    E --> F[Exponential dilation schedule]
    F --> G[Residual TCN block]
    G --> H[Parallel temporal prediction]
    G --> I[Recursive horizon forecasting]
```

## Representation-learning branch

```mermaid
flowchart TD
    A[High-dimensional temporal data] --> B[PCA]
    A --> C[Kernel PCA]
    A --> D[Temporal autoencoder]
    A --> E[Self-supervised pretext task]

    D --> F[Encoder bottleneck h]
    E --> G[Contrastive pairs]
    G --> H[Encoder h]
    H --> I[Projection head z]
    I --> J[Temperature-scaled contrastive loss]

    B --> K[Downstream task]
    C --> K
    F --> K
    H --> K

    K --> L[Clustering]
    K --> M[Anomaly detection]
    K --> N[Supervised classification]
```

## Transformer branch

```mermaid
flowchart TD
    A[Sequence embeddings] --> B[Positional encoding]
    B --> C[Queries, keys, values]
    C --> D[Scaled dot-product attention]
    D --> E[Multi-head attention]
    E --> F[Residual + normalization]
    F --> G[Position-wise feed-forward]
    G --> H[Encoder representation]

    I[Shifted target embeddings] --> J[Masked self-attention]
    J --> K[Encoder-decoder attention]
    H --> K
    K --> L[Decoder feed-forward]
    L --> M[Linear output + softmax]

    M --> N[Teacher-forced training]
    M --> O[Autoregressive inference]
    E --> P[Low-rank/sparse time-series attention]
    E --> Q[Hierarchical multiresolution transformer]
```
