# Equation Reference

Every key formula in the notes on one page, grouped by chapter. Each section links to the chapter where the formula is derived and explained.

## Stationarity and classical models ([Ch 2](../notes/02_classical_time_series_models.md))

```math
\begin{aligned}
&\text{strict stationarity:} && F(x_{t_1},\ldots,x_{t_n})=F(x_{t_1+\tau},\ldots,x_{t_n+\tau})\\
&\text{weak stationarity:} && \mathbb E[x_t]=\mu,\quad \mathrm{Cov}(x_t,x_{t+k})=\gamma(k)\\
&\text{autocorrelation:} && \rho_k=\frac{\gamma(k)}{\gamma(0)},\qquad r_k=\frac{\sum_{t=k+1}^{n}(x_t-\bar x)(x_{t-k}-\bar x)}{\sum_{t=1}^{n}(x_t-\bar x)^2}
\end{aligned}
```

**Stirred tank.** Time constant $T=V/f$, step response $x_t=w_0(1-e^{-t/T})$, sampled recurrence $x_t=b\,w_t+(1-b)x_{t-1}$ with $b=1-e^{-\Delta t/T}$, lag-1 correlation $\rho=e^{-\Delta t/T}$.

```math
\begin{aligned}
&\text{AR}(1): && x_t=c+\phi x_{t-1}+e_t, && \mathbb E x_t=\tfrac{c}{1-\phi},\;\; \mathrm{SD}(x_t)=\tfrac{\sigma}{\sqrt{1-\phi^2}},\;\; \rho_k=\phi^k\\
&\text{AR}(p): && x_t=c+\textstyle\sum_{i=1}^p\phi_ix_{t-i}+e_t\\
&\text{MA}(1): && x_t=\mu+e_t-\theta e_{t-1}, && \mathrm{Var}(x_t)=\sigma^2(1+\theta^2),\;\; \rho_1=\tfrac{-\theta}{1+\theta^2},\;\; \rho_{k>1}=0\\
&\text{MA}(q): && x_t=\mu+e_t-\textstyle\sum_{i=1}^q\theta_ie_{t-i}, && \rho_{k>q}=0\\
&\text{ARIMA}(p,d,q): && \Phi(B)(1-B)^dx_t=\Theta(B)e_t, && Bx_t=x_{t-1}\\
&\text{seasonal difference:} && (1-B^{s})x_t=x_t-x_{t-s}, && \text{both: }(1-B)(1-B^{12})x_t\\
&\text{local level:} && x_t=\mu_t+e_t,\;\;\mu_t=\mu_{t-1}+\delta_t\;\Rightarrow\;\Delta x_t=\epsilon_t-\theta\epsilon_{t-1}\;\;(\text{IMA}(1,1))
\end{aligned}
```

## Filters and smoothing ([Ch 3](../notes/03_filters_smoothing_and_decomposition.md))

```math
\begin{aligned}
&\text{linear filter:} && y_t=\textstyle\sum_k\beta_kx_{t-k}; \qquad \text{moving average: } y_t=\tfrac1K\sum_{k=0}^{K-1}x_{t-k}\\
&\text{EWMA:} && y_t=\lambda x_t+(1-\lambda)y_{t-1}=\lambda\textstyle\sum_{k\ge0}(1-\lambda)^kx_{t-k}, \qquad \hat x_{t+1}=y_t\\
&\text{EWMA lag on a trend:} && \mathbb E[y_t]=\beta_0+\beta_1t-\tfrac{1-\lambda}{\lambda}\beta_1\\
&\text{Holt:} && L_t=\lambda_1x_t+(1-\lambda_1)(L_{t-1}+b_{t-1}),\;\; b_t=\lambda_2(L_t-L_{t-1})+(1-\lambda_2)b_{t-1},\;\; \hat x_{t+h}=L_t+hb_t\\
&\text{Holt–Winters (additive):} && L_t=\lambda_1(x_t-S_{t-s})+\ldots,\;\; S_t=\lambda_3(x_t-L_t)+(1-\lambda_3)S_{t-s},\;\; \hat x_{t+h}=L_t+hb_t+S_{t+h-s}\\
&\text{model selection:} && \mathrm{AIC}=-2\log L+2p,\qquad \mathrm{BIC}=-2\log L+p\log T\\
&\text{errors:} && \mathrm{MSE}=\tfrac1T\textstyle\sum(x_t-\hat x_t)^2,\;\; \mathrm{MAE}=\tfrac1T\sum\lvert x_t-\hat x_t\rvert,\;\; \mathrm{MAPE}=\tfrac{100}{T}\sum\left\lvert\tfrac{x_t-\hat x_t}{x_t}\right\rvert\\
&\text{transfer function:} && \Phi(B)\,y_t=\Psi(B)\,x_t+\Theta(B)\,e_t
\end{aligned}
```

## Wavelets ([Ch 4](../notes/04_wavelets.md))

```math
\begin{aligned}
&\text{basis expansion:} && \mathbf y=W\boldsymbol\beta,\qquad \hat{\boldsymbol\beta}=W^\top\mathbf y\;\;(\text{orthonormal }W)\\
&\text{Haar step:} && a=\tfrac{x_{2i}+x_{2i+1}}{\sqrt2},\quad d=\tfrac{x_{2i}-x_{2i+1}}{\sqrt2};\qquad x_{2i}=\tfrac{a+d}{\sqrt2},\quad x_{2i+1}=\tfrac{a-d}{\sqrt2}\\
&\text{coefficient count:} && 2^{M-K}+\textstyle\sum_{j=1}^{K}2^{M-j}=2^M\\
&\text{thresholds:} && H_\lambda(w)=w\,\mathbf 1\{\lvert w\rvert>\lambda\},\qquad S_\lambda(w)=\mathrm{sgn}(w)\,(\lvert w\rvert-\lambda)_+,\qquad \lambda_{\text{univ}}=\hat\sigma\sqrt{2\log N}
\end{aligned}
```

## PCA and regularization ([Ch 5](../notes/05_pca_and_regularization.md))

```math
\begin{aligned}
&\text{PCA:} && \Sigma=VDV^\top,\quad z_m=\mathbf v_m^\top\mathbf x,\quad \mathrm{Var}(z_m)=\lambda_m,\quad \textstyle\sum_m\lambda_m=\mathrm{tr}\,\Sigma\\
&\text{Mahalanobis / }T^2: && d_M^2=(\mathbf x-\boldsymbol\mu)^\top\Sigma^{-1}(\mathbf x-\boldsymbol\mu)=\textstyle\sum_m z_m^2/\lambda_m\\
&\text{bias–variance:} && \mathbb E(\hat f-f)^2=\mathrm{Var}(\hat f)+\mathrm{Bias}(\hat f)^2\;(+\sigma^2\text{ for a new }y)\\
&\text{ridge:} && \hat{\boldsymbol\beta}=(X^\top X+\lambda I)^{-1}X^\top\mathbf y,\qquad X\hat{\boldsymbol\beta}=\textstyle\sum_m\frac{d_m^2}{d_m^2+\lambda}\mathbf u_m\mathbf u_m^\top\mathbf y\\
&\text{lasso:} && \min_{\boldsymbol\beta}\lVert\mathbf y-X\boldsymbol\beta\rVert_2^2+\lambda\lVert\boldsymbol\beta\rVert_1;\qquad \text{orthonormal }X:\;\beta_j^\star=\mathrm{sgn}(b_j)(\lvert b_j\rvert-\tfrac\lambda2)_+
\end{aligned}
```

## Markov models, HMMs and EM ([Ch 6](../notes/06_markov_models_hmm_and_em.md))

```math
\begin{aligned}
&\text{Markov chain:} && a_{ij}=P(x_{t+1}=j\mid x_t=i),\qquad \boldsymbol\pi^\top(t)=\boldsymbol\pi^\top A^t,\qquad \boldsymbol\pi^{\star\top}=\boldsymbol\pi^{\star\top}A\\
&\text{gambler's ruin:} && P_i=\frac{1-(q/p)^i}{1-(q/p)^N}\;(p\ne q),\qquad P_i=\tfrac iN\;(p=q)\\
&\text{forward:} && \alpha_1(i)=\pi_iF_i(y_1),\quad \alpha_{t+1}(j)=\big[\textstyle\sum_i\alpha_t(i)a_{ij}\big]F_j(y_{t+1}),\quad P(\mathbf y)=\sum_i\alpha_T(i)\\
&\text{backward:} && \beta_T(i)=1,\quad \beta_t(i)=\textstyle\sum_ja_{ij}F_j(y_{t+1})\beta_{t+1}(j)\\
&\text{posteriors:} && \gamma_t(i)=\frac{\alpha_t(i)\beta_t(i)}{P(\mathbf y)},\qquad \xi_t(i,j)=\frac{\alpha_t(i)a_{ij}F_j(y_{t+1})\beta_{t+1}(j)}{P(\mathbf y)}\\
&\text{Viterbi:} && \delta_{t+1}(j)=\max_i[\delta_t(i)a_{ij}]\,F_j(y_{t+1}),\qquad \psi_{t+1}(j)=\arg\max_i\delta_t(i)a_{ij}\\
&\text{Baum–Welch:} && \hat\pi_i=\gamma_1(i),\quad \hat a_{ij}=\frac{\sum_{t<T}\xi_t(i,j)}{\sum_{t<T}\gamma_t(i)},\quad \hat p_i(k)=\frac{\sum_t\mathbf 1(y_t=k)\gamma_t(i)}{\sum_t\gamma_t(i)}\\
&\text{GMM E-step:} && \gamma(z_{nk})=\frac{\pi_k\mathcal N(\mathbf x_n\mid\mu_k,\Sigma_k)}{\sum_j\pi_j\mathcal N(\mathbf x_n\mid\mu_j,\Sigma_j)}\\
&\text{GMM M-step:} && N_k=\textstyle\sum_n\gamma(z_{nk}),\;\; \mu_k=\frac1{N_k}\sum_n\gamma(z_{nk})\mathbf x_n,\;\; \Sigma_k=\frac1{N_k}\sum_n\gamma(z_{nk})(\mathbf x_n-\mu_k)(\mathbf x_n-\mu_k)^\top,\;\; \pi_k=\frac{N_k}{N}
\end{aligned}
```

## Neural networks ([Ch 7](../notes/07_neural_network_foundations.md))

```math
\begin{aligned}
&\text{forward:} && \mathbf h=\mathrm{act}(W_1\mathbf x+\mathbf b_1),\quad \mathbf o=W_2\mathbf h+\mathbf b_2,\quad \hat{\mathbf y}=\bar g(\mathbf o)\\
&\text{softmax:} && \bar g_k(\mathbf o)=e^{o_k}/\textstyle\sum_je^{o_j}\\
&\text{cross-entropy:} && L=-\textstyle\sum_i\sum_ky_{ik}\log\bar g_k(\mathbf x_i;\theta)\;=\;-\log\prod_i\prod_kp_{ik}^{y_{ik}}\\
&\text{Bernoulli MLE:} && \hat p=\bar y\\
&\text{sigmoid gradient:} && \partial\sigma(\mathbf w^\top\mathbf x)/\partial\mathbf w=\sigma(1-\sigma)\,\mathbf x,\qquad 0<\sigma(1-\sigma)\le\tfrac14\\
&\text{minibatch SGD:} && \theta\leftarrow\theta-\eta\textstyle\sum_{i\in b}\nabla L_i(\theta)\\
&\text{weight decay:} && L^*=L+\lambda\textstyle\sum_jw_j^2,\qquad \partial L^*/\partial w_j=\partial L/\partial w_j+2\lambda w_j\\
&\text{momentum:} && \mathbf v\leftarrow\beta\mathbf v+\nabla L,\qquad \theta\leftarrow\theta-\eta\mathbf v\\
&\text{parameters:} && \textstyle\sum_\ell(n_{\ell-1}n_\ell+n_\ell)
\end{aligned}
```

## RNN, LSTM, GRU ([Ch 8](../notes/08_rnn_lstm_gru_and_seq2seq.md))

```math
\begin{aligned}
&\text{RNN:} && \mathbf h_t=\phi(U\mathbf h_{t-1}+W_1\mathbf x_t+\mathbf b_1),\quad \mathbf o_t=W_2\mathbf h_t+\mathbf b_2,\quad \hat{\mathbf y}_t=g(\mathbf o_t)\\
&\text{gradient chain:} && \frac{\partial\mathbf h_t}{\partial\mathbf h_k}=\prod_{j=k+1}^{t}\mathrm{diag}(\phi'(\mathbf a_j))U,\qquad \Big\lVert\frac{\partial\mathbf h_t}{\partial\mathbf h_k}\Big\rVert\le(\gamma\lVert U\rVert)^{t-k}\\
&\text{LSTM gates:} && \mathbf f_t,\mathbf i_t,\mathbf o_t=\sigma(W_{\{f,i,o\}}[\mathbf x_t;\mathbf h_{t-1}]+\mathbf b_{\{f,i,o\}}),\quad \tilde{\mathbf c}_t=\tanh(W_c[\mathbf x_t;\mathbf h_{t-1}]+\mathbf b_c)\\
&\text{LSTM state:} && \mathbf c_t=\mathbf f_t\odot\mathbf c_{t-1}+\mathbf i_t\odot\tilde{\mathbf c}_t,\qquad \mathbf h_t=\mathbf o_t\odot\tanh(\mathbf c_t)\\
&\text{GRU:} && \mathbf r_t,\mathbf z_t=\sigma(\cdot),\quad \tilde{\mathbf h}_t=\tanh(W[\mathbf x_t;\mathbf r_t\odot\mathbf h_{t-1}]+\mathbf b),\quad \mathbf h_t=(1-\mathbf z_t)\odot\mathbf h_{t-1}+\mathbf z_t\odot\tilde{\mathbf h}_t\\
&\text{Bahdanau attention:} && \alpha_{t,j}=\mathrm{softmax}_j\,\mathrm{score}(\mathbf s_{t-1},\mathbf h_j),\qquad \mathbf c_t=\textstyle\sum_j\alpha_{t,j}\mathbf h_j
\end{aligned}
```

## TCN ([Ch 9](../notes/09_temporal_convolutional_networks.md))

```math
\begin{aligned}
&\text{two-sided filter:} && F(t)=\textstyle\sum_{i=-K}^{K}f(i)x_{t+i}\\
&\text{causal (width }k): && F(t)=\textstyle\sum_{i=0}^{k-1}f(i)x_{t-i}\\
&\text{dilated causal:} && F(t)=\textstyle\sum_{i=0}^{k-1}f(i)x_{t-d\,i}\\
&\text{receptive field:} && R=1+L(k-1)\;\;(\text{no dilation}),\qquad R=1+(k-1)\textstyle\sum_\ell d_\ell,\qquad d_\ell=2^{\ell-1}\Rightarrow R=1+(k-1)(2^L-1)\\
&\text{residual block:} && \mathbf y=\mathrm{ReLU}\big(\mathcal F(\mathbf x)+\mathcal I(\mathbf x)\big),\quad \mathcal I=\text{identity or }1\times1\text{ conv}
\end{aligned}
```

## Representation learning ([Ch 10](../notes/10_representation_learning.md))

```math
\begin{aligned}
&\text{kernels:} && K=\phi(\mathbf x_i)^\top\phi(\mathbf x_j);\quad (1+\mathbf x_i^\top\mathbf x_j)^d;\quad \exp(-\gamma\lVert\mathbf x_i-\mathbf x_j\rVert^2);\quad \tanh(\beta\mathbf x_i^\top\mathbf x_j-\delta)\\
&\text{centered kernel:} && \tilde K=K-\mathbf 1_NK-K\mathbf 1_N+\mathbf 1_NK\mathbf 1_N\\
&\text{autoencoder:} && \mathbf h=\phi(W_1\mathbf x+\mathbf b_1),\quad \hat{\mathbf x}=g(W_2\mathbf h+\mathbf b_2),\quad L=\textstyle\sum_i\lVert\mathbf x_i-\hat{\mathbf x}_i\rVert_2^2\\
&\text{cosine similarity:} && \mathrm{sim}(\mathbf u,\mathbf v)=\mathbf u^\top\mathbf v/(\lVert\mathbf u\rVert\lVert\mathbf v\rVert)\\
&\text{NT-Xent:} && \ell_{i,j}=-\log\frac{\exp(\mathrm{sim}(\mathbf z_i,\mathbf z_j)/\tau)}{\sum_{k\ne i}\exp(\mathrm{sim}(\mathbf z_i,\mathbf z_k)/\tau)}\\
&\text{triplet:} && \max(0,\;\mathbf u^\top\mathbf v^--\mathbf u^\top\mathbf v^++m)\\
&\text{mutual information:} && I(X;Y)=H(X)-H(X\mid Y)=H(X)+H(Y)-H(X,Y),\qquad H(X)=-\textstyle\sum_xp(x)\log p(x)\\
&\text{InfoNCE bound:} && I\ge\log N-\mathcal L_{\text{InfoNCE}}\\
&\text{artificial-contrast score:} && P(y{=}1\mid\mathbf x)=1-\frac{p_{\text{real}}(\mathbf x)}{p_{\text{real}}(\mathbf x)+p_{\text{ref}}(\mathbf x)}
\end{aligned}
```

## Transformers ([Ch 11](../notes/11_transformers.md))

```math
\begin{aligned}
&\text{projections:} && \mathbf q_i=W_Q\mathbf x_i,\quad \mathbf k_i=W_K\mathbf x_i,\quad \mathbf v_i=W_V\mathbf x_i\\
&\text{attention:} && w_{ij}=\mathrm{softmax}_j\Big(\frac{\mathbf q_i^\top\mathbf k_j}{\sqrt{d_k}}\Big),\qquad \mathbf z_i=\textstyle\sum_jw_{ij}\mathbf v_j,\qquad Z=\mathrm{softmax}\Big(\frac{QK^\top}{\sqrt{d_k}}+M\Big)V\\
&\text{multi-head:} && \mathbf z_i=W_O\,\mathrm{Concat}(\boldsymbol\zeta_i^{(1)},\ldots,\boldsymbol\zeta_i^{(H)}),\qquad d_k=d_v=d_{\text{model}}/H\\
&\text{FFN:} && \mathrm{FFN}(\mathbf x)=W_2\,\mathrm{ReLU}(W_1\mathbf x+\mathbf b_1)+\mathbf b_2\\
&\text{sublayer:} && \mathrm{LayerNorm}(\mathbf x+\mathrm{Sublayer}(\mathbf x))\;\;(\text{post-LN}),\qquad \mathbf x+\mathrm{Sublayer}(\mathrm{LayerNorm}(\mathbf x))\;\;(\text{pre-LN})\\
&\text{positional encoding:} && \mathrm{PE}(p,2i)=\sin(p/10000^{2i/d}),\qquad \mathrm{PE}(p,2i+1)=\cos(p/10000^{2i/d})\\
&\text{output:} && P(y_j\mid y_{<j},X)=\mathrm{softmax}(W_{\text{out}}\mathbf r_j+\mathbf b_{\text{out}})
\end{aligned}
```

## Differenced local-level model ([Ch 12](../notes/12_handwritten_appendix.md))

```math
\begin{aligned}
&d_t=x_t-x_{t-1}=\delta_t+e_t-e_{t-1},\qquad \mathrm{Var}(d_t)=\sigma_\delta^2+2\sigma_e^2\\
&\mathrm{Cov}(d_t,d_{t+1})=-\sigma_e^2,\qquad \mathrm{Cov}(d_t,d_{t+k})=0\;(k\ge2),\qquad \rho_1=-\frac{\sigma_e^2}{\sigma_\delta^2+2\sigma_e^2}\\
&\text{MA coefficient:} \quad \theta=\frac{(q+2)-\sqrt{q^2+4q}}{2},\quad q=\sigma_\delta^2/\sigma_e^2,\qquad \text{EWMA weight }\lambda=1-\theta
\end{aligned}
```
