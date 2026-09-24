# 2. Stationarity and Classical Time-Series Models

This chapter builds one continuous idea:

```text
stable statistical behavior (stationarity)
    → temporal dependence (autocorrelation)
    → AR and MA models
    → differencing for non-stationary series
    → ARIMA / SARIMAX
    → residuals that should look like white noise
```

The goal isn't "does the model fit?" but **has the model captured the temporal structure so well that nothing predictable is left in the residuals?**

## 1. Stationarity: what has to stay the same?

A stationary series can wiggle up and down. Stationarity is a property of the **process that generates the data**, not of how flat the plot looks.

![Stationarity patterns](../assets/clean_diagrams/stationarity_patterns.png)

**Strict (strong) stationarity:** shifting all time indices by the same $\tau$ leaves every joint distribution unchanged,

```math
F(x_{t_1},\ldots,x_{t_n}) = F(x_{t_1+\tau},\ldots,x_{t_n+\tau}).
```

**Weak (second-order) stationarity:** only the first two moments are shift-invariant. The mean is constant, the variance is constant, and the covariance depends only on the lag,

```math
\mathrm{Cov}(x_i,x_j) = \mathrm{Cov}(x_{i+\tau},x_{j+\tau}) = \gamma(|i-j|).
```

| Process | Same distribution over time | Independent | Uncorrelated across time |
|---|:---:|:---:|:---:|
| IID | yes | yes | yes (when moments exist) |
| White noise | constant mean and variance | not required | yes |
| Weakly stationary | constant first two moments | not required | not necessarily |

White noise needn't be IID and needn't be Gaussian. It only has to be uncorrelated with constant mean and variance.

Why it matters: stationarity is what lets the past teach us about the future. If the distribution keeps changing, yesterday's statistics don't describe tomorrow.

## 2. Between IID and "anything goes"

```text
IID                         AR / MA / state-space                     no structure
no temporal dependence  ←   chosen, interpretable dependence   →   maximum flexibility, little guidance
```

Classical time-series models sit in the middle. A physical example shows why "current depends on previous" is natural.

## 3. Intuition: the stirred tank

A well-mixed tank of volume $V$ with flow rate $f$, input concentration $w_t$ and output concentration $x_t$ has time constant $T=V/f$. After a step change of size $w_0$ in the input, the output approaches it exponentially:

```math
x_t = w_0\left(1-e^{-t/T}\right), \qquad w_0 - x_t = w_0e^{-t/T}.
```

Sampled every $\Delta t$ this becomes

```math
x_t = b\,w_t + (1-b)\,x_{t-1}, \qquad b = 1-e^{-\Delta t/T},
```

an AR(1) model driven by the input. If $w_t$ is white noise, the lag-1 correlation is $\rho=e^{-\Delta t/T}$:

| $\Delta t/T$ | 3.0 | 2.0 | 1.0 | 0.5 | 0.25 | 0.10 |
|---:|---:|---:|---:|---:|---:|---:|
| $\rho=e^{-\Delta t/T}$ | 0.05 | 0.14 | 0.37 | 0.61 | 0.78 | 0.90 |

**Sampling faster makes neighbors more correlated**, because the system has less time to change between samples.

## 4. Autocorrelation

```math
\rho_k = \frac{\mathrm{Cov}(x_t,x_{t-k})}{\mathrm{Var}(x_t)}, \qquad
r_k = \frac{\sum_{t=k+1}^{n}(x_t-\bar{x})(x_{t-k}-\bar{x})}{\sum_{t=1}^{n}(x_t-\bar{x})^2}.
```

$r_k$ is the sample estimate. A **lag plot** of $(x_{t-k},x_t)$ shows the same thing visually: a tight diagonal cloud means strong dependence.

## 5. Autoregressive models

### AR(1)

```math
x_t = c + \phi x_{t-1} + e_t, \qquad |\phi|<1, \qquad e_t \overset{\text{iid}}{\sim} (0,\sigma^2).
```

For the stationary process,

```math
\mathbb{E}[x_t]=\frac{c}{1-\phi}, \qquad \mathrm{SD}(x_t)=\frac{\sigma}{\sqrt{1-\phi^2}}, \qquad \rho_k=\phi^k .
```

A shock decays geometrically and never fully ends. With $|\phi|\ge1$ the process isn't stationary; $\phi=1$ gives the **random walk** $x_t=x_{t-1}+e_t$.

### AR(p)

```math
x_t = c + \phi_1x_{t-1} + \cdots + \phi_px_{t-p} + e_t .
```

Two tanks in series give an AR(2).

### One-step versus multi-step forecasts

One-step prediction always uses the latest *observed* value. A 20-step forecast must feed its own predictions back in, so errors compound and the forecast drifts toward the process mean. A model can look excellent one step ahead and poor 20 steps ahead.

## 6. Moving-average models

### MA(1)

With the sign convention used here,

```math
x_t = \mu + e_t - \theta e_{t-1}, \qquad |\theta|<1,
```

```math
\mathbb{E}[x_t]=\mu, \qquad \mathrm{Var}(x_t)=\sigma^2(1+\theta^2), \qquad \rho_1=\frac{-\theta}{1+\theta^2}, \qquad \rho_k=0\;(k>1).
```

Adjacent values share the shock $e_{t-1}$; values two steps apart share nothing.

### MA(q)

```math
x_t = \mu + e_t - \theta_1e_{t-1} - \cdots - \theta_qe_{t-q}, \qquad \rho_k=0 \text{ for } k>q .
```

A shock affects exactly $q$ future values, then it's gone.

![AR and MA memory](../assets/clean_diagrams/ar_ma_memory.png)

## 7. AR versus MA

| | AR($p$) | MA($q$) |
|---|---|---|
| Built from | past observations | current and past shocks |
| A shock's effect | decays forever, recursively | lasts exactly $q$ steps |
| ACF | decays gradually | cuts off after lag $q$ |
| PACF | cuts off after lag $p$ | decays gradually |

> [!WARNING]
> **MA model ≠ moving-average filter**
>
> An MA *model* combines random shocks. A moving-average *filter* (Chapter 3) averages observed values. Same name, different objects.

## 8. Differencing and integrated models

A local-level model has a slowly wandering mean:

```math
x_t=\mu_t+e_t, \qquad \mu_t=\mu_{t-1}+\delta_t .
```

If $\delta_t$ is tiny, the series is white noise around a slow drift; if $e_t$ is tiny, it's nearly a random walk. It can be rewritten as

```math
x_t = x_{t-1} + \epsilon_t - \theta\epsilon_{t-1}, \qquad \Delta x_t = x_t-x_{t-1} = \epsilon_t - \theta\epsilon_{t-1},
```

so the **first difference is MA(1)**. This is the IMA(1,1) model: one difference, one MA term. The exact derivation is in [Chapter 12](12_handwritten_appendix.md).

Differencing also removes polynomial trends. One difference turns a degree-$d$ polynomial trend into a degree-$(d-1)$ one, so a linear trend needs one difference and a quadratic trend two. Don't over-difference: each difference adds noise and can introduce artificial negative autocorrelation.

## 9. ARIMA and SARIMAX

**ARIMA($p,d,q$):** AR order $p$, $d$ differences, MA order $q$.

**SARIMAX** adds seasonal AR/differencing/MA terms and e**X**ogenous regressors. With the backshift operator $Bx_t=x_{t-1}$, monthly seasonality uses $B^{12}x_t=x_{t-12}$, seasonal differencing is $(1-B^{12})x_t$, and regular plus seasonal differencing is $(1-B)(1-B^{12})x_t$. The Mauna Loa CO₂ series needs both: it has a long-term rise and an annual cycle.

## 10. ACF, PACF and model order

- **ACF:** correlation at each lag.
- **PACF:** correlation between $x_t$ and $x_{t-k}$ after removing the effect of the lags in between.

Rules of thumb: an AR($p$) has a PACF that cuts off after $p$; an MA($q$) has an ACF that cuts off after $q$; a mixed ARMA tails off in both. These are starting points, not rules. Compare candidates with AIC/BIC (Chapter 3) and residual checks.

## 11. Residual checks: the real test

With residuals $r_t=x_t-\hat x_t$, ask four separate questions:

| Check | Question |
|---|---|
| residual plot | any remaining trend, level shift, or changing spread? |
| residual mean | is the model biased high or low? |
| residual ACF (and Ljung–Box test) | any remaining autocorrelation? |
| histogram / Q–Q plot | does the distribution match the assumed one? |

They aren't interchangeable. A clean ACF can hide a model that consistently under-predicts.

**Worked refinement, on a trending series:**

1. **ARIMA(1,0,0):** misses the trend and under-predicts.
2. **ARIMA(1,1,0):** differencing fixes the level, but lag-1 residual correlation remains.
3. **ARIMA(1,1,1):** add an MA term for the remaining lag-1 structure.

![ARIMA refinement loop](../assets/clean_diagrams/arima_refinement_loop.png)

```python
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.stats.diagnostic import acorr_ljungbox

fit = ARIMA(y_train, order=(1, 1, 1)).fit()
print(fit.aic, fit.bic)
print(acorr_ljungbox(fit.resid, lags=[10]))   # large p-value = no leftover autocorrelation
```

## 12. What to remember

| Model | Signature |
|---|---|
| White noise | no autocorrelation |
| AR($p$) | past values; ACF decays, PACF cuts at $p$ |
| MA($q$) | past shocks; ACF cuts at $q$ |
| IMA | difference once, then MA |
| ARIMA | AR + differencing + MA |
| SARIMAX | + seasonal terms + exogenous inputs |

```text
plot → remove trend/season if needed → ACF/PACF → fit a small model
     → check forecasts and residuals → refine only if structure remains
```

## 13. Questions and answers

<details><summary>How do I choose the differencing order without over-differencing?</summary>

Difference until a unit-root test (ADF, or KPSS with the opposite null) says stationary, usually $d\le2$. A lag-1 ACF of the differenced series near $-0.5$, or a rising residual variance, signals over-differencing.
</details>

<details><summary>How far ahead can an AR model forecast usefully?</summary>

The forecast reverts to the mean at rate $\phi^h$, and the prediction variance grows toward $\mathrm{Var}(x_t)$. Once $\phi^h$ is negligible (e.g. $h\approx3/(1-\phi)$ for $\phi$ near 1), the forecast is just the mean.
</details>

<details><summary>How do I find the seasonal period when the application doesn't tell me?</summary>

Look for peaks in the ACF at lag $s$, $2s$, …, or in the periodogram (Chapter 4).
</details>

<details><summary>Several models leave white-noise residuals. Which one do I pick?</summary>

Compare AIC/BIC for fit versus complexity, then out-of-sample forecast error on a time-ordered holdout. Prefer the simplest model that's about as good.
</details>

---

[← Previous: Temporal Data](01_temporal_data_and_learning.md) · [Course map](../course_map.md) · [Next: Filters and Decomposition →](03_filters_smoothing_and_decomposition.md)
