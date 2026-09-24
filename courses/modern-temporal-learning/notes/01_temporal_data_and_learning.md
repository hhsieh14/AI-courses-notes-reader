# 1. Temporal Data and Temporal Learning

Most of classical machine learning treats rows as independent draws. Time-ordered data breaks that assumption: today's value depends on yesterday's. This chapter defines the kinds of temporal data, the learning tasks we pose on them, and the main ways to represent a sequence before a model sees it. Representation is a theme that returns throughout the course.

> [!TIP]
> **The one idea to take from this chapter**
>
> In ordinary ML, rows are treated as independent. In a time series, observations are related, so order, lag, spacing and sampling rate all carry information. Shuffle a time series and you destroy the thing you're trying to model.

## 1. Temporal data

Temporal data are elements ordered in time. Their values can be numeric or categorical, univariate or multivariate. Examples: sensor streams, stock prices, EEG, purchases and card transactions, medical records, web logs, traffic and gestures.

### Total versus partial order

A time series has a **total order**: any two observations can be compared, $a<b<c<d$. Some event data have only a **partial order**, with precedence constraints like $a<b$ and $c<d$ but no relation between the pairs (think of steps in parallel workflows). Most methods in this course assume a total order.

## 2. Time series and stochastic processes

A **time series** is a sequence of real values observed at successive, usually equally spaced times:

```math
x = (x_1, x_2, \ldots, x_n).
```

A **stochastic process** is a family of random variables indexed by time, $X=(x_t:t\in T)$: discrete if $T\subseteq\mathbb Z$, continuous if $T$ is an interval of $\mathbb R$. The time series is one observed **realization** of the process. The process is the probabilistic mechanism, and the series is the single sample path we actually see. That is why stationarity (Chapter 2) matters so much: it is what lets one path teach us about the mechanism.

## 3. Describing time

- **Point versus interval:** `08:00:00, 1 Jan 2022` versus `1 Jan 2022`.
- **Granularity:** milliseconds up to years. Mixing granularities is a common source of bugs.
- **Absolute versus relative anchors:** a calendar timestamp versus "5 days after discharge".
- **Interval relations** (Allen's interval algebra): before, meets, overlaps, starts, during, finishes, equals, and their inverses. These matter for event data such as "did medication A overlap with symptom B?"

## 4. Forms of temporal data

| Form | Typical sampling | Examples |
|---|---|---|
| Time series | regular | sensors, prices, vitals |
| Event sequence | irregular | transactions, appointments, clicks |
| Stream | unbounded, arrives continuously | logs, IoT feeds |
| Spatiotemporal / video | time × space or pixels | traffic, gestures |

## 5. Learning tasks

- **Clustering:** group series with similar behavior (patients, sessions, weeks of demand, machines, sensors).
- **Classification:** assign a label to a whole sequence (arrhythmia or not).
- **Anomaly detection**, in three flavors that are easy to confuse:

| | What it is | When it's detected |
|---|---|---|
| **Outlier** | one or a few unusual observations (e.g. a 20-minute glitch) | anytime |
| **Change point** | the generating process itself changes (a new mean, variance or dynamics) | retrospectively, after the data are in |
| **Process control** | detect a change as soon as it happens | online, in real time (heart arrhythmia, tool failure, demand spikes) |

- **Forecasting:** predict future values over a **horizon**. Short-range forecasts drive inventory, staffing and bed counts; long-range forecasts drive capacity planning.
- **Sequential pattern mining:** frequent ordered patterns, such as peanut butter → jelly, drug → side effect, page-view paths, intrusion steps.

## 6. Learning from time series

### Instance-based

Compare sequences directly and feed the similarities to a method like KNN. Distances include Euclidean,

```math
d(x,y)=\sqrt{\sum_{t=1}^{T}(x_t-y_t)^2},
```

absolute and maximum difference, and the **longest common subsequence** (LCSS). Euclidean distance needs equal lengths and aligned timing. A one-step shift can make two identical shapes look far apart. **Dynamic time warping** (DTW) fixes this by finding the best monotone alignment between the two series before summing differences, at $O(T^2)$ cost.

### Feature-based

Summarize each sequence as a feature vector, then use any standard model:

```text
sequence → segment / transform → feature vector → clustering or classification
```

Features range from global statistics (mean, standard deviation, slope) to per-segment statistics, distances to reference patterns, and learned representations (Chapter 10).

Should segments be mutually exclusive? Not necessarily. Non-overlapping windows give independent-looking features and fewer rows. Overlapping (sliding) windows give more training rows and smoother coverage, but adjacent rows become correlated, so validation must use time-blocked splits (Chapter 8).

## 7. Symbolic representations

Discretize values into symbols with thresholds and compare strings:

```text
B A B B B A A B B A C A A C
```

Edit distance (match cost 0, mismatch 1) then measures dissimilarity. SAX (symbolic aggregate approximation) is the standard version: PAA first (below), then map levels to symbols using Gaussian quantiles. Symbols throw away numeric detail but make string tools such as edit distance, suffix trees and pattern mining available.

## 8. Piecewise approximation

**Piecewise aggregate approximation** (PAA) replaces each window by its mean:

```math
\bar{x}_j=\frac{1}{|S_j|}\sum_{t\in S_j}x_t .
```

```text
x1 x2 x3 | x4 x5 x6 | x7 x8 x9 | ...
mean     | mean     | mean     | ...
```

A **shape language** instead labels local movement, e.g. `up → stable → down → up`. PAA keeps coarse level; shape labels keep direction.

## 9. Custom features

Domain-specific features often beat generic ones. For a peaky signal, represent each series by the heights and widths of its peaks, and plot series as points in that 2-D feature space. You can summarize a whole series (mean peak height) or keep several values (the three largest peaks). The first is compact; the second preserves more structure.

## 10. Landmarks and segmentation

**Landmarks** are meaningful time points used to anchor features: R-peaks in an ECG, stimulus onsets in EEG. Between landmarks, a segment can be approximated as constant, linear or a spline, with fixed or adaptive length. Artifacts and noise complicate landmark detection.

(In survival analysis, "landmark analysis" means something different: pick a landmark time and analyze only subjects still at risk then. Same word, different idea.)

## 11. Choosing a representation

| Representation | Keeps | Loses |
|---|---|---|
| Raw sequence | everything | may be long, hard to align |
| Pairwise distance | similarity | depends entirely on the distance chosen |
| Global features | compact summary | local order and events |
| Segment features | some local structure | depends on segmentation |
| Symbolic | order of coarse states | numeric detail |
| PAA | coarse level | fine fluctuations |
| Landmark features | event-centered structure | needs meaningful landmarks |
| Learned representation | task-relevant information | interpretability |

## 12. Common confusions

- **Outlier ≠ change point.** One extreme value isn't a change in the process.
- **Process ≠ series.** The process is the random mechanism; the series is one realization.
- **Feature extraction vs temporal model.** Hand-made features can discard order before learning; sequence models (Chapters 8–11) learn from the ordered data directly.

## 13. Questions and answers

<details><summary>When should I use the raw sequence instead of engineered features?</summary>

When there's enough data for a sequence model to learn its own features and the relevant patterns are hard to specify by hand. With little data or strong domain knowledge, engineered features usually win.
</details>

<details><summary>How do I tell a short anomaly from a persistent change?</summary>

Look at what happens afterward. An outlier returns to the old distribution; after a change point the new statistics persist. Change-point methods (CUSUM, likelihood-ratio tests, Bayesian online change-point detection) formalize "persists".
</details>

<details><summary>Which distances work for sequences of unequal length or shifted timing?</summary>

DTW and LCSS (they align before comparing), or distances between fixed-length representations (features, PAA/SAX, embeddings).
</details>

<details><summary>How do I pick landmarks when there's no natural event?</summary>

Use data-driven change points or peaks, fixed windows, or skip landmarks and use a representation that is shift-tolerant (wavelets, convolutional features).
</details>

---

[Course map](../course_map.md) · [Next: Stationarity and Classical Models →](02_classical_time_series_models.md)
