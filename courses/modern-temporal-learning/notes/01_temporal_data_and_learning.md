---
course: "ASU CSE 598 Modern Temporal Learning"
chapter: 1
title: "Temporal Data and Temporal Learning"
source_pages: "2-4"
status: "strong-draft"
release: "v0.2.1"
math_style: "github-native"
---

# Temporal Data and Temporal Learning

## 1. Chapter overview

This opening section defines temporal data, distinguishes time series from
event sequences, introduces the main temporal-learning tasks, and surveys
several ways to represent a temporal sequence before applying a learning
algorithm.

The course begins with an important structural difference between ordinary
tabular machine learning and temporal learning:

> **Handwritten annotation:** Rows or instances are usually treated as
> independent in ordinary machine learning, but observations within a time
> series are related.

This dependence is the reason that temporal order, lag, interval, and
sampling structure matter.

**Sources:** CSE598MTL.pdf, pp. 2-4

---

## 2. Learning objectives

After this chapter, the reader should be able to:

1. distinguish time series, event sequences, streaming data, and
   spatiotemporal data;
2. explain why temporal ordering can be total or partial;
3. describe a time series as observed values of a stochastic process;
4. distinguish outliers, change points, and online process changes;
5. compare instance-based and feature-based learning from time series;
6. describe symbolic, piecewise, feature, and landmark representations.

**Sources:** CSE598MTL.pdf, pp. 2-4

---

## 3. Temporal data

### 3.1 General definition

Temporal data are sequences whose elements have an ordering in time.
The primary values may be numerical, categorical, univariate,
multivariate, or composite.

Examples listed in the course include:

- sensor measurements;
- stock prices;
- EEG signals;
- purchases and credit-card transactions;
- medical records;
- message traces and web logs;
- traffic and gesture-recognition data.

**Source:** CSE598MTL.pdf, p. 2

### 3.2 Total and partial ordering

The course emphasizes **total ordering**, rather than only partial
ordering, as a common property of temporal elements.

A total order can be represented as:

```text
a < b < c < d
```

Every pair of elements is comparable.

A partial order may specify only some relations:

```text
a < b       c < d
```

The relation between the two pairs may be unspecified.

> **Handwritten annotation:** “Total ordering vs. partial ordering.”

### Clarification

Time-series observations normally follow a single timeline and therefore
have a total order. Some event systems instead contain only precedence
constraints, so not every event pair has to be comparable.

**Source:** CSE598MTL.pdf, p. 2

---

## 4. Time series and stochastic processes

### 4.1 Time series

The slide defines a time series as a sequence of real values observed at
successive, equally spaced points in time:

```math
x = (x_1, x_2, \ldots, x_n)
```

### 4.2 Stochastic process

A real stochastic process $X$ is a family of real random variables:

```math
X = (x_t : t \in T)
```

where $T$ is the index set.

- If $T \subseteq \mathbb{Z}$, the process is discrete.
- If $T$ is an interval of $\mathbb{R}$, the process is continuous.
- A time series is the set of observed values of a stochastic process.

> **Handwritten annotation:** A stochastic process is a series of random
> variables generated over time.

### Clarification

The stochastic process is the underlying probabilistic mechanism.
The time series is the finite realization that is actually observed.

**Source:** CSE598MTL.pdf, p. 2

---

## 5. Temporal data language

The meaning of a temporal measurement depends on how time is represented.

### 5.1 Time point and interval

A record may refer to:

- a specific time point, such as `08:00:00, January 1, 2022`;
- a calendar interval, such as `January 1, 2022`.

### 5.2 Granularity

Granularity, or resolution, describes the temporal precision:

- milliseconds;
- seconds;
- minutes;
- hours;
- days;
- years.

### 5.3 Absolute and relative anchors

An observation can be anchored to:

- an absolute date and time;
- a time relative to an event, such as five days after hospital discharge;
- a relative calendar unit, such as a month, day, or year.

### 5.4 Qualitative temporal relations

The course page also lists interval relations such as:

- before;
- overlaps;
- during;
- meets;
- starts;
- finishes;
- equals;
- after;
- contains;
- started by;
- finished by.

These relations describe how two intervals are positioned relative to
one another.

**Source:** CSE598MTL.pdf, p. 2

---

## 6. Forms of temporal data

### 6.1 Time series

Time series are usually regularly sampled numerical sequences, although
the underlying information can be univariate, multivariate, or composite.

### 6.2 Event sequences

Event sequences are often irregularly spaced. Examples include:

- transaction data;
- credit-card purchases;
- health appointments;
- web-log pages.

### 6.3 Streaming data

Streaming data are not stored as a complete fixed dataset before
processing. Observations arrive continuously.

### 6.4 Spatiotemporal data and video

These data combine time with location or image structure. Examples on the
page include traffic and gesture recognition.

**Source:** CSE598MTL.pdf, p. 2

---

## 7. Temporal-learning tasks

### 7.1 Clustering

Clustering groups temporal objects with similar behavior.

Course examples include:

- patients with similar health conditions, activity, or medication doses;
- web sessions with similar page views;
- weeks or locations with similar product demand;
- tools with similar performance or maintenance conditions;
- sensors with similar measurements.

### 7.2 Classification

Classification assigns outcomes or labels to temporal sequences.

### 7.3 Anomaly detection

The course separates several forms of unusual behavior.

#### Outliers

An outlier is one or a few unusual observations.

> **Handwritten example:** A short batch of about 20 minutes may be treated
> as an outlying segment.

#### Change points

A change point indicates that the stochastic process generating the data
has changed. It is detected retrospectively after the observations are
available.

> **Handwritten annotation:** A change point is more than only a few points
> being far away from the rest of the data; it represents a distributional
> change.

#### Process control

Process control seeks to detect a change as soon as possible in real time.

Examples listed include:

- heart arrhythmia;
- tool failure;
- maintenance problems;
- unusual demand;
- supply shortage.

### Clarification

A useful distinction in the notes is:

```text
Outlier:
    isolated or short unusual observation

Change point:
    persistent change in the generating process,
    typically detected retrospectively

Process control:
    online detection of process change
```

**Source:** CSE598MTL.pdf, p. 2

### 7.4 Prediction

Prediction includes forecasting future values.

The course distinguishes:

- short-range forecasts for inventory, scheduling, staffing, or available beds;
- long-range forecasts for capacity planning, real estate, material sources,
  or hospital capacity;
- a forecast horizon, which specifies how far into the future predictions extend.

### 7.5 Sequential pattern mining

Sequential pattern mining looks for recurring ordered behavior.

Course examples include:

- peanut-butter purchases followed by jelly purchases;
- drugs followed by side effects;
- cancer-treatment regimens;
- website page-view order;
- intrusion-detection operations.

**Source:** CSE598MTL.pdf, p. 2

---

## 8. Learning from time series

### 8.1 Ordinary machine-learning assumption

The page asks what machine learning usually assumes about instances of
data. The handwritten response is that rows or instances are generally
treated as independent.

For temporal data, adjacent and lagged values are usually related.
A model must therefore preserve or explicitly represent this dependence.

**Source:** CSE598MTL.pdf, p. 3

### 8.2 Instance-based learning

Instance-based methods use the original sequence, or a direct comparison
between original sequences, as the model input.

The course lists several similarity measures:

- Euclidean distance;
- absolute difference;
- maximum difference;
- longest common subsequence (LCSS).

For equal-length time series $x$ and $y$, Euclidean distance is:

```math
d(x,y) =
\sqrt{\sum_{t=1}^{T}(x_t-y_t)^2}
```

The slide associates this type of distance with methods such as K-nearest
neighbors.

### Clarification

A direct distance requires the sequences to be comparable. Different
lengths, shifts, local timing differences, and irregular sampling may
require a more specialized measure.

**Source:** CSE598MTL.pdf, p. 3

### 8.3 Feature-based learning

Feature-based methods transform a sequence into a smaller feature vector.

The course lists examples such as:

- mean;
- standard deviation;
- slope;
- global features;
- features computed over several segments;
- edit distance;
- differences after threshold-based tokenization;
- dynamic time warping;
- learned representations.

A generic representation is:

```text
Original sequence
    -> temporal segmentation or transformation
    -> feature vector
    -> clustering or classification model
```

The page raises an open design question:

> Should segments be mutually exclusive?

This is preserved as a course-design question rather than resolved here.

**Source:** CSE598MTL.pdf, p. 3

---

## 9. Symbolic representations

A numerical series can be converted into a sequence of symbols by applying
thresholds or discretization rules.

Example symbolic sequence shown on the page:

```text
B A B B B A A B B A C A A C
```

A second symbolic sequence can be compared using edit distance.

The slide uses:

- match cost $= 0$;
- mismatch cost $= 1$.

### Clarification

A symbolic representation reduces numerical detail but allows sequence
comparison through methods originally designed for strings or event
sequences.

**Source:** CSE598MTL.pdf, p. 4

---

## 10. Piecewise approximation

### 10.1 Piecewise aggregate approximation

Piecewise aggregate approximation (PAA) divides a sequence into windows
and replaces each window with its average.

For a segment $S_j$, a conceptual form is:

```math
\bar{x}_j =
\frac{1}{|S_j|}
\sum_{t \in S_j} x_t
```

The course example averages groups of three values.

```text
Original points:
x1 x2 x3 | x4 x5 x6 | x7 x8 x9 | ...

PAA values:
mean(1:3) | mean(4:6) | mean(7:9) | ...
```

### 10.2 Shape-description language

The page also shows a qualitative representation that labels local
movement with terms such as:

```text
up -> stable -> down -> up
```

### Clarification

PAA preserves coarse level information. Shape labels preserve qualitative
direction. Both reduce the original sequence to a smaller representation.

**Source:** CSE598MTL.pdf, p. 4

---

## 11. Custom features

The course proposes representing each sequence with engineered features.

One illustrated example is a scatter plot in which each time series is
summarized by:

- the widths of its peaks;
- the heights of its peaks.

```text
Time series
    -> detect peaks
    -> measure peak width and height
    -> represent each series as points in feature space
```

The page asks whether the measurements should summarize the entire
sequence or retain several feature values. This remains an open
representation-design choice.

**Source:** CSE598MTL.pdf, p. 4

---

## 12. Landmarks and segmentation

### 12.1 Landmark features

The slide asks whether features should be extracted from intervals between
landmarks.

Examples include:

- electrocardiograms, with beats and multiple channels;
- electroencephalograms, with multiple channels;
- signals containing artifacts and noise.

### 12.2 Fixed or adaptive segments

Segments may have:

- fixed length;
- adaptive length.

Within a segment, the signal can be approximated as:

- piecewise constant;
- piecewise linear;
- a spline function.

### 12.3 Landmark-analysis definition on the page

The pasted definition states:

> Landmark analysis designates a time point during follow-up, called the
> landmark time, and analyzes only subjects who have survived until that
> time.

### Clarification

That definition comes from survival-analysis usage. In the surrounding
course slide, “landmarks” are also discussed more generally as meaningful
temporal locations used to define intervals and extract features.
The two uses should not be silently treated as identical.

**Source:** CSE598MTL.pdf, p. 4

---

## 13. Representation choices

The first three academic pages introduce a spectrum of representations:

| Representation | Preserves | Loses or simplifies |
|---|---|---|
| Raw sequence | Original values and order | May be high-dimensional and difficult to align |
| Pairwise distance | Direct sequence similarity | Depends strongly on the selected distance |
| Global features | Compact summary | Local order and events may disappear |
| Segment features | Some local structure | Depends on segmentation |
| Symbolic sequence | Order of coarse states | Numerical detail |
| PAA | Coarse levels over windows | Fine-scale fluctuations |
| Landmark features | Event-centered structure | Requires meaningful landmarks |
| Learned representation | Task-adapted information | Interpretation may be more difficult |

**Sources:** CSE598MTL.pdf, pp. 3-4

---

## 14. Common confusions

### Outlier versus change point

An outlier is not automatically a change point. A change point represents
a change in the process or distribution, not merely one extreme value.

### Time series versus stochastic process

A stochastic process is the probabilistic family of random variables.
A time series is an observed realization.

### Feature representation versus temporal model

Feature extraction may remove temporal structure before learning.
A temporal model instead attempts to learn directly from the ordered
sequence.

### Landmark terminology

The page combines a survival-analysis definition with a broader
signal-segmentation discussion. Their relationship requires care.

**Sources:** CSE598MTL.pdf, pp. 2-4

---

## 15. Questions preserved for later discussion

1. When should a raw sequence be used instead of engineered features?
2. Should temporal segments be mutually exclusive or overlapping?
3. When does a symbolic representation discard too much numerical detail?
4. How should landmarks be selected when no natural event boundary exists?
5. How can we distinguish a short anomalous interval from a persistent
   distributional change?
6. Which distances remain meaningful when sequences have unequal length
   or shifted timing?

These questions are motivated by the slides and annotations but are not
fully answered on pages 2-4.

---

## 16. Source map

| PDF page | Material reconstructed |
|---:|---|
| 2 | Temporal data, stochastic processes, temporal language, data types, learning tasks |
| 3 | Independence assumption, instance-based and feature-based learning |
| 4 | Symbolic representation, PAA, custom features, landmarks and segmentation |

## Review status

- Printed slide content: `[VERIFIED]`
- Major handwritten annotations: `[VERIFIED]`
- Small handwritten examples and abbreviated marks: `[INTERPRETED]`
- Landmark terminology relationship: `[NEEDS REVIEW]`
