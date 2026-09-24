# 6. Classification Evaluation

A classifier is only as good as the number we use to judge it. This chapter starts with why accuracy misleads on imbalanced data, builds the confusion matrix and the metrics derived from it, and then evaluates a score-based classifier across all thresholds with ROC and precision–recall curves.

## 6.1 Why accuracy alone can mislead

Suppose 1% of patients have leukemia. A "classifier" that always answers **healthy** is right 99% of the time:

$$ \mathrm{Accuracy}=0.99, $$

and it misses every patient who is sick. One aggregate percentage hides *which* mistakes are being made.

## 6.2 The confusion matrix

|  | Predicted positive | Predicted negative |
|---|---:|---:|
| **Actually positive** | True positive ($TP$) | False negative ($FN$) |
| **Actually negative** | False positive ($FP$) | True negative ($TN$) |

In hypothesis-testing language a false positive is a **Type I error** and a false negative a **Type II error**.

![Confusion matrix and ROC curve](assets/diagrams/06_classification_evaluation.svg)

## 6.3 Accuracy

$$ \mathrm{Accuracy}=\frac{TP+TN}{TP+FN+FP+TN}. $$

It is fine when classes are balanced and both errors cost about the same, and misleading otherwise.

## 6.4 Recall (sensitivity)

$$ \mathrm{Recall}=\mathrm{Sensitivity}=\frac{TP}{TP+FN} \quad\text{(denominator: all actual positives).} $$

*Of everything that was positive, how much did we catch?* In medicine, a highly sensitive test is good for **ruling out** disease: it rarely misses a sick patient, so a negative result is reassuring.

## 6.5 Specificity

$$ \mathrm{Specificity}=\frac{TN}{TN+FP} \quad\text{(denominator: all actual negatives).} $$

*Of everything that was negative, how much did we correctly clear?* A highly specific test is good for **ruling in**: it rarely flags a healthy patient, so a positive result is convincing.

## 6.6 Precision

$$ \mathrm{Precision}=\frac{TP}{TP+FP} \quad\text{(denominator: all predicted positives).} $$

*When the model says positive, how often is it right?* Recall and precision share the numerator and differ in the denominator: actual positives versus predicted positives.

## 6.7 Two worked examples

**Dog detector.** 12 dog and 10 cat images. The program flags 8 images as dogs, of which 5 are dogs and 3 are cats:

$$ \mathrm{Precision}=\frac{5}{8}=0.625, \qquad \mathrm{Recall}=\frac{5}{12}\approx0.417 . $$

**Search engine.** It returns 30 pages, 20 of them relevant, and misses 40 other relevant pages. So $TP=20$, $FP=10$, $FN=40$:

$$ \mathrm{Precision}=\frac{20}{30}\approx0.67, \qquad \mathrm{Recall}=\frac{20}{60}\approx0.33 . $$

The results it shows are mostly good, but it finds only a third of what exists.

## 6.8 The F1 score

$$ F_1=2\,\frac{\mathrm{Precision}\cdot\mathrm{Recall}}{\mathrm{Precision}+\mathrm{Recall}}. $$

The harmonic mean is high only if **both** components are high. Precision 1.0 with recall 0.01 gives $F_1\approx0.02$, whereas the arithmetic mean would say 0.5. If recall matters $\beta$ times as much as precision, use $F_\beta=(1+\beta^2)\frac{PR}{\beta^2P+R}$. Note that $F_1$ ignores $TN$ entirely.

## 6.9 Always name the positive class

Every metric above depends on which class is called "positive". In the leukemia example, making *leukemia* the positive class means recall directly measures how many sick patients we detect, which is exactly what the always-healthy classifier fails at (recall = 0).

## 6.10 From scores to decisions

A probabilistic classifier outputs $h_\theta(x)=p(y=1\mid x)$, and a threshold $\tau$ turns it into a label: $\hat y=1$ if $h_\theta(x)\ge\tau$. Moving $\tau$ changes the whole confusion matrix. Lowering it catches more positives (higher recall) at the cost of more false alarms. To evaluate the *model* rather than one threshold, sweep $\tau$.

## 6.11 The ROC curve

The receiver operating characteristic curve plots, for every threshold,

$$ \mathrm{TPR}=\frac{TP}{TP+FN} \;\;(\text{vertical}) \qquad\text{against}\qquad \mathrm{FPR}=\frac{FP}{FP+TN}=1-\mathrm{Specificity}\;\;(\text{horizontal}). $$

Each threshold gives one $(\mathrm{FPR},\mathrm{TPR})$ point; sweeping the threshold traces the curve from $(0,0)$ (predict nothing positive) to $(1,1)$ (predict everything positive).

## 6.12 Reading an ROC plot

- The top-left corner $(0,1)$ is a perfect classifier.
- The diagonal is random guessing.
- Above the diagonal is better than random; below it is worse (and flipping the predictions would put it above).

## 6.13 AUC

The area under the ROC curve summarizes it in one number: 1.0 perfect, 0.5 random. Curves with AUC 0.9, 0.75 and 0.5 sit progressively closer to the diagonal.

> [!NOTE]
> **Beyond the lecture: what AUC actually measures**
>
> AUC equals the probability that a randomly chosen positive gets a higher score than a randomly chosen negative:
>
> $$ \mathrm{AUC}=P\left(s(x^+)>s(x^-)\right). $$
>
> So AUC measures **ranking quality** only. It doesn't change if you apply any monotone transform to the scores, which means it says nothing about whether the probabilities are *calibrated*. For calibration, use log loss or a reliability diagram.

## 6.14 Precision–recall curves for rare positives

> [!NOTE]
> **Beyond the lecture: why PR-AUC for imbalanced problems**
>
> With 1 positive per 1,000 negatives, an FPR of 1% sounds small, but it means 10 false alarms for every positive. ROC hides this because FPR divides by the huge number of negatives. A **precision–recall curve** plots precision against recall across thresholds, and its area (average precision) reacts strongly to false positives when positives are rare. A random classifier's PR-AUC equals the positive rate (0.001 here), not 0.5, so the baseline is honest.
>
> This is why my [harmful-content](https://github.com/hhsieh14/ml-system-design-learning-notes/blob/main/chapters/04_harmful_content_detection/chapter_04_harmful_content_detection.md) and [ad-click](https://github.com/hhsieh14/ml-system-design-learning-notes/blob/main/chapters/02_ad_click_prediction/chapter_02_ad_click_prediction.md) design notes use PR-AUC as the primary offline metric.

## 6.15 Which metric answers which question?

| Metric | Denominator | Question it answers |
|---|---|---|
| Accuracy | all examples | how often are we right overall? |
| Recall / sensitivity / TPR | actual positives | how many positives did we find? |
| Specificity | actual negatives | how many negatives did we clear? |
| FPR | actual negatives | how often do negatives raise a false alarm? |
| Precision | predicted positives | how trustworthy is a positive prediction? |
| $F_1$ | precision and recall | are both reasonably high? |
| ROC-AUC | all thresholds | how well do scores rank positives above negatives? |
| PR-AUC | all thresholds | how well do we find rare positives without false alarms? |

Pick the metric from the cost of each error. In screening, missing a patient (FN) is the expensive error, so optimize recall at an acceptable precision. For a spam filter, losing a real email (FP) is the expensive one, so optimize precision.

## 6.16 Common mistakes

1. **Reporting accuracy alone on imbalanced data.**
2. **Swapping the actual and predicted axes** of the confusion matrix.
3. **Mixing up FP and FN**, or **precision and recall**.
4. **Not stating the positive class.**
5. **Treating one threshold as the whole ROC curve.**
6. **Putting specificity instead of $1-$specificity on the ROC x-axis.**
7. **Reading AUC as accuracy or as calibration.** It is a ranking measure.
8. **Using ROC-AUC alone when positives are very rare.** Add PR-AUC.

## 6.17 Summary

- The confusion matrix separates the two kinds of correct and incorrect predictions.
- Recall and specificity divide by the true classes; precision divides by the predicted positives.
- $F_1$ is the harmonic mean of precision and recall.
- ROC plots TPR against FPR across thresholds; AUC is the probability of ranking a random positive above a random negative.
- For rare positives, precision–recall curves are more informative.

## 6.18 Self-check

1. Why is 99% accuracy useless in the leukemia example?
2. In the dog example, why is precision 5/8 but recall 5/12?
3. Why does high sensitivity help rule a disease *out*?
4. Why is $F_1$ low when either precision or recall is low?
5. What does AUC = 0.8 mean in words?
6. With 0.1% positives, what PR-AUC does a random classifier get?

<details>
<summary>Answers</summary>

1. Its recall on the class we care about is 0.
2. Precision divides by the 8 predicted dogs; recall divides by the 12 actual dogs.
3. A sensitive test rarely produces false negatives, so a negative result makes the disease unlikely.
4. The harmonic mean is dominated by the smaller value.
5. A random positive outscores a random negative 80% of the time.
6. About 0.001, the positive rate.

</details>
