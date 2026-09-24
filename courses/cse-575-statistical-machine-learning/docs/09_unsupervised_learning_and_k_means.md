# 9. Unsupervised Learning and K-Means

Without labels there's no "right answer" to fit. The goal becomes finding structure already in the inputs. This chapter separates the two main unsupervised tasks and develops K-means in detail: the algorithm, the objective it minimizes, why initialization matters, K-means++, and how to choose $K$.

## 9.1 What changes without labels

The dataset is just inputs,

$$ \mathcal{D}=\left\lbrace x^{(i)}\right\rbrace_{i=1}^{n}, $$

and the workflow is: fit a model to unlabeled data, extract a structure (groups, a low-dimensional space), then map new points into that structure.

![K-means workflow](assets/diagrams/09_kmeans_workflow.png)

*The assign/update loop, inertia, sensitivity to initialization, K-means++ seeding and the elbow heuristic.*

## 9.2 Two kinds of unsupervised learning

| Task | Goal | Example |
|---|---|---|
| **Clustering** | find groups of similar points | group news articles by (unknown) topic |
| **Dimensionality reduction** | find a simpler representation | compress high-resolution images (Chapter 11) |

The number of clusters is a modeling choice, not something the data hands you. The same web-app users described by age could reasonably be split into two groups for one purpose and five for another.

## 9.3 Notation

Choose $K$. K-means keeps an assignment $c^{(i)}\in\lbrace1,\ldots,K\rbrace$ for every point and a centroid $\mu_j\in\mathbb{R}^d$ for every cluster.

## 9.4 The algorithm

1. **Initialize** $K$ centroids $\mu_1,\ldots,\mu_K$ (randomly, or with K-means++ below).
2. **Assign** each point to its nearest centroid:
   $$ c^{(i)}\leftarrow\underset{j}{\mathrm{arg\,min}}\left\lVert x^{(i)}-\mu_j\right\rVert^2 . $$
3. **Update** each centroid to the mean of its points:
   $$ \mu_j\leftarrow\frac{\sum_{i}\mathbb{1}[c^{(i)}=j]\,x^{(i)}}{\sum_{i}\mathbb{1}[c^{(i)}=j]} . $$
4. **Repeat** steps 2–3 until no assignment changes.

(If a cluster ends up empty, re-seed its centroid, for example at the point farthest from its current centroid.)

## 9.5 The objective: inertia

K-means minimizes the within-cluster sum of squared distances, also called **inertia** or distortion:

$$ J(c,\mu)=\sum_{i=1}^{n}\left\lVert x^{(i)}-\mu_{c^{(i)}}\right\rVert^2 . $$

The two steps are **coordinate descent** on $J$:

- with centroids fixed, assigning each point to its nearest centroid minimizes $J$ over $c$;
- with assignments fixed, the mean minimizes the sum of squared distances within each cluster, so the update minimizes $J$ over $\mu$.

Neither step can increase $J$, and there are finitely many assignments, so the loop always terminates. But it terminates at a **local** minimum. Finding the global one is NP-hard in general.

## 9.6 Why initialization matters

Different starting centroids lead to different final clusterings. Three runs on the same data might end at inertias of 12.645, 12.943 and 13.112: all "converged", different quality. One random run isn't enough.

> [!TIP]
> **Intuition**
>
> The first assignments depend on where the centroids start, the first update depends on those assignments, and so on. A bad start, such as two centroids in the same natural cluster, can lock in a bad split.

In practice run K-means several times (scikit-learn's `n_init`) and keep the run with the lowest inertia.

## 9.7 K-means++

A smarter seeding that spreads the initial centroids out:

1. Pick the first centroid uniformly at random from the data.
2. For each point compute $D(x^{(i)})$, its distance to the nearest centroid chosen so far.
3. Pick the next centroid with probability proportional to $D(x^{(i)})^2$:
   $$ p\left(x^{(i)}\right)=\frac{D\left(x^{(i)}\right)^2}{\sum_{r}D\left(x^{(r)}\right)^2}. $$
4. Repeat until there are $K$ centroids, then run ordinary K-means.

Points far from every current centroid are likely to be chosen, so seeds tend to land in different clusters. K-means++ also comes with a guarantee: its expected inertia is within a factor $O(\log K)$ of optimal (Arthur & Vassilvitskii, 2007).

## 9.8 Choosing K

**From the application.** Sometimes $K$ is dictated by the use case: group jobs across 4 CPU cores ($K=4$), or design clothing in 10 sizes ($K=10$).

**Elbow method.** Plot inertia against $K$. Inertia always falls as $K$ grows (at $K=n$ it is 0), so look for the **elbow**, the point after which extra clusters buy little.

> [!WARNING]
> **Lowest inertia doesn't pick K**
>
> Choosing the $K$ with the smallest inertia always picks the largest $K$ you tried. The elbow looks for diminishing returns instead. The elbow is often ambiguous; the silhouette score or the gap statistic give a second opinion.

## 9.9 Where this leads

- **K-means → a probabilistic version → EM.** Replacing hard assignments with probabilities gives Gaussian mixture models, trained with the EM algorithm ([Chapter 10](10_gaussian_mixture_models_and_em.md)). K-means is the limit of a GMM with equal, spherical covariances shrinking to zero.
- **Letting K be learned.** A Dirichlet-process mixture is a nonparametric Bayesian model in which the number of clusters grows with the data instead of being fixed in advance.

> [!NOTE]
> **Beyond the lecture: what K-means assumes**
>
> Minimizing squared Euclidean distance to a centroid implicitly assumes clusters that are roughly spherical, of similar size and similar density. K-means will split an elongated cluster and merge two small nearby ones. For other shapes use GMMs (elliptical clusters), DBSCAN (arbitrary shapes, with noise) or spectral clustering. Features need scaling, as for KNN.

## 9.10 Common mistakes

1. **Treating cluster IDs as meaningful labels.** Cluster "1" and "2" can be swapped freely.
2. **Recomputing a centroid from all points** instead of its own cluster.
3. **Stopping after one assignment step.**
4. **Assuming convergence means the best clustering.** It is a local optimum.
5. **Choosing K by minimum inertia.**
6. **Forgetting to scale features.**

## 9.11 Summary

- Unsupervised learning finds structure in unlabeled inputs: clustering or dimensionality reduction.
- K-means alternates nearest-centroid assignment and mean updates, which is coordinate descent on inertia.
- It converges to a local optimum that depends on initialization; K-means++ seeding and multiple restarts help.
- Choose $K$ from the application or from the elbow of the inertia curve.

## 9.12 Self-check

1. Why can't the K-means objective increase during the loop?
2. Why is the centroid update the mean?
3. How does K-means++ pick the next seed, and why squared distance?
4. Why does inertia always decrease with $K$?
5. What cluster shapes does K-means handle poorly?

<details>
<summary>Answers</summary>

1. Each step exactly minimizes $J$ over one block of variables with the other fixed.
2. The mean minimizes $\sum_i\lVert x^{(i)}-\mu\rVert^2$ (set the gradient $-2\sum_i(x^{(i)}-\mu)$ to zero).
3. With probability proportional to $D(x)^2$, which strongly favors far-away points while still being random.
4. More centroids can only bring each point closer to its nearest centroid; $K=n$ gives zero.
5. Elongated, non-convex, or very unequal-size/density clusters.

</details>
