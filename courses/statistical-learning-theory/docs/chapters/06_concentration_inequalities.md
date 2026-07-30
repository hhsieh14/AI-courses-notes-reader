---
course: "Statistical Learning Theory"
chapter: "06"
title: "Concentration Inequalities"
source_pages: "598SLT.pdf, pp. 24-37"
status: "consolidated v1.0"
---

# Concentration Inequalities

**Source:** 598SLT.pdf, pp. 24-37.

## 1. Why concentration matters in learning

For a prediction function $f$, the population risk is

$$R(f)=\mathbb{E}_{(X,Y)\sim P}\left[\ell(f(X),Y)\right].$$

Given an independent training sample

$$\left\lbrace (X_{i},Y_{i})\right\rbrace_{i=1}^{n},$$

the empirical risk is

$$\widehat{R}_{n}(f)=\frac{1}{n}\sum_{i=1}^{n}\ell(f(X_{i}),Y_{i}).$$

Many learning algorithms choose a function by empirical-risk minimization:

$$f_{n}\in\underset{f\in\mathcal{F}}{\arg\min}\;\widehat{R}_{n}(f).$$

The soft-margin SVM from the previous chapters is an example of empirical loss plus regularization:

$$\min_{w\in\mathcal{H},b}\;\sum_{i=1}^{n}\Phi\left(y_{i}\left(\langle w,\phi(x_{i})\rangle_{\mathcal{H}}+b\right)\right)+\lambda\lVert w\rVert_{\mathcal{H}}^{2}.$$

The central question is why a small empirical risk should imply a small population risk. Concentration inequalities answer this by controlling the probability that a random empirical quantity is far from its expectation.

The notes first present the schematic type of condition

$$P\left(R(f)>\widehat{R}_{n}(f)+\varepsilon\right)\leq \exp\left(-c\varepsilon^{2}n\right),$$

where $c>0$ is a constant.

!!! clarification "Pointwise versus data-dependent functions"
    A bound stated for every fixed $f\in\mathcal{F}$ does not automatically justify substituting the data-dependent minimizer $f_{n}$. To control a function selected after seeing the sample, the later chapters seek a uniform bound that holds simultaneously over the entire class $\mathcal{F}$.

## 2. The expectation of empirical risk is population risk

Let $P^{n}$ denote the joint distribution of the $n$ independent training examples. Then

$$\mathbb{E}_{\{(X_{i},Y_{i})\}_{i=1}^{n}\sim P^{n}}\left[\widehat{R}_{n}(f)\right]=\mathbb{E}\left[\frac{1}{n}\sum_{i=1}^{n}\ell(f(X_{i}),Y_{i})\right].$$

By linearity of expectation,

$$\mathbb{E}\left[\widehat{R}_{n}(f)\right]=\frac{1}{n}\sum_{i=1}^{n}\mathbb{E}\left[\ell(f(X_{i}),Y_{i})\right].$$

Every training pair has distribution $P$, so each term has the same expectation:

$$\mathbb{E}\left[\ell(f(X_{i}),Y_{i})\right]=R(f).$$

Therefore,

$$\mathbb{E}\left[\widehat{R}_{n}(f)\right]=R(f).$$

Thus, the learning problem becomes a concentration problem: under what assumptions is $\widehat{R}_{n}(f)$ close to its expectation $R(f)$?

## 3. Recap: Markov and Chebyshev inequalities

### Markov inequality

If $Z$ is a nonnegative random variable and $t>0$, then

$$P(Z\geq t)\leq \frac{\mathbb{E}[Z]}{t}.$$

The proof follows from

$$Z\geq t\mathbf{1}_{\{Z\geq t\}},$$

and therefore

$$\mathbb{E}[Z]\geq tP(Z\geq t).$$

### Chebyshev inequality

For any random variable $X$ and any $t>0$,

$$P\left(\lvert X-\mathbb{E}[X]\rvert\geq t\right)\leq \frac{\mathrm{Var}(X)}{t^{2}}.$$

Apply Markov's inequality to the nonnegative random variable

$$Z=(X-\mathbb{E}[X])^{2}.$$

Then

$$P\left(\lvert X-\mathbb{E}[X]\rvert\geq t\right)=P\left((X-\mathbb{E}[X])^{2}\geq t^{2}\right)\leq \frac{\mathbb{E}[(X-\mathbb{E}[X])^{2}]}{t^{2}}.$$

The numerator is $\mathrm{Var}(X)$.

!!! note "Why seek a stronger inequality?"
    Chebyshev gives a polynomial tail proportional to $1/t^{2}$. For bounded independent variables, Hoeffding's inequality gives an exponentially decreasing tail, which is much more useful for learning-theory bounds.

## 4. Exponential Markov method

Let $X_{1},\ldots,X_{n}$ be independent random variables and define

$$S_{n}=\sum_{i=1}^{n}X_{i}.$$

For any $s>0$, the exponential function is increasing, so

$$S_{n}-\mathbb{E}[S_{n}]\geq t$$

is equivalent to

$$\exp\left(s(S_{n}-\mathbb{E}[S_{n}])\right)\geq \exp(st).$$

Markov's inequality gives

$$P\left(S_{n}-\mathbb{E}[S_{n}]\geq t\right)\leq \frac{\mathbb{E}\left[\exp\left(s(S_{n}-\mathbb{E}[S_{n}])\right)\right]}{\exp(st)}.$$

Because

$$S_{n}-\mathbb{E}[S_{n}]=\sum_{i=1}^{n}(X_{i}-\mathbb{E}[X_{i}]),$$

and the centered variables remain independent,

$$\mathbb{E}\left[\exp\left(s(S_{n}-\mathbb{E}[S_{n}])\right)\right]=\prod_{i=1}^{n}\mathbb{E}\left[\exp\left(s(X_{i}-\mathbb{E}[X_{i}])\right)\right].$$

The remaining task is to bound each centered moment-generating function.

## 5. Hoeffding's lemma

### Lemma 1: Hoeffding's lemma

Let $X$ be a random variable satisfying

$$\mathbb{E}[X]=0$$

and

$$a\leq X\leq b.$$

Then, for every $s>0$,

$$\mathbb{E}[\exp(sX)]\leq \exp\left(\frac{s^{2}(b-a)^{2}}{8}\right).$$

The key ingredient is the convexity of $x\mapsto\exp(sx)$.

## 6. Proof of Hoeffding's lemma

For each $x\in[a,b]$, write

$$x=pa+qb,$$

where

$$p=\frac{b-x}{b-a}$$

and

$$q=\frac{x-a}{b-a}.$$

Here $p,q\in[0,1]$ and $p+q=1$. Convexity gives

$$\exp(sx)\leq \frac{b-x}{b-a}\exp(sa)+\frac{x-a}{b-a}\exp(sb).$$

Taking expectations and using $\mathbb{E}[X]=0$,

$$\mathbb{E}[\exp(sX)]\leq \frac{b}{b-a}\exp(sa)-\frac{a}{b-a}\exp(sb).$$

Define

$$\theta=-\frac{a}{b-a}$$

and

$$u=s(b-a).$$

Since $\mathbb{E}[X]=0$ and $X\in[a,b]$, the interval must contain zero, so $\theta\in[0,1]$. Also,

$$sa=-\theta u.$$

The preceding upper bound becomes

$$\mathbb{E}[\exp(sX)]\leq \exp(-\theta u)\left(1-\theta+\theta\exp(u)\right).$$

Define

$$f_{\theta}(u)=\log\left(\exp(-\theta u)\left(1-\theta+\theta\exp(u)\right)\right).$$

The source computes

$$f_{\theta}(0)=0$$

and

$$f_{\theta}'(0)=0.$$

Its second derivative can be written as

$$f_{\theta}''(u)=t(1-t),$$

where

$$t=\frac{\theta\exp(u)}{1-\theta+\theta\exp(u)}.$$

Because $t\in[0,1]$,

$$f_{\theta}''(u)\leq \frac{1}{4}.$$

Taylor's theorem with a second-order remainder gives, for some intermediate point between $0$ and $u$,

$$f_{\theta}(u)=f_{\theta}(0)+f_{\theta}'(0)u+\frac{f_{\theta}''(\xi)}{2}u^{2}.$$

Therefore,

$$f_{\theta}(u)\leq \frac{u^{2}}{8}.$$

Exponentiating and substituting $u=s(b-a)$ yields

$$\mathbb{E}[\exp(sX)]\leq \exp\left(\frac{s^{2}(b-a)^{2}}{8}\right).$$

!!! clarification "Why does the proof stop at second order?"
    The handwritten notes ask why an infinite Taylor series is unnecessary. The source uses Taylor's theorem with a remainder evaluated at an intermediate point $\xi$, not an infinite-series approximation. Once the second derivative is uniformly bounded by $1/4$, the second-order remainder already provides the required global upper bound.

## 7. Hoeffding's inequality

Assume the independent variables satisfy

$$a_{i}\leq X_{i}\leq b_{i}$$

for every $1\leq i\leq n$. The centered variable $X_{i}-\mathbb{E}[X_{i}]$ has expectation zero and lies in an interval of the same width $b_{i}-a_{i}$. Hoeffding's lemma gives

$$\mathbb{E}\left[\exp\left(s(X_{i}-\mathbb{E}[X_{i}])\right)\right]\leq \exp\left(\frac{s^{2}(b_{i}-a_{i})^{2}}{8}\right).$$

Substituting these bounds into the exponential Markov inequality gives

$$P\left(S_{n}-\mathbb{E}[S_{n}]\geq t\right)\leq \exp\left(\frac{s^{2}}{8}\sum_{i=1}^{n}(b_{i}-a_{i})^{2}-st\right).$$

This inequality holds for every $s>0$. Let

$$V=\sum_{i=1}^{n}(b_{i}-a_{i})^{2}.$$

The exponent

$$\frac{V}{8}s^{2}-ts$$

is minimized at

$$s^{\star}=\frac{4t}{V}.$$

Substitution gives the upper-tail bound

$$P\left(S_{n}-\mathbb{E}[S_{n}]\geq t\right)\leq \exp\left(-\frac{2t^{2}}{\sum_{i=1}^{n}(b_{i}-a_{i})^{2}}\right).$$

Applying the same result to $-X_{1},\ldots,-X_{n}$ gives

$$P\left(\mathbb{E}[S_{n}]-S_{n}\geq t\right)\leq \exp\left(-\frac{2t^{2}}{\sum_{i=1}^{n}(b_{i}-a_{i})^{2}}\right).$$

By the union bound,

$$P\left(\lvert S_{n}-\mathbb{E}[S_{n}]\rvert\geq t\right)\leq 2\exp\left(-\frac{2t^{2}}{\sum_{i=1}^{n}(b_{i}-a_{i})^{2}}\right).$$

## 8. Application to empirical risk

Fix a prediction function $f$ and assume

$$0\leq \ell(f(x),y)\leq 1$$

for every possible input-output pair. Define

$$X_{i}=\frac{1}{n}\ell(f(X_{i}^{\mathrm{data}}),Y_{i}^{\mathrm{data}}).$$

Then

$$0\leq X_{i}\leq \frac{1}{n}$$

and

$$\sum_{i=1}^{n}X_{i}=\widehat{R}_{n}(f).$$

Since

$$\sum_{i=1}^{n}\left(\frac{1}{n}-0\right)^{2}=\frac{1}{n},$$

Hoeffding's inequality gives

$$P\left(\widehat{R}_{n}(f)-R(f)\geq t\right)\leq \exp(-2nt^{2})$$

and

$$P\left(R(f)-\widehat{R}_{n}(f)\geq t\right)\leq \exp(-2nt^{2}).$$

Therefore,

$$P\left(\lvert\widehat{R}_{n}(f)-R(f)\rvert\geq t\right)\leq 2\exp(-2nt^{2}).$$

Equivalently, with probability at least

$$1-2\exp(-2nt^{2}),$$

we have

$$\lvert\widehat{R}_{n}(f)-R(f)\rvert<t.$$

!!! clarification "Why require the loss to lie in $[0,1]$?"
    The width of each summand's interval determines the Hoeffding exponent. The ordinary hinge loss $\max\{1-y\widehat{y},0\}$ is not bounded above, so this particular application does not apply to it directly. The notes mention the clipped loss $\min\{\max\{1-y\widehat{y},0\},1\}$ as one bounded alternative.

## 9. Functions of bounded difference

Hoeffding's inequality applies directly to sums of independent bounded variables. Many learning statistics are more general functions

$$g(X_{1},\ldots,X_{n}).$$

### Definition: Bounded difference

A function $g:\mathcal{X}^{n}\to\mathbb{R}$ has bounded differences with constants $c_{1},\ldots,c_{n}$ when changing only coordinate $i$ can change the function by at most $c_{i}$. Formally, for every $i$,

$$\sup_{x_{1},\ldots,x_{n},x_{i}'}\left\lvert g(x_{1},\ldots,x_{i},\ldots,x_{n})-g(x_{1},\ldots,x_{i}',\ldots,x_{n})\right\rvert\leq c_{i}.$$

### Example: A sum

If

$$g(x_{1},\ldots,x_{n})=\sum_{i=1}^{n}x_{i}$$

and $x_{i}\in[a_{i},b_{i}]$, then changing only coordinate $i$ changes the sum by at most

$$c_{i}=b_{i}-a_{i}.$$

### Example: Empirical loss

For

$$\widehat{R}_{n}(f)=\frac{1}{n}\sum_{i=1}^{n}\ell(f(X_{i}),Y_{i}),$$

if $\ell\in[0,1]$, replacing one training example changes the average by at most

$$c_{i}=\frac{1}{n}.$$

## 10. McDiarmid's inequality

### Theorem 1: McDiarmid's inequality

Let

$$X^{n}=(X_{1},\ldots,X_{n})$$

contain independent random variables. Suppose $g:\mathcal{X}^{n}\to\mathbb{R}$ has bounded differences $c_{1},\ldots,c_{n}$. Then, for every $t>0$,

$$P\left(g(X^{n})-\mathbb{E}[g(X^{n})]\geq t\right)\leq \exp\left(-\frac{2t^{2}}{\sum_{i=1}^{n}c_{i}^{2}}\right).$$

Similarly,

$$P\left(\mathbb{E}[g(X^{n})]-g(X^{n})\geq t\right)\leq \exp\left(-\frac{2t^{2}}{\sum_{i=1}^{n}c_{i}^{2}}\right).$$

Thus,

$$P\left(\lvert g(X^{n})-\mathbb{E}[g(X^{n})]\rvert\geq t\right)\leq 2\exp\left(-\frac{2t^{2}}{\sum_{i=1}^{n}c_{i}^{2}}\right).$$

The course states that the detailed proof is in the reference material and is not required. It provides the following proof structure.

## 11. Sketch of the McDiarmid proof

Define the progressively revealed sample

$$X^{i}=(X_{1},\ldots,X_{i}).$$

Define the martingale differences

$$V_{i}=\mathbb{E}[g(X^{n})\mid X^{i}]-\mathbb{E}[g(X^{n})\mid X^{i-1}].$$

The terms telescope:

$$\sum_{i=1}^{n}V_{i}=g(X^{n})-\mathbb{E}[g(X^{n})].$$

They also satisfy the conditional mean-zero property

$$\mathbb{E}[V_{i}\mid X^{i-1}]=0.$$

The bounded-difference property implies that, after conditioning on $X^{i-1}$, the possible values of $V_{i}$ lie in an interval whose width is at most $c_{i}$. Conditional Hoeffding's lemma therefore gives

$$\mathbb{E}[\exp(sV_{i})\mid X^{i-1}]\leq \exp\left(\frac{s^{2}c_{i}^{2}}{8}\right).$$

Applying conditional expectation iteratively yields

$$\mathbb{E}\left[\exp\left(s\sum_{i=1}^{n}V_{i}\right)\right]\leq \exp\left(\frac{s^{2}}{8}\sum_{i=1}^{n}c_{i}^{2}\right).$$

The same exponential Markov argument and the same optimization over $s$ as in Hoeffding's inequality complete the proof.

!!! note "Where is the difficult step?"
    The handwritten notes flag that bounding the change in $g$ is not immediately the same as bounding the conditional range of $V_{i}$. The reference proof constructs conditional lower and upper envelopes for the $i$th revealed coordinate and shows that their gap is at most $c_{i}$. This is the technical bridge needed before applying the conditional Hoeffding lemma.

## 12. Hoeffding as a special case of McDiarmid

Take

$$g(X^{n})=\sum_{i=1}^{n}X_{i}.$$

If $X_{i}\in[a_{i},b_{i}]$, then $g$ has bounded-difference constants

$$c_{i}=b_{i}-a_{i}.$$

McDiarmid's inequality becomes

$$P\left(g(X^{n})-\mathbb{E}[g(X^{n})]\geq t\right)\leq \exp\left(-\frac{2t^{2}}{\sum_{i=1}^{n}(b_{i}-a_{i})^{2}}\right),$$

which is Hoeffding's upper-tail bound. The lower-tail result follows in the same way.

## 13. Estimating the probability of a fixed set

Let $P$ be a probability distribution on $\mathcal{X}$, and let $A\subseteq\mathcal{X}$ be measurable. Define the indicator

$$\delta_{x}(A)=\mathbf{1}_{\{x\in A\}}.$$

For independent observations $X_{1},\ldots,X_{n}\sim P$, define the empirical probability

$$P_{n}(A)=\frac{1}{n}\sum_{i=1}^{n}\delta_{X_{i}}(A).$$

Its expectation is

$$\mathbb{E}[P_{n}(A)]=\frac{1}{n}\sum_{i=1}^{n}\mathbb{E}[\delta_{X_{i}}(A)]=P(A).$$

Changing one observation can change $P_{n}(A)$ by at most $1/n$, so

$$c_{i}=\frac{1}{n}.$$

McDiarmid's inequality gives

$$P\left(\lvert P_{n}(A)-P(A)\rvert\geq t\right)\leq 2\exp(-2nt^{2}).$$

Thus, for a fixed set $A$, with probability at least

$$1-2\exp(-2nt^{2}),$$

we have

$$\lvert P_{n}(A)-P(A)\rvert<t.$$

## 14. Why separate bounds do not handle infinitely many sets

For two fixed sets $A$ and $B$, the union bound gives simultaneous accuracy with failure probability at most

$$4\exp(-2nt^{2}).$$

For a finite collection

$$\mathcal{A}=\{A_{1},\ldots,A_{M}\},$$

the union bound gives

$$P\left(\text{there exists }A\in\mathcal{A}\text{ with }\lvert P_{n}(A)-P(A)\rvert\geq t\right)\leq 2M\exp(-2nt^{2}).$$

As $M$ becomes very large, this bound weakens. For an infinite class, directly multiplying by the number of sets is not meaningful.

This motivates controlling the worst deviation as one random variable rather than bounding each set separately.

## 15. Uniform deviation over a class of sets

Define

$$g(X^{n})=\sup_{A\in\mathcal{A}}\lvert P_{n}(A)-P(A)\rvert,$$

where $\mathcal{A}$ may be infinite.

The source uses the useful identities

$$\sup_{a\in A}h_{1}(a)-\inf_{b\in B}h_{2}(b)=\sup_{a\in A,b\in B}(h_{1}(a)-h_{2}(b))$$

and

$$\inf_{a\in A}h_{1}(a)-\sup_{b\in B}h_{2}(b)=\inf_{a\in A,b\in B}(h_{1}(a)-h_{2}(b)).$$

When one sample coordinate is replaced, for every fixed $A$ the empirical probability changes by at most $1/n$. Taking the supremum cannot increase that coordinate sensitivity, so

$$\left\lvert g(x_{1},\ldots,x_{i},\ldots,x_{n})-g(x_{1},\ldots,x_{i}',\ldots,x_{n})\right\rvert\leq \frac{1}{n}.$$

Therefore, $g$ has bounded-difference constants

$$c_{i}=\frac{1}{n}.$$

McDiarmid's inequality gives

$$P\left(\lvert g(X^{n})-\mathbb{E}[g(X^{n})]\rvert\geq t\right)\leq 2\exp(-2nt^{2}).$$

Hence, with probability at least $1-2\exp(-2nt^{2})$,

$$g(X^{n})\leq \mathbb{E}[g(X^{n})]+t.$$

Equivalently, simultaneously for every $A\in\mathcal{A}$,

$$\lvert P_{n}(A)-P(A)\rvert\leq \mathbb{E}\left[\sup_{A\in\mathcal{A}}\lvert P_{n}(A)-P(A)\rvert\right]+t.$$

The concentration step is now complete. The remaining challenge is to bound the expectation

$$\mathbb{E}\left[\sup_{A\in\mathcal{A}}\lvert P_{n}(A)-P(A)\rvert\right],$$

which measures the complexity of the class $\mathcal{A}$. The next chapter develops tools for this purpose.

!!! clarification "What changed compared with the finite union bound?"
    Instead of assigning a separate failure event to every set and summing those probabilities, the supremum is treated as a single random function of the sample. McDiarmid controls its fluctuation around its expectation even when the class is infinite. The size of the class has not disappeared; it is now encoded in the expected supremum.

## 16. Reference example: Kernel density estimation

The final page gives a reference-level application that the course says is not required in detail.

Let $X_{1},\ldots,X_{n}$ be independent real-valued observations from a density $f$. Let $K:\mathbb{R}\to\mathbb{R}$ be nonnegative and satisfy

$$\int_{\mathbb{R}}K(u)\,du=1.$$

For bandwidth $h>0$, define the kernel density estimator

$$\widehat{f}_{n}(x)=\frac{1}{nh}\sum_{i=1}^{n}K\left(\frac{x-X_{i}}{h}\right).$$

Measure error with the $L^{1}$ distance

$$g(X^{n})=\lVert\widehat{f}_{n}-f\rVert_{1}=\int_{\mathbb{R}}\lvert\widehat{f}_{n}(x)-f(x)\rvert\,dx.$$

If only observation $X_{i}$ is replaced by $X_{i}'$, the reverse triangle inequality gives

$$\left\lvert g(X^{n})-g((X^{n})')\right\rvert\leq \lVert\widehat{f}_{n}-\widehat{f}_{n}'\rVert_{1}.$$

Only one kernel term changes, so

$$\lVert\widehat{f}_{n}-\widehat{f}_{n}'\rVert_{1}\leq \frac{1}{nh}\int_{\mathbb{R}}\left\lvert K\left(\frac{x-X_{i}}{h}\right)-K\left(\frac{x-X_{i}'}{h}\right)\right\rvert\,dx.$$

Because $K$ is nonnegative and integrates to one, each shifted and rescaled kernel has integral $h$, and therefore

$$\lVert\widehat{f}_{n}-\widehat{f}_{n}'\rVert_{1}\leq \frac{2}{n}.$$

Thus, $c_{i}=2/n$ and

$$\sum_{i=1}^{n}c_{i}^{2}=\frac{4}{n}.$$

McDiarmid's inequality yields

$$P\left(g(X^{n})-\mathbb{E}[g(X^{n})]\geq t\right)\leq \exp\left(-\frac{nt^{2}}{2}\right).$$

This shows that the random $L^{1}$ error concentrates around its expected value. A separate argument is still needed to bound that expectation.

## Chapter summary

- Empirical risk is an unbiased estimate of population risk for a fixed prediction function.
- Exponential Markov bounds convert moment-generating-function estimates into tail probabilities.
- Hoeffding's lemma bounds the centered moment-generating function of a bounded variable.
- Hoeffding's inequality gives exponential concentration for sums of independent bounded variables.
- McDiarmid's inequality extends the same idea to general functions with bounded coordinate sensitivity.
- Fixed-set empirical probabilities concentrate at rate $\exp(-2nt^{2})$.
- Uniform control over an infinite class is obtained by concentrating the supremum around its expectation.
- The expected supremum becomes the complexity term that must be analyzed in the next chapter.

---

[← Previous: RKHS and the Representer Theorem](05_rkhs_and_representer.md) · [Course map](../course_map.md) · [Next: Generalization Bounds and Rademacher Complexity →](07_generalization_and_rademacher.md)
