---
title: "Probability and Statistics"
---

> **Level 0 · Module 4** · Prerequisites: L0-01 · Time: 6 to 8 hours · Code: `code/L0-04-probability-and-statistics/`

## What you will be able to do

- Describe common distributions by their mean, variance and typical ML use, and check those properties by simulation.
- Apply Bayes' rule, and derive maximum likelihood and MAP estimates for simple models.
- Compute entropy, cross-entropy and KL divergence, and explain why the cross-entropy loss is maximum likelihood.
- Estimate expectations by Monte Carlo, predict how the error shrinks, and recognize when importance sampling fails.
- Build and interpret confidence intervals and p-values, and compare two models on one test set with a paired test.

## 1. Random variables and distributions

A **random variable** is a quantity whose value depends on chance, such as a model's error on a random test example or the next token a language model samples. Its **distribution** says how likely each value is.

A discrete variable has a **probability mass function** (PMF) $P(X = x)$, summing to 1. A continuous variable has a **probability density function** (PDF) $p(x)$, and probabilities are areas: $P(a \le X \le b) = \int_a^b p(x)\,dx$. A density can therefore exceed 1 (the uniform density on $[0, 0.5]$ equals 2). The distributions you will meet constantly:

| Distribution | Values | Mean | Variance | Where it shows up in ML |
|---|---|---|---|---|
| Bernoulli($p$) | $\{0, 1\}$ | $p$ | $p(1-p)$ | one label; one test example right or wrong |
| Binomial($n, p$) | $0..n$ | $np$ | $np(1-p)$ | number correct out of $n$ test examples |
| Categorical($\mathbf{p}$) | $1..K$ | | | a class label; the next token |
| Poisson($\lambda$) | $0, 1, 2, ...$ | $\lambda$ | $\lambda$ | counts per interval (clicks, arrivals) |
| Uniform($a, b$) | $[a, b]$ | $(a+b)/2$ | $(b-a)^2/12$ | initialization, random search |
| Normal($\mu, \sigma^2$) | $\mathbb{R}$ | $\mu$ | $\sigma^2$ | noise models, sums of many small effects |
| Exponential($\lambda$) | $[0, \infty)$ | $1/\lambda$ | $1/\lambda^2$ | waiting times, latencies |

The normal density is $p(x) = \frac{1}{\sqrt{2\pi\sigma^2}}\exp\!\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$. Script 1 samples 200,000 values from each distribution and recovers these means and variances to two or three decimals.

## 2. Expectation, variance, and why averages work

The **expectation** $\mathbb{E}[X] = \sum_x x\,P(X = x)$ (or $\int x\,p(x)\,dx$) is the probability-weighted average. The **variance** $\mathrm{Var}(X) = \mathbb{E}[(X - \mathbb{E}X)^2]$ measures spread; its square root is the **standard deviation**. Expectation is **linear**, $\mathbb{E}[aX + bY] = a\mathbb{E}X + b\mathbb{E}Y$, even for dependent variables, and for **independent** variables variances add.

Together these give the most useful formula in this module. The mean $\bar X$ of $n$ independent draws with variance $\sigma^2$ has variance $\sigma^2/n$, so its **standard error** is

$$\mathrm{SE}(\bar X) = \frac{\sigma}{\sqrt{n}}.$$

A model with true accuracy 0.9 has per-example standard deviation $\sqrt{0.9 \times 0.1} = 0.3$, so on 100 test examples its measured accuracy has standard error 0.03: a reported 0.90 could easily be 0.87 or 0.93. On 10,000 examples the standard error is 0.003. Halving the error costs four times the data.

The **law of large numbers** says $\bar X \to \mathbb{E}X$ as $n$ grows. The **central limit theorem** (CLT) adds that for large $n$, $\bar X$ is approximately normal with standard deviation $\sigma/\sqrt n$, whatever the original distribution. Two caveats, both visible in script 1:

- **"Large" depends on the distribution.** The exponential distribution has skewness 2, and the mean of $n$ draws has skewness $2/\sqrt n$. At $n = 30$, a popular rule-of-thumb threshold, the standardized mean's two tails still hold 1.5% and 3.4% instead of 2.5% each.
- **The variance must exist.** The Cauchy distribution has tails so heavy that its mean is undefined, and the running average of a million draws keeps jumping. Heavy-tailed quantities in ML (per-example losses with rare huge values, latencies, some RL rewards) can sit closer to this than to the textbook picture.

## 3. Conditional probability and Bayes' rule

The conditional probability $P(A \mid B) = P(A \cap B)/P(B)$ is the probability of $A$ once you know $B$ happened. Writing $P(A \cap B)$ both ways gives **Bayes' rule**:

$$P(H \mid D) = \frac{P(D \mid H)\,P(H)}{P(D)}, \qquad P(D) = \sum_{h} P(D \mid h)\,P(h).$$

$P(H)$ is the **prior** for hypothesis $H$, $P(D \mid H)$ the **likelihood** of data $D$, $P(H \mid D)$ the **posterior**, and $P(D)$ the **evidence**, computed by summing over hypotheses.

Worked example: a condition affects 1% of people; a test detects it 99% of the time with a 5% false-positive rate. After a positive test,

$$P(\text{condition} \mid +) = \frac{0.99 \times 0.01}{0.99 \times 0.01 + 0.05 \times 0.99} = \frac{0.0099}{0.0594} \approx 0.167.$$

Only one positive in six is real: false positives from the healthy 99% outnumber true positives from the 1%. Script 2 confirms it by simulating a million people. Classifiers for rare classes obey the same arithmetic: **precision depends on the base rate**, so a fraud model that looks excellent on a balanced benchmark can produce mostly false alarms in deployment (L1-01).

## 4. Maximum likelihood and MAP

A model with parameters $\theta$ assigns probability to data. The **likelihood** is that probability as a function of $\theta$ with the data fixed: $L(\theta) = \prod_i p(x_i \mid \theta)$ for independent observations. **Maximum likelihood estimation** (MLE) picks the $\theta$ that makes the observed data most probable. Products of many small numbers underflow (L0-01), so in practice you maximize the **log-likelihood** $\ell(\theta) = \sum_i \log p(x_i \mid \theta)$.

**Coin.** With $k$ heads in $n$ flips, $\ell(\theta) = k\log\theta + (n-k)\log(1-\theta)$. Setting $\ell'(\theta) = k/\theta - (n-k)/(1-\theta) = 0$ gives $\hat\theta_{\text{MLE}} = k/n$.

**Gaussian noise.** If $y_i = f_\theta(x_i) + \varepsilon_i$ with Gaussian $\varepsilon_i$ of fixed variance $\sigma^2$, then $-\ell(\theta) = \frac{1}{2\sigma^2}\sum_i (y_i - f_\theta(x_i))^2 + \text{const}$. Minimizing squared error *is* maximum likelihood under Gaussian noise, which is also why squared error is so sensitive to outliers.

MLE can be **overconfident with little data** (3 heads in 3 flips gives $\hat\theta = 1$: tails are "impossible") and can be **biased**: the Gaussian MLE of variance divides by $n$ and has expectation $\frac{n-1}{n}\sigma^2$. In script 2 its average at $n = 5$ is 3.2 for a true variance of 4.

**MAP estimation** (maximum a posteriori) maximizes $\log p(D \mid \theta) + \log p(\theta)$, adding a prior. With a Beta($a, b$) prior, the coin's posterior is Beta($k + a$, $n - k + b$), whose mode is

$$\hat\theta_{\text{MAP}} = \frac{k + a - 1}{n + a + b - 2}.$$

The prior acts like $a - 1$ imaginary heads and $b - 1$ imaginary tails. With $a = b = 2$, 3 heads in 3 flips gives $4/5 = 0.8$; at 700 heads in 1000 the pseudo-counts are swamped and MLE and MAP agree to three decimals (script 2). In ML, a Gaussian prior on weights adds a $-\lambda\|\theta\|^2$ term, which is L2 regularization (weight decay), and a Laplace prior gives L1 (L1-02). Regularized training is MAP estimation.

A fully **Bayesian** treatment keeps the whole posterior: after 3 heads in 3 flips, the central 95% posterior interval is $[0.36, 0.96]$, an honest "we know very little". Frequentist and Bayesian statisticians disagree about what probability means; ML practice borrows from both, and regularizer priors are usually chosen for convenience rather than belief.

## 5. Entropy, cross-entropy and KL divergence

[Shannon, 1948] defined the **entropy** of a discrete distribution $p$ as

$$H(p) = -\sum_x p(x) \log p(x),$$

measured in bits with $\log_2$ and in nats with the natural log. It measures uncertainty, equivalently the average length of an optimal code. A fair coin has 1 bit, a fair 8-sided die 3 bits, a 90/10 coin 0.469 bits, a certain outcome 0 (script 3).

The **cross-entropy** $H(p, q) = -\sum_x p(x)\log q(x)$ is the average code length when outcomes come from $p$ but the code was built for $q$. The excess is the **Kullback-Leibler (KL) divergence**:

$$\mathrm{KL}(p \,\|\, q) = H(p, q) - H(p) = \sum_x p(x) \log\frac{p(x)}{q(x)}.$$

KL is never negative and is zero only when $q = p$. It is **not a distance**: it is asymmetric, and infinite if $q$ gives zero probability to something $p$ allows. For $p = (0.7, 0.2, 0.1)$ and $q = (0.4, 0.4, 0.2)$, $\mathrm{KL}(p\|q) = 0.265$ bits but $\mathrm{KL}(q\|p) = 0.277$ bits.

**Why this is your training loss.** For a classification example with true label $y$, the target is one-hot, so the cross-entropy against the predicted distribution $q$ collapses to $-\log q(y)$, the negative log-likelihood of the label. The average cross-entropy loss is the average negative log-likelihood, so minimizing it is MLE; and because the data's own entropy $H(\hat p)$ does not depend on the model, it also minimizes $\mathrm{KL}(\hat p \,\|\, q)$. In script 3, the distribution with the lowest cross-entropy on 1000 samples is the sample's own frequencies, which beats even the true generating distribution (1.5043 vs 1.5055 bits): overfitting in miniature.

**Direction matters.** Fit one Gaussian $q$ to a mixture $p$ with bumps at $-3$ and $3$. Minimizing the **forward** $\mathrm{KL}(p\|q)$, as maximum likelihood does, gives $\mu = 0, \sigma = 3.15$: one wide Gaussian covering both bumps and centered where $p$ has almost no mass. Minimizing the **reverse** $\mathrm{KL}(q\|p)$ gives $\mu = -3, \sigma = 1$: it locks onto one bump (the other fits equally well; the grid search found this one first). Forward KL is punished wherever $p > 0$ but $q \approx 0$, so it covers mass; reverse KL is punished wherever $q > 0$ but $p \approx 0$, so it seeks a mode. The distinction returns in generative models (L3-03) and in the KL penalties of post-training (L3-02).

## 6. Sampling and Monte Carlo

**Monte Carlo** estimation replaces an expectation by a sample average:

$$\mathbb{E}_{p}[g(X)] \approx \frac{1}{N}\sum_{i=1}^N g(x_i), \qquad x_i \sim p.$$

By section 2 the error is about $\sigma_g/\sqrt N$ in any dimension, but each extra decimal digit costs 100 times more samples. In script 4, estimating $\pi = \mathbb{E}[4/(1 + U^2)]$ with uniform $U$, the RMS error times $\sqrt N$ stays near 0.62 from $N = 100$ to $100{,}000$, and the standard error each run reports about itself matches the real error.

**Rare events** defeat plain Monte Carlo. $P(Z > 4)$ for a standard normal is about $3.2 \times 10^{-5}$; 100,000 samples produced one hit, an estimate off by a factor of three. **Importance sampling** draws from a proposal $q$ that visits the region often and reweights by $w = p(x)/q(x)$, using $\mathbb{E}_p[g(X)] = \mathbb{E}_q[g(X)\,p(X)/q(X)]$. With $q = \mathcal{N}(4, 1)$ the estimate is within 1%.

A poor proposal is worse than useless. Estimating $\mathbb{E}[X^2] = 1$ under $\mathcal{N}(0,1)$ with the too-narrow proposal $\mathcal{N}(0, 0.5^2)$ makes the weights explode in the tails (their variance is infinite here). The estimate came out $0.829 \pm 0.066$, more than two reported standard errors from the truth: the error bar is computed from the same badly weighted samples, so it is wrong too. The **effective sample size** $(\sum w)^2/\sum w^2$, 4,207 of 100,000 here, is the warning sign. Importance weights reappear in off-policy RL and PPO (L3-05).

## 7. Confidence intervals

A **95% confidence interval** comes from a procedure that captures the true value in 95% of repeated samples. It is a property of the procedure, not a 95% probability that this particular interval contains the truth (that is what a Bayesian credible interval claims, given its prior).

For an accuracy $\hat p = k/n$:

- **Wald**: $\hat p \pm 1.96\sqrt{\hat p(1-\hat p)/n}$, straight from the CLT.
- **Wilson**: obtained by inverting a test rather than plugging in $\hat p$; its center is pulled toward 0.5 and it stays inside $[0, 1]$ (formula in `intervals.py`).
- **Bootstrap** [Efron, 1979]: resample the test set with replacement many times, recompute the statistic, and take the 2.5th and 97.5th percentiles. It works for nearly any statistic (F1, AUC, a median latency) without a formula.

The Wald interval misbehaves near 0 and 1, exactly where good classifiers live; [Brown et al., 2001] show its coverage is erratic even for fairly large $n$ and recommend Wilson or Jeffreys intervals for small $n$. Script 6 reproduces this: at true accuracy 0.99 and $n = 50$, the "95%" Wald interval covers the truth only 40% of the time, because about 60% of test sets ($0.99^{50} \approx 0.605$) contain no errors and Wald then reports zero width. Wilson covers 91% there and between 94.8% and 96.8% in the other rows.

## 8. Hypothesis tests and comparing models

A **hypothesis test** asks whether data are surprising under a **null hypothesis** $H_0$, usually "no difference". The **p-value** is the probability, computed assuming $H_0$, of a result at least as extreme as the observed one. The American Statistical Association's statement [Wasserstein and Lazar, 2016] gives six principles; paraphrased: p-values can indicate incompatibility between data and a model; they do not measure the probability that the hypothesis is true or that the data arose by chance alone; decisions should not rest only on crossing a threshold; proper inference needs full reporting; a p-value does not measure the size or importance of an effect; and alone it is not a good measure of evidence.

A **Type I error** rejects a true $H_0$. Under $H_0$, p-values are uniformly distributed, so a threshold $\alpha = 0.05$ produces false positives 5% of the time (script 6: 5.1%, with evenly spaced deciles). A **Type II error** misses a real effect; **power** is the probability of detecting an effect that exists. For comparing models this means:

- **Pair the comparison when models share a test set.** Their errors are correlated, and treating them as independent wastes power. The **McNemar test** looks only at examples where the models disagree; under $H_0$ each disagreement favors either model with probability 0.5. [Dietterich, 1998] found it had acceptable Type I error for algorithms that can be run only once, and recommended a 5x2 cross-validation test when retraining several times is affordable. In script 5, logistic regression beats a decision tree on 171 test examples with 10 disagreements to 1, McNemar $p = 0.012$, and the paired standard error of the difference (0.019) is well below the unpaired one (0.027).
- **Small gains need large test sets.** Detecting 85% vs 87% accuracy with independent test sets has power 0.15 at $n = 500$ per model and about 0.8 only at $n = 5000$ (script 6).
- **Trying many variants manufactures significance.** With 20 variants exactly as good as the baseline, at least one was "significant" at 0.05 in 42% of simulated experiments. (Independent tests would give $1 - 0.95^{20} \approx 64\%$; these share one baseline, so they are correlated.) Say how many things you tried, correct for it, or confirm on fresh data.
- **A test set measures one source of variation.** Script 5's intervals cover the sampling of test examples, not the randomness of the split, initialization or data order, which is why Dietterich's recommended design retrains the models. Evaluation methodology is L1-01 and L4-03.

## Hands-on

Run the scripts from `code/L0-04-probability-and-statistics/`. Each takes a few seconds and uses fixed seeds, so with the same library versions the output should match exactly.

### Script 1: distributions, the CLT, and where it fails

```python
"""Common distributions: check sample moments against theory, then watch the CLT work (and fail)."""
import numpy as np
from scipy import stats

rng = np.random.default_rng(0)
N = 200_000

# (name, sampler, theoretical mean, theoretical variance)
cases = [
    ("Bernoulli(p=0.3)", lambda: rng.binomial(1, 0.3, N), 0.3, 0.3 * 0.7),
    ("Binomial(n=20, p=0.3)", lambda: rng.binomial(20, 0.3, N), 20 * 0.3, 20 * 0.3 * 0.7),
    ("Poisson(lam=4)", lambda: rng.poisson(4, N), 4.0, 4.0),
    ("Uniform(0, 1)", lambda: rng.uniform(0, 1, N), 0.5, 1 / 12),
    ("Normal(mu=2, sigma=3)", lambda: rng.normal(2, 3, N), 2.0, 9.0),
    ("Exponential(rate=0.5)", lambda: rng.exponential(1 / 0.5, N), 2.0, 4.0),
]
print(f"{'distribution':<24} {'mean':>8} {'theory':>8} {'var':>8} {'theory':>8}")
for name, sample, mu, var in cases:
    x = sample()
    print(f"{name:<24} {x.mean():8.4f} {mu:8.4f} {x.var():8.4f} {var:8.4f}")

# Central limit theorem: the mean of n Exponential(1) draws.
# Exponential(1) has skewness 2, so the mean of n draws has skewness 2/sqrt(n).
print("\nMean of n Exponential(1) draws (50,000 repetitions each)")
print(f"{'n':>5} {'mean':>8} {'std':>8} {'1/sqrt(n)':>9} {'skew':>7} {'2/sqrt(n)':>9} {'P(z<-1.96)':>11} {'P(z>1.96)':>10}")
for n in [1, 2, 5, 30, 200]:
    means = rng.exponential(1.0, size=(50_000, n)).mean(axis=1)
    z = (means - 1.0) / (1.0 / np.sqrt(n))  # standardize with the true mean and std
    lo, hi = np.mean(z < -1.96), np.mean(z > 1.96)  # each 0.025 if exactly normal
    print(f"{n:>5} {means.mean():8.4f} {means.std():8.4f} {1 / np.sqrt(n):9.4f} "
          f"{stats.skew(means):7.3f} {2 / np.sqrt(n):9.3f} {lo:11.4f} {hi:10.4f}")

# A distribution with no mean: the running average of Cauchy draws never settles.
print("\nRunning mean of standard Cauchy draws (no finite mean, so no LLN or CLT)")
c = rng.standard_cauchy(1_000_000)
g = rng.normal(size=1_000_000)
for k in [1_000, 10_000, 100_000, 1_000_000]:
    print(f"  first {k:>9,} draws: Cauchy mean {c[:k].mean():9.3f}   Normal mean {g[:k].mean():8.4f}")
```

```text
distribution                 mean   theory      var   theory
Bernoulli(p=0.3)           0.2991   0.3000   0.2097   0.2100
Binomial(n=20, p=0.3)      6.0043   6.0000   4.2181   4.2000
Poisson(lam=4)             4.0070   4.0000   4.0122   4.0000
Uniform(0, 1)              0.5003   0.5000   0.0834   0.0833
Normal(mu=2, sigma=3)      2.0047   2.0000   9.0052   9.0000
Exponential(rate=0.5)      2.0011   2.0000   4.0115   4.0000

Mean of n Exponential(1) draws (50,000 repetitions each)
    n     mean      std 1/sqrt(n)    skew 2/sqrt(n)  P(z<-1.96)  P(z>1.96)
    1   0.9996   0.9949    1.0000   2.042     2.000      0.0000     0.0506
    2   0.9995   0.7091    0.7071   1.424     1.414      0.0000     0.0489
    5   0.9985   0.4450    0.4472   0.873     0.894      0.0004     0.0429
   30   0.9998   0.1828    0.1826   0.369     0.365      0.0145     0.0344
  200   0.9998   0.0705    0.0707   0.133     0.141      0.0211     0.0283

Running mean of standard Cauchy draws (no finite mean, so no LLN or CLT)
  first     1,000 draws: Cauchy mean     1.773   Normal mean   0.0069
  first    10,000 draws: Cauchy mean    -1.176   Normal mean   0.0097
  first   100,000 draws: Cauchy mean    -0.121   Normal mean   0.0052
  first 1,000,000 draws: Cauchy mean     1.332   Normal mean   0.0013
```

At $n = 1$ the lower tail is empty because an exponential variable cannot fall below $1 - 1.96$; the tails even out only slowly. The Cauchy mean is still wandering after a million draws.

### Script 2: Bayes' rule, MLE and MAP

```python
"""Bayes' rule, maximum likelihood, and MAP on small examples you can check by hand."""
import numpy as np
from scipy import stats

rng = np.random.default_rng(0)

# ---- 1. Bayes' rule: a rare condition and a good test ----
prior = 0.01          # P(condition)
sensitivity = 0.99    # P(positive | condition)
false_pos = 0.05      # P(positive | no condition)
evidence = sensitivity * prior + false_pos * (1 - prior)  # P(positive), law of total probability
posterior = sensitivity * prior / evidence
print(f"P(condition | positive) by Bayes' rule: {posterior:.4f}")

# Same answer by brute-force simulation of one million people.
n = 1_000_000
has = rng.random(n) < prior
positive = np.where(has, rng.random(n) < sensitivity, rng.random(n) < false_pos)
print(f"P(condition | positive) by simulation:  {has[positive].mean():.4f} "
      f"({positive.sum():,} positives, {has[positive].sum():,} true)")

# ---- 2. Coin bias: MLE vs MAP vs full posterior with a Beta(a, b) prior ----
a, b = 2.0, 2.0  # prior pseudo-counts: a gentle pull toward 0.5
print(f"\nCoin bias with Beta({a:.0f},{b:.0f}) prior")
print(f"{'heads/flips':>12} {'MLE':>7} {'MAP':>7} {'post. mean':>10} {'95% credible interval':>24}")
for heads, flips in [(3, 3), (7, 10), (70, 100), (700, 1000)]:
    mle = heads / flips
    map_ = (heads + a - 1) / (flips + a + b - 2)  # mode of Beta(heads+a, tails+b)
    post = stats.beta(heads + a, flips - heads + b)
    lo, hi = post.ppf([0.025, 0.975])
    print(f"{heads:>5}/{flips:<6} {mle:7.3f} {map_:7.3f} {post.mean():10.3f}      [{lo:.3f}, {hi:.3f}]")

# ---- 3. The Gaussian MLE of variance is biased low for small samples ----
sigma2, n_small = 4.0, 5
samples = rng.normal(0, np.sqrt(sigma2), size=(200_000, n_small))
mle_var = samples.var(axis=1, ddof=0)      # divide by n: the MLE
unbiased = samples.var(axis=1, ddof=1)     # divide by n-1
print(f"\nTrue variance {sigma2}, n={n_small}: average MLE = {mle_var.mean():.3f} "
      f"(theory {sigma2 * (n_small - 1) / n_small:.3f}), average with n-1 = {unbiased.mean():.3f}")
```

```text
P(condition | positive) by Bayes' rule: 0.1667
P(condition | positive) by simulation:  0.1670 (59,176 positives, 9,880 true)

Coin bias with Beta(2,2) prior
 heads/flips     MLE     MAP post. mean    95% credible interval
    3/3        1.000   0.800      0.714      [0.359, 0.957]
    7/10       0.700   0.667      0.643      [0.386, 0.861]
   70/100      0.700   0.696      0.692      [0.601, 0.777]
  700/1000     0.700   0.700      0.699      [0.670, 0.727]

True variance 4.0, n=5: average MLE = 3.199 (theory 3.200), average with n-1 = 3.999
```

### Script 3: entropy, cross-entropy and KL

```python
"""Entropy, cross-entropy, and KL divergence, and why minimizing cross-entropy is maximum likelihood."""
import numpy as np
from scipy import stats


def entropy(p):
    p = np.asarray(p, float)
    nz = p > 0  # 0 * log 0 is taken to be 0
    return -np.sum(p[nz] * np.log2(p[nz])) + 0.0  # + 0.0 turns -0.0 into 0.0


def cross_entropy(p, q):
    p, q = np.asarray(p, float), np.asarray(q, float)
    nz = p > 0
    return -np.sum(p[nz] * np.log2(q[nz]))  # infinite if q = 0 where p > 0


def kl(p, q):
    return cross_entropy(p, q) - entropy(p)


print("Entropy in bits")
for name, p in [("fair coin", [0.5, 0.5]), ("90/10 coin", [0.9, 0.1]),
                ("99/1 coin", [0.99, 0.01]), ("fair 8-sided die", [1 / 8] * 8),
                ("certain outcome", [1.0, 0.0])]:
    print(f"  {name:<17} H = {entropy(p):.4f}")

p = np.array([0.7, 0.2, 0.1])
q = np.array([0.4, 0.4, 0.2])
print(f"\np = {p}, q = {q}")
print(f"  H(p) = {entropy(p):.4f}  H(p, q) = {cross_entropy(p, q):.4f}  "
      f"KL(p||q) = {kl(p, q):.4f}  KL(q||p) = {kl(q, p):.4f}")

# Cross-entropy loss of a classifier is the average negative log-likelihood of the labels.
rng = np.random.default_rng(0)
labels = rng.integers(0, 3, size=8)
logits = rng.normal(size=(8, 3))
probs = np.exp(logits) / np.exp(logits).sum(axis=1, keepdims=True)  # softmax
nll = -np.mean(np.log(probs[np.arange(8), labels]))
one_hot = np.eye(3)[labels]
ce = np.mean([-np.sum(one_hot[i] * np.log(probs[i])) for i in range(8)])
print(f"\nclassifier: mean NLL of true labels = {nll:.6f}, mean cross-entropy vs one-hot = {ce:.6f} (nats)")

# Cross-entropy H(p_hat, q) = H(p_hat) + KL(p_hat || q) is smallest when q equals the data frequencies.
data = rng.choice(3, size=1000, p=[0.5, 0.3, 0.2])
p_hat = np.bincount(data, minlength=3) / len(data)
print("\n1000 draws from [0.5, 0.3, 0.2]; empirical frequencies p_hat =", np.round(p_hat, 3))
for q_try in [p_hat, [0.5, 0.3, 0.2], [1 / 3] * 3, [0.8, 0.1, 0.1]]:
    print(f"  q = {np.round(q_try, 3)}: cross-entropy with data = {cross_entropy(p_hat, q_try):.4f} bits")

# Forward vs reverse KL: fit one Gaussian q to a two-bump mixture p.
x = np.linspace(-10, 10, 4001)
dx = x[1] - x[0]
p_x = 0.5 * stats.norm.pdf(x, -3, 1) + 0.5 * stats.norm.pdf(x, 3, 1)


def kl_cont(a, b):
    mask = a > 1e-300
    return np.sum(a[mask] * (np.log(a[mask]) - np.log(b[mask] + 1e-300))) * dx  # nats


grid = [(mu, sigma) for mu in np.linspace(-5, 5, 101) for sigma in np.linspace(0.5, 5, 91)]
print("\nOne Gaussian q fitted by grid search to p = 50/50 mixture of N(-3,1) and N(3,1):")
for name, div in [("forward KL(p||q)", lambda q: kl_cont(p_x, q)), ("reverse KL(q||p)", lambda q: kl_cont(q, p_x))]:
    vals = [div(stats.norm.pdf(x, mu, sigma)) for mu, sigma in grid]
    mu, sigma = grid[int(np.argmin(vals))]
    print(f"  minimizing {name}: mu = {mu:+.2f}, sigma = {sigma:.2f}, divergence = {min(vals):.3f} nats")
```

```text
Entropy in bits
  fair coin         H = 1.0000
  90/10 coin        H = 0.4690
  99/1 coin         H = 0.0808
  fair 8-sided die  H = 3.0000
  certain outcome   H = 0.0000

p = [0.7 0.2 0.1], q = [0.4 0.4 0.2]
  H(p) = 1.1568  H(p, q) = 1.4219  KL(p||q) = 0.2651  KL(q||p) = 0.2771

classifier: mean NLL of true labels = 1.100167, mean cross-entropy vs one-hot = 1.100167 (nats)

1000 draws from [0.5, 0.3, 0.2]; empirical frequencies p_hat = [0.48  0.311 0.209]
  q = [0.48  0.311 0.209]: cross-entropy with data = 1.5043 bits
  q = [0.5 0.3 0.2]: cross-entropy with data = 1.5055 bits
  q = [0.333 0.333 0.333]: cross-entropy with data = 1.5850 bits
  q = [0.8 0.1 0.1]: cross-entropy with data = 1.8819 bits

One Gaussian q fitted by grid search to p = 50/50 mixture of N(-3,1) and N(3,1):
  minimizing forward KL(p||q): mu = +0.00, sigma = 3.15, divergence = 0.462 nats
  minimizing reverse KL(q||p): mu = -3.00, sigma = 1.00, divergence = 0.689 nats
```

The forward-KL fit's $\sigma = 3.15$ matches the mixture's standard deviation $\sqrt{1 + 9} \approx 3.16$ to grid resolution (for a Gaussian $q$, forward KL matches the mean and variance of $p$). The reverse-KL minimum of 0.689 nats is close to $\ln 2 \approx 0.693$, the price of ignoring half the mass.

### Script 4: Monte Carlo and importance sampling

```python
"""Monte Carlo estimation: error shrinks like 1/sqrt(N), and rare events need importance sampling."""
import numpy as np
from scipy import stats

rng = np.random.default_rng(0)

# ---- 1. Estimate E[g(X)] for X ~ Uniform(0,1), g(x) = 4 / (1 + x^2). The exact answer is pi. ----
print("Estimating pi = E[4/(1+X^2)], X ~ Uniform(0,1); 200 independent runs per N")
print(f"{'N':>9} {'RMS error':>16} {'error * sqrt(N)':>16} {'reported std err':>17}")
for N in [100, 1_000, 10_000, 100_000]:
    errors, ses = [], []
    for _ in range(200):
        g = 4 / (1 + rng.uniform(size=N) ** 2)
        errors.append(g.mean() - np.pi)
        ses.append(g.std(ddof=1) / np.sqrt(N))  # what one run can report about itself
    rmse = np.sqrt(np.mean(np.square(errors)))
    print(f"{N:>9,} {rmse:16.6f} {rmse * np.sqrt(N):16.4f} {np.mean(ses):17.6f}")

# ---- 2. A rare event: P(Z > 4) for Z ~ N(0,1). ----
exact = stats.norm.sf(4)
N = 100_000
z = rng.normal(size=N)
naive = np.mean(z > 4)
print(f"\nP(Z > 4): exact {exact:.3e}")
print(f"  naive MC, N={N:,}: {naive:.3e} ({int(np.sum(z > 4))} of {N:,} samples landed in the region)")

# Importance sampling: draw from N(4, 1), which puts half its mass in the region,
# and reweight each sample by p(y)/q(y) so the estimate stays unbiased.
y = rng.normal(loc=4, size=N)
w = stats.norm.pdf(y) / stats.norm.pdf(y, loc=4)
vals = (y > 4) * w
est, se = vals.mean(), vals.std(ddof=1) / np.sqrt(N)
print(f"  importance sampling, N={N:,}: {est:.3e} +/- {se:.1e} (relative error {abs(est - exact) / exact:.2%})")

# ---- 3. A bad proposal: importance weights with huge variance. ----
# Target p = N(0, 1), proposal q = N(0, 0.5^2), which is too narrow and misses the tails.
y = rng.normal(scale=0.5, size=N)
w = stats.norm.pdf(y) / stats.norm.pdf(y, scale=0.5)
vals = y ** 2 * w  # estimates E_p[X^2] = 1
ess = w.sum() ** 2 / np.sum(w ** 2)  # effective sample size
print(f"\nE[X^2] under N(0,1) with too-narrow proposal N(0, 0.5^2), N={N:,}:")
print(f"  estimate {vals.mean():.3f} +/- {vals.std(ddof=1) / np.sqrt(N):.3f} (truth 1.000), "
      f"effective sample size {ess:,.0f}")
print(f"  largest single weight {w.max():.1f}; top 10 samples carry {np.sort(vals)[-10:].sum() / vals.sum():.1%} of the sum")
```

```text
Estimating pi = E[4/(1+X^2)], X ~ Uniform(0,1); 200 independent runs per N
        N        RMS error  error * sqrt(N)  reported std err
      100         0.067240           0.6724          0.063976
    1,000         0.020439           0.6463          0.020317
   10,000         0.006266           0.6266          0.006433
  100,000         0.001926           0.6091          0.002034

P(Z > 4): exact 3.167e-05
  naive MC, N=100,000: 1.000e-05 (1 of 100,000 samples landed in the region)
  importance sampling, N=100,000: 3.195e-05 +/- 2.1e-07 (relative error 0.89%)

E[X^2] under N(0,1) with too-narrow proposal N(0, 0.5^2), N=100,000:
  estimate 0.829 +/- 0.066 (truth 1.000), effective sample size 4,207
  largest single weight 952.2; top 10 samples carry 17.3% of the sum
```

### Script 5: comparing two classifiers on one test set

`intervals.py` holds helpers shared by scripts 5 and 6. Script 5 uses the Wisconsin breast cancer dataset bundled with scikit-learn (569 examples), so nothing is downloaded.

```python
"""Shared helpers: intervals for a proportion, and a two-proportion z-test."""
import numpy as np
from scipy import stats

Z = 1.96  # two-sided 95%


def wald(k, n):
    p = k / n
    half = Z * np.sqrt(p * (1 - p) / n)
    return p - half, p + half


def wilson(k, n):
    p = k / n
    center = (p + Z ** 2 / (2 * n)) / (1 + Z ** 2 / n)
    half = Z * np.sqrt(p * (1 - p) / n + Z ** 2 / (4 * n ** 2)) / (1 + Z ** 2 / n)
    return center - half, center + half


def two_proportion_p(ka, kb, n):
    """Two-sided p-value for 'equal accuracy' with independent test sets of size n."""
    pooled = (ka + kb) / (2 * n)
    z = (kb - ka) / n / np.sqrt(pooled * (1 - pooled) * 2 / n)
    return 2 * stats.norm.sf(np.abs(z))
```

```python
"""Two classifiers, one test set: confidence intervals and a paired test for the difference."""
import numpy as np
from scipy import stats
from sklearn.datasets import load_breast_cancer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from intervals import wald, wilson

X, y = load_breast_cancer(return_X_y=True)  # 569 examples, bundled with scikit-learn
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)
model_a = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X_tr, y_tr)
model_b = DecisionTreeClassifier(random_state=0).fit(X_tr, y_tr)
correct_a = model_a.predict(X_te) == y_te
correct_b = model_b.predict(X_te) == y_te
n = len(y_te)


def bootstrap_ci(values, reps=10_000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(values), size=(reps, len(values)))  # resample test items
    return np.percentile(values[idx].mean(axis=1), [2.5, 97.5])


print(f"test set size n = {n}")
for name, c in [("A: logistic regression", correct_a), ("B: decision tree", correct_b)]:
    k = c.sum()
    (w1, w2), (s1, s2), (b1, b2) = wald(k, n), wilson(k, n), bootstrap_ci(c.astype(float))
    print(f"{name:<23} acc={k / n:.4f}  Wald [{w1:.3f}, {w2:.3f}]  Wilson [{s1:.3f}, {s2:.3f}]  "
          f"bootstrap [{b1:.3f}, {b2:.3f}]")

# Paired view: both models saw the same items, so look at where they disagree.
only_a = int(np.sum(correct_a & ~correct_b))
only_b = int(np.sum(~correct_a & correct_b))
print(f"\nboth right {np.sum(correct_a & correct_b)}, both wrong {np.sum(~correct_a & ~correct_b)}, "
      f"only A right {only_a}, only B right {only_b}")
# Exact McNemar test: under 'equal accuracy', each disagreement is a fair coin flip.
print(f"exact McNemar p-value: {stats.binomtest(only_a, only_a + only_b, 0.5).pvalue:.4f}")

diff = correct_a.astype(float) - correct_b.astype(float)
lo, hi = bootstrap_ci(diff)  # resampling items keeps each A/B pair together
print(f"accuracy difference A - B = {diff.mean():+.4f}, paired bootstrap 95% CI [{lo:+.4f}, {hi:+.4f}]")
se_unpaired = np.sqrt(correct_a.var(ddof=1) / n + correct_b.var(ddof=1) / n)
print(f"standard error of the difference: unpaired {se_unpaired:.4f}, "
      f"paired {diff.std(ddof=1) / np.sqrt(n):.4f}")
```

```text
test set size n = 171
A: logistic regression  acc=0.9591  Wald [0.929, 0.989]  Wilson [0.918, 0.980]  bootstrap [0.930, 0.988]
B: decision tree        acc=0.9064  Wald [0.863, 0.950]  Wilson [0.853, 0.942]  bootstrap [0.860, 0.947]

both right 154, both wrong 6, only A right 10, only B right 1
exact McNemar p-value: 0.0117
accuracy difference A - B = +0.0526, paired bootstrap 95% CI [+0.0175, +0.0936]
standard error of the difference: unpaired 0.0270, paired 0.0190
```

The per-model intervals overlap, yet the paired analysis says the difference is unlikely to be test-set noise. This is one split; exercise 9 measures how much the split matters.

### Script 6: what intervals and p-values promise

```python
"""Simulations of what confidence intervals and p-values do and do not promise."""
import numpy as np

from intervals import two_proportion_p, wald, wilson

rng = np.random.default_rng(0)
TRIALS = 20_000

# 1. Coverage: how often does a "95%" interval contain the true accuracy?
print(f"Coverage of nominal 95% intervals ({TRIALS:,} simulated test sets per row)")
print(f"{'true acc':>8} {'n':>6} {'Wald':>7} {'Wilson':>7}")
for p_true in [0.80, 0.95, 0.99]:
    for n in [50, 200, 1000]:
        k = rng.binomial(n, p_true, size=TRIALS)
        cover = [np.mean((lo <= p_true) & (p_true <= hi)) for lo, hi in (wald(k, n), wilson(k, n))]
        print(f"{p_true:8.2f} {n:6d} {cover[0]:7.3f} {cover[1]:7.3f}")

# 2. When the null is true (both models 0.85 accurate), p-values are uniform.
p = two_proportion_p(rng.binomial(500, 0.85, TRIALS), rng.binomial(500, 0.85, TRIALS), 500)
print(f"\nNull true: fraction p < 0.05 = {np.mean(p < 0.05):.3f}, p < 0.01 = {np.mean(p < 0.01):.3f}")
print("  p-value deciles:", np.round(np.quantile(p, np.linspace(0.1, 0.9, 9)), 2))

# 3. Twenty variants, none truly better than the baseline: report the best one.
hits = [np.any(two_proportion_p(rng.binomial(500, 0.85), rng.binomial(500, 0.85, 20), 500) < 0.05)
        for _ in range(2_000)]
print(f"\n20 equally good variants: P(at least one p < 0.05) = {np.mean(hits):.3f}")

# 4. Power: how large a test set is needed to detect 0.85 -> 0.87?
print("\nPower to detect 0.85 -> 0.87 (independent test sets, alpha = 0.05)")
for n in [500, 2000, 5000]:
    p = two_proportion_p(rng.binomial(n, 0.85, 5000), rng.binomial(n, 0.87, 5000), n)
    print(f"  n = {n:>5} per model: power = {np.mean(p < 0.05):.3f}")
```

```text
Coverage of nominal 95% intervals (20,000 simulated test sets per row)
true acc      n    Wald  Wilson
    0.80     50   0.938   0.951
    0.80    200   0.942   0.958
    0.80   1000   0.946   0.948
    0.95     50   0.920   0.963
    0.95    200   0.928   0.968
    0.95   1000   0.942   0.951
    0.99     50   0.399   0.906
    0.99    200   0.862   0.948
    0.99   1000   0.924   0.961

Null true: fraction p < 0.05 = 0.051, p < 0.01 = 0.011
  p-value deciles: [0.1  0.19 0.29 0.4  0.49 0.6  0.72 0.79 0.93]

20 equally good variants: P(at least one p < 0.05) = 0.422

Power to detect 0.85 -> 0.87 (independent test sets, alpha = 0.05)
  n =   500 per model: power = 0.147
  n =  2000 per model: power = 0.440
  n =  5000 per model: power = 0.813
```

## Common mistakes

- **Ignoring the base rate.** A low false-positive rate can still mean most positives are wrong when the condition is rare.
- **Reading a confidence interval as a probability about this interval, or a p-value as the probability that the null is true.** Neither is correct.
- **Using the Wald interval for accuracies near 1.** After zero errors it reports zero width. Use Wilson or a bootstrap. With zero errors in $n$ trials, the "rule of three" gives a rough 95% upper bound of $3/n$ on the error rate, because $(1 - 3/n)^n \approx e^{-3} \approx 0.05$.
- **Unpaired comparisons of models scored on the same examples**, and **reporting the best of many variants without saying so**. The first misses real differences; the second manufactures fake ones.
- **Trusting $\sigma/\sqrt N$ for heavy-tailed quantities.** Cauchy-like data and heavy-tailed importance weights produce error bars that are themselves wrong. Look at the largest contributions.
- **Confusing $\mathrm{KL}(p\|q)$ with $\mathrm{KL}(q\|p)$**, or calling KL a distance.
- **Taking the log of a probability that rounded to 0.** Compute cross-entropy from logits with a stable log-softmax (L0-01).

## Exercises

1. [warm-up] A model has true accuracy 0.8. What is the standard error of its accuracy on 400 test examples, and how many examples would halve it?
2. [warm-up] Compute the entropy in bits of $(0.5, 0.25, 0.25)$ and give a binary code that achieves it.
3. [warm-up] Redo the Bayes example with prevalence 10%.
4. [core] Derive the MLE of a Poisson rate $\lambda$ from $n$ observations, and check it on `rng.poisson(4, 1000)`.
5. [core] Show that minimizing $\frac{1}{n}\sum_i -\log q_\theta(y_i)$ over $\theta$ is the same as minimizing $\mathrm{KL}(\hat p \,\|\, q_\theta)$, where $\hat p$ is the empirical label distribution.
6. [core] In `monte_carlo.py`, try proposals $\mathcal{N}(2, 1)$ and $\mathcal{N}(6, 1)$ for $P(Z > 4)$. Which is better, and why do both work when the too-narrow proposal fails?
7. [core] Add the Jeffreys interval, the 2.5% and 97.5% quantiles of Beta($k + 0.5$, $n - k + 0.5$), to the coverage table in `simulate_inference.py`, and compare it with Wilson.
8. [stretch] Apply a Bonferroni correction (threshold $0.05/20$) to the 20-variant experiment. What happens to the false-positive rate, and to the power for one variant that is truly 2 points better at $n = 500$?
9. [stretch] Repeat `compare_models.py` for 50 values of the split's `random_state`. Compare the spread of the accuracy difference across splits with the width of one split's paired bootstrap interval. Which uncertainty does a single test set capture?

### Hints and answers

1. $\sqrt{0.8 \times 0.2/400} = 0.02$; four times the data (1600 examples) halves it.
2. $H = 0.5 \times 1 + 2 \times 0.25 \times 2 = 1.5$ bits; the code `0`, `10`, `11` has average length 1.5.
3. $0.099/(0.099 + 0.045) \approx 0.69$, up from 0.17.
4. $\ell(\lambda) = \sum_i x_i\log\lambda - n\lambda + \text{const}$, so $\hat\lambda = \bar x$.
5. The average equals $H(\hat p, q_\theta) = H(\hat p) + \mathrm{KL}(\hat p\|q_\theta)$, and $H(\hat p)$ does not depend on $\theta$.
6. $\mathcal{N}(4,1)$ is near ideal; the others have larger but finite variance. For $q = \mathcal{N}(m, 1)$ the weight $p/q = e^{m^2/2 - mx}$ is bounded on $x > 4$ for $m > 0$. The failing proposal is narrower than the target, so $p/q$ grows without bound in the tails.
7. When I ran it with the same seed, Jeffreys was within about 0.01 of 95% in most rows, high (0.985) at accuracy 0.99 with small $n$, and 0.886 at accuracy 0.95 with $n = 50$, where Wilson gave 0.963. Coverage of discrete intervals oscillates; [Brown et al., 2001] base their recommendation on overall behavior, not on every cell.
8. Bonferroni bounds the family-wise false-positive rate by 5% (lower here because the tests are correlated), but the per-test threshold of 0.0025 cuts power for a real 2-point gain sharply.
9. Expect split-to-split variation comparable to or larger than the single-split interval. One test set captures test-sampling noise only, which motivates cross-validation and repeated splits (L1-01) and designs like Dietterich's 5x2 cv test.

## Check your understanding

**1. Why does the standard error of an accuracy shrink like $1/\sqrt n$, not $1/n$?** Accuracy is a mean of $n$ independent 0/1 outcomes. Their variances add, so the sum has variance $n\sigma^2$ and the mean $\sigma^2/n$; the square root is $\sigma/\sqrt n$. Halving the error costs four times the data.

**2. Why is training with cross-entropy loss maximum likelihood?** With a one-hot target, cross-entropy equals $-\log q(y)$, the negative log-probability of the true label. Averaged over examples, it is the dataset's negative log-likelihood divided by $n$, so minimizing one minimizes the other.

**3. When do MLE and MAP differ, and what does a prior correspond to in ordinary training?** They differ when data are scarce relative to the prior's strength, and converge as data grow. A Gaussian prior on weights corresponds to L2 regularization (weight decay) and a Laplace prior to L1, so regularized training is MAP.

**4. Model B beats model A by 1.5 points on a shared test set of 1000 examples, $p = 0.03$. What would you ask?** How many variants or seeds were tried; whether the test was paired; whether the gain survives other seeds and splits (the p-value covers only test-set sampling); whether the test set was used for tuning; and whether 1.5 points matters. A p-value measures compatibility with "no difference", not the size of the gain.

**5. How can importance sampling report a small error bar and be far from the truth?** The error bar is estimated from the same weighted samples. With heavy-tailed weights, a few rare samples dominate the true answer; until they appear, both the estimate and its error bar are wrong. A low effective sample size and a few dominant weights are the warning signs.

## Further reading

- **Joseph K. Blitzstein, Jessica Hwang, "Introduction to Probability"** (2nd edition, 2019, CRC Press; free online version from the authors, linked from Harvard's Statistics 110 site). <https://probabilitybook.net/>
  *Read for:* an example-driven first course in probability, conditioning and the common distributions. *Level:* intro.
- **Daniel Kunin, Jingru Guo, Tyler Dae Devlin, Daniel Xiang, "Seeing Theory"** (Brown University, interactive website, archived). <https://seeing-theory.brown.edu/>
  *Read for:* interactive visualizations of distributions and of frequentist and Bayesian inference. *Level:* intro.
- **Marc Peter Deisenroth, A. Aldo Faisal, Cheng Soon Ong, "Mathematics for Machine Learning"** (2020, Cambridge University Press; free PDF). <https://mml-book.github.io/>
  *Read for:* chapter 6, Probability and Distributions, including Bayes' theorem, the Gaussian and conjugacy. *Level:* intro to intermediate.
- **Ian Goodfellow, Yoshua Bengio, Aaron Courville, "Deep Learning"** (2016, MIT Press; free HTML). <https://www.deeplearningbook.org/>
  *Read for:* chapter 3, Probability and Information Theory, a compact summary aimed at deep learning. *Level:* intro to intermediate.
- **Kevin P. Murphy, "Probabilistic Machine Learning: An Introduction"** (2022, MIT Press; free draft PDF under CC BY-NC-ND). <https://probml.github.io/pml-book/book1.html>
  *Read for:* the probabilistic view of ML end to end: MLE, MAP, Bayesian inference, information theory. *Level:* intermediate.
- **David J. C. MacKay, "Information Theory, Inference, and Learning Algorithms"** (2003, Cambridge University Press; free to view on screen from the author's site). <https://www.inference.org.uk/itila/book.html>
  *Read for:* entropy, KL and Monte Carlo methods in unusual depth. *Level:* intermediate to advanced.
- **Claude E. Shannon, "A Mathematical Theory of Communication"** (1948, Bell System Technical Journal 27, 379 to 423 and 623 to 656; publisher copy may be paywalled). <https://doi.org/10.1002/j.1538-7305.1948.tb01338.x>
  *Read for:* the founding paper of information theory, where entropy is defined. *Level:* advanced.
- **Bradley Efron, "Bootstrap Methods: Another Look at the Jackknife"** (1979, The Annals of Statistics 7(1), 1 to 26; open access). <https://projecteuclid.org/euclid.aos/1176344552>
  *Read for:* the original bootstrap paper, built around worked examples. *Level:* advanced.
- **Lawrence D. Brown, T. Tony Cai, Anirban DasGupta, "Interval Estimation for a Binomial Proportion"** (2001, Statistical Science 16(2), 101 to 133). <https://doi.org/10.1214/ss/1009213286>
  *Read for:* why the Wald interval is unreliable and which intervals to use instead. *Level:* intermediate.
- **Ronald L. Wasserstein, Nicole A. Lazar, "The ASA's Statement on p-Values: Context, Process, and Purpose"** (2016, The American Statistician 70(2), 129 to 133). <https://doi.org/10.1080/00031305.2016.1154108>
  *Read for:* six short principles on what p-values do and do not mean. *Level:* intro.
- **Thomas G. Dietterich, "Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms"** (1998, Neural Computation 10, 1895 to 1923). <https://doi.org/10.1162/089976698300017197>
  *Read for:* which tests to trust when comparing learning algorithms, including McNemar's test and the 5x2 cv test. *Level:* intermediate.
- **SciPy developers, "Statistical functions (scipy.stats)"** (SciPy documentation, v1.18.0 as of October 2026). <https://docs.scipy.org/doc/scipy/reference/stats.html>
  *Read for:* the distributions, hypothesis tests and `bootstrap` function you will use in practice. *Level:* intro.
