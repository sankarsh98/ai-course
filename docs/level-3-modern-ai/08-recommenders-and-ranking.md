# Recommenders and Ranking

> **Level 3 · Module 8** · Prerequisites: L0-02, L1-01, L1-02, L2-02, L2-06 · Time: 7 to 9 hours · Code: `code/L3-08-recommenders-and-ranking/`

## What you will be able to do

- Frame recommendation as rating prediction or top-N ranking from explicit or implicit feedback, and set up the baselines that a new method has to beat.
- Derive and implement biased matrix factorization and implicit ALS, and explain why a plain SVD is the wrong tool for a matrix with missing entries.
- Build a two-tower retrieval model with in-batch negatives, explain the sampling-bias correction, and use side features for cold start.
- Lay out a multi-stage system (candidate generation, ranking, re-ranking) and say what each stage optimizes.
- Evaluate honestly: use full-catalogue metrics, recognize exposure bias and sampled-metric pitfalls, and explain what online experiments and off-policy estimators add.
- Explain feedback loops and popularity bias, and summarize the evidence on recommenders' societal effects, including where studies disagree.

## 1. The problem

There are $m$ users and $n$ items, and a matrix $R \in \mathbb{R}^{m \times n}$ in which almost every entry is missing. **Explicit feedback** is a stated preference such as a 1 to 5 star rating. **Implicit feedback** is behavior: clicks, plays, purchases, watch time. Implicit data is far more plentiful but has no negatives: an unobserved cell may mean "disliked" or, much more often, "never seen".

Two tasks follow. **Rating prediction** asks for $\hat r_{ui}$ for a given pair and is scored by RMSE; the Netflix Prize made it famous, and Koren et al. note that the competition showed matrix factorization beating classic nearest-neighbor methods [Koren et al., 2009]. **Top-N recommendation** asks for a short ranked list per user and is what products actually show. It is scored with ranking metrics (section 6).

The data are **missing not at random** (MNAR): whether a cell is observed depends on what the system exposed and on what the user chose to rate. In script 01's synthetic world, users rate what they see and like, so the mean observed rating is 4.16 while the mean true rating over all pairs is 3.00. Every method in this module learns from such data, and every offline evaluation is computed on it.

Throughout, the code uses a synthetic world (`recdata.py`) with 600 users, 400 items and a known true preference for every pair. Real logs never give you that. Here it lets us check what each metric gets right.

**Baselines first.** The global mean, user and item biases, and global popularity are cheap and often strong. Ferrari Dacrema et al. tried to reproduce 18 neural recommendation algorithms from top conferences; only 7 could be reproduced with reasonable effort, and 6 of those 7 were often beaten by simple nearest-neighbor or graph methods [Ferrari Dacrema et al., 2019]. Tune your baselines as carefully as your model.

## 2. Collaborative filtering by neighborhoods

**Collaborative filtering** predicts from the behavior of similar users or items, without content features. The item-based version represents item $i$ by its column $x_{\cdot i}$ of the interaction matrix $X$ and computes a shrunk cosine similarity

$$S_{ij} = \frac{x_{\cdot i}^\top x_{\cdot j}}{\lVert x_{\cdot i}\rVert\, \lVert x_{\cdot j}\rVert + \beta},$$

where the shrinkage $\beta$ damps similarities computed from few co-occurrences. A user's score for item $j$ is $\sum_{i \in \text{items}(u)} S_{ij}$, which in matrix form is $XS$. This is a few lines of code (`item_knn_scores` in `recdata.py`), easy to explain ("because you watched..."), and on our held-out clicks it has the best Recall@10 (0.297) of the untuned models in script 02. Its weaknesses are cost ($n^2$ similarities), noise for rare items, and no notion of a user beyond their history.

## 3. Matrix factorization

**The model.** Give every user a vector $p_u \in \mathbb{R}^k$ and every item a vector $q_i \in \mathbb{R}^k$, plus biases, and predict

$$\hat r_{ui} = \mu + b_u + b_i + p_u^\top q_i ,$$

where $\mu$ is the global mean, $b_u$ captures that some users rate high, and $b_i$ that some items are better liked. The dot product scores how well a user's latent tastes match an item's latent attributes. Fit on the set $\mathcal{K}$ of observed pairs:

$$\min_{b, P, Q} \sum_{(u,i) \in \mathcal{K}} \big(r_{ui} - \hat r_{ui}\big)^2 + \lambda\Big(b_u^2 + b_i^2 + \lVert p_u\rVert^2 + \lVert q_i\rVert^2\Big).$$

This is the biased MF of [Koren et al., 2009], which also describes extensions for implicit signals, time and confidence.

**Why not just take the SVD?** L0-02 showed that the truncated SVD is the best rank-$k$ approximation of a *full* matrix (Eckart and Young). Here 94% of the entries are missing. To use the SVD you must fill them in, and filling with the mean asserts that every unseen item would get an average rating. Script 01 does exactly that: rank-8 SVD of the mean-imputed matrix gets test RMSE 0.880, barely better than predicting the global mean (0.898), while MF fitted only on observed entries gets 0.744. The objective above counts only observed cells, which makes it non-convex with no closed form. Two algorithms dominate.

**Stochastic gradient descent.** For one observed rating, let $e_{ui} = r_{ui} - \hat r_{ui}$. The gradient of the per-example loss gives the updates, with learning rate $\eta$:

$$b_u \leftarrow b_u + \eta(e_{ui} - \lambda b_u), \quad p_u \leftarrow p_u + \eta(e_{ui} q_i - \lambda p_u), \quad q_i \leftarrow q_i + \eta(e_{ui} p_u - \lambda q_i),$$

and similarly for $b_i$ (the factor 2 is absorbed into $\eta$ and $\lambda$). Update $p_u$ and $q_i$ simultaneously, using the old values on the right, as `fit_mf` does.

**Alternating least squares (ALS).** Fix all item vectors. Then each user's vector solves an ordinary ridge regression (L1-02) on the items that user rated:

$$p_u = \big(Q_u^\top Q_u + \lambda I\big)^{-1} Q_u^\top r_u ,$$

where $Q_u$ stacks the vectors of items $u$ rated and $r_u$ holds the ratings (minus biases). Then fix users and solve for items the same way, and alternate. Each step cannot increase the loss, and the per-user solves are independent, so ALS parallelizes well. We use ALS for implicit data in section 4.

**What script 01 shows.** The true world has 8 latent factors. With $k = 8$, MF reaches test RMSE 0.744 against a noise floor of 0.498 (the RMSE of the true expected rating, which cannot predict the rating noise). With $k = 32$ and no regularization it memorizes: training RMSE 0.315, test 0.898, no better than the global mean. Regularization $\lambda = 0.2$ brings $k = 32$ back to 0.844. On this data $\lambda$ did not help at $k = 8$ (0.744 without, 0.751 with $\lambda = 0.05$), which is why $k$ and $\lambda$ are tuned together on validation data.

**Dot product or a neural network?** Neural collaborative filtering replaced the dot product with a learned MLP. Rendle et al. revisited those experiments and found that a properly tuned dot product substantially outperformed the MLP, that an MLP finds it hard to learn a dot product, and that dot products allow fast retrieval while MLP scores do not [Rendle et al., 2020]. The dot product is a strong default.

## 4. Implicit feedback

With clicks there are no ratings to regress on, and the unobserved cells matter: they are the only source of "negatives". Two classic formulations:

**Weighted matrix factorization (iALS).** Hu, Koren and Volinsky turn the click count $x_{ui}$ into a binary preference $p_{ui} = [x_{ui} > 0]$ and a confidence $c_{ui} = 1 + \alpha x_{ui}$, then fit *every* cell, observed or not:

$$\min_{X, Y} \sum_{u,i} c_{ui}\big(p_{ui} - x_u^\top y_i\big)^2 + \lambda\Big(\sum_u \lVert x_u\rVert^2 + \sum_i \lVert y_i\rVert^2\Big)$$

[Hu et al., 2008]. Unobserved cells are treated as weak zeros with confidence 1. ALS still applies: $x_u = (Y^\top C^u Y + \lambda I)^{-1} Y^\top C^u p(u)$, with $C^u$ the diagonal matrix of user $u$'s confidences. The paper's key trick is $Y^\top C^u Y = Y^\top Y + Y^\top (C^u - I) Y$: the first term is shared by all users, and the second involves only the items the user interacted with, so each update costs time proportional to the user's interactions rather than the catalogue. Our `ials` uses dense solves, which is fine for 400 items.

**Bayesian personalized ranking (BPR).** Rendle et al. instead optimize the order: for a user $u$, an item $i$ they interacted with should score above an item $j$ they did not. With scores $\hat x_{ui}$, maximize

$$\sum_{(u,i,j)} \ln \sigma\big(\hat x_{ui} - \hat x_{uj}\big) - \lambda \lVert \Theta \rVert^2$$

over sampled triples by SGD, where $\sigma$ is the logistic function and $\Theta$ the parameters [Rendle et al., 2009]. This is a smooth stand-in for the AUC of each user's ranking, and it applies to MF or to neighborhood models.

Both make the same assumption: what a user did not click, they like less than what they clicked. Under heavy exposure bias that assumption is shaky, because most unclicked items were never shown.

## 5. Two-tower models, candidate generation and ranking

**Multiple stages.** A large service cannot score millions of items with an expensive model for every request. The standard design is a funnel. **Candidate generation** (retrieval) cuts millions of items to hundreds with cheap models. **Ranking** scores those hundreds with a rich model using many features. **Re-ranking** applies diversity, freshness, policy and business rules. Covington et al. describe this two-stage structure for YouTube [Covington et al., 2016], and Google's recommendation course splits the funnel into the three stages above [Google, Recommendation Systems course].

**Two-tower retrieval.** A user tower $f$ maps user features (history, context, profile) to a vector, an item tower $g$ maps item features to a vector, and the score is $s(u, i) = \langle f(x_u), g(z_i)\rangle$. Item vectors are computed offline and indexed for approximate nearest-neighbor search (L2-06, L3-06), so retrieval costs one user-tower pass and one index query. MF is the special case where the only feature is the ID.

**Training with in-batch negatives.** Treat retrieval as classification over items with a softmax, $P(i \mid u) = e^{s(u,i)} / \sum_j e^{s(u,j)}$. The sum over all items is too expensive, so a common trick uses the other items in the same minibatch as negatives. Yi et al. point out the bias: an item appears in a batch with probability roughly proportional to its popularity, so popular items are overly penalized as negatives. Their fix, borrowed from sampled softmax, subtracts the log of each item's sampling probability $p_j$ from its logit during training, $s^c(u, j) = s(u, j) - \log p_j$, with $p_j$ estimated from the data stream; they also normalize embeddings and divide by a temperature [Yi et al., 2019].

Script 03 trains a two-tower model with and without this correction. The outcome is instructive:

| model | Recall@10 (held-out clicks) | true prec@10 | share from 20 most-exposed items |
|---|---|---|---|
| iALS | 0.295 | 0.554 | 28% |
| two-tower, no correction | 0.096 | 0.716 | 1% |
| two-tower, logQ correction | 0.361 | 0.679 | 31% |

"True prec@10" is the mean *true* click probability of the 10 recommended items, available only because the world is synthetic. The corrected model estimates the softmax over the *logged* clicks, which in our world are shaped mostly by popularity-driven exposure, so it recommends popular items and wins on held-out clicks. The uncorrected model over-penalizes popular items, which here happens to cancel much of the exposure bias, so it scores worse on held-out clicks and better on true preference. Yi et al. report improvements from the correction in offline experiments and live A/B tests on YouTube [Yi et al., 2019], where popularity carries real signal as well as exposure. The lesson is not "skip the correction". It is that an estimator can be correct for the logged distribution and still encode the exposure process that produced the log.

**Cold start.** A new item has no interactions, so MF and iALS have no vector for it (its learned vector is near zero). A two-tower model can score it from features. In script 03, recommending among 80 items held out of training entirely, iALS reaches true prec@10 of 0.164 (random is 0.203), while the two-tower models reach 0.553 and 0.499 (the oracle is 0.766). New users are handled the same way through user features. The other standard tools are popularity fallbacks and **exploration**: Li et al. framed news recommendation as a contextual bandit, and their LinUCB algorithm gave a 12.5% click lift over a context-free bandit on Yahoo! front-page traffic ([Li et al., WWW 2010](https://arxiv.org/abs/1003.0146)).

**Ranking and re-ranking.** The ranker can afford cross features, sequence models of recent behavior and several objectives at once (click, watch time, purchase, satisfaction surveys), and its outputs are often combined into one score with hand-set weights, which is a product decision as much as a modeling one. Re-ranking then enforces constraints on the final list, for example raising the share of long-tail items with a personalized diversification step as in Abdollahpouri et al. ([2019](https://arxiv.org/abs/1901.07555)).

**Generative recommenders, as of October 2026.** A recent line of industrial work treats recommendation as sequence modeling. Meta's HSTU work reports sequential transducers with up to 1.5 trillion parameters and a 12.4% improvement in online A/B metrics ([Zhai et al., ICML 2024](https://arxiv.org/abs/2402.17152)). Kuaishou's OneRec replaces the retrieval and ranking cascade with a single encoder-decoder that generates item tokens, and reports serving 25% of requests in its apps with operating cost 10.6% of the traditional pipeline ([Zhou et al., 2025](https://arxiv.org/abs/2506.13695)). These are reports by the companies that built the systems, on their own metrics; I did not find independent evidence of how widely the approach has displaced multi-stage pipelines, so treat it as a direction rather than the standard.

## 6. Evaluation

**Ranking metrics.** For a user with a set $T_u$ of held-out relevant items and a top-$K$ list $L_u$,

$$\text{Recall@}K = \frac{|L_u \cap T_u|}{\min(K, |T_u|)}, \qquad \text{NDCG@}K = \frac{\sum_{r=1}^{K} [L_u(r) \in T_u] / \log_2(r+1)}{\sum_{r=1}^{\min(K, |T_u|)} 1/\log_2(r+1)} ,$$

averaged over users, where $L_u(r)$ is the item at rank $r$. NDCG rewards putting hits near the top. Exclude items the user interacted with in training from the list, and rank against the *whole* catalogue.

**Sampled metrics are inconsistent.** Ranking each held-out item against, say, 100 random negatives is much cheaper than ranking the catalogue and was common in papers. Krichene and Rendle showed that such sampled metrics are inconsistent with their exact versions (only AUC agrees) and can reorder methods; they advise against sampling or, if unavoidable, applying a correction [Krichene and Rendle, 2020]. Script 02 shows a flip: iALS with $k = 16$ beats popularity on full NDCG@10 (0.185 against 0.162), but popularity wins on sampled hit rate (0.503 against 0.470).

**Split like deployment.** Random holdouts let a model train on a user's future to predict their past. When timestamps exist, split by time (L1-01 covers leakage).

**Exposure bias in offline metrics.** Held-out clicks were generated by whatever the old system showed. Script 02 makes the consequence stark:

| model | Recall@10 | NDCG@10 | true prec@10 |
|---|---|---|---|
| random | 0.029 | 0.018 | 0.214 |
| popularity | 0.240 | 0.162 | 0.236 |
| item-kNN | 0.297 | 0.228 | 0.340 |
| iALS $k = 8$ | 0.287 | 0.208 | 0.579 |
| oracle (true preferences) | 0.185 | 0.115 | 0.952 |

The oracle, which recommends exactly what users like most, has the *worst* Recall@10 of the non-random models, and popularity, barely better than random for real users, looks respectable. The held-out clicks measure who was shown what as much as who likes what. Tuning inherits the bias: across 18 iALS configurations, the one with the best held-out Recall@10 (0.320) has true precision 0.450, while the best by true precision (0.588) has Recall@10 0.236; the two criteria correlate at only +0.26, and the offline pick draws 66% of its recommendations from the 20 most-exposed items against 28% for the truth pick. In explicit data the same bias appears as MNAR: in script 01 every fitted model predicts about 3.8 to 4.2 for random pairs whose true mean is 3.00, and RMSE on random pairs (1.210 for the best MF) is far above RMSE on the observed test set (0.751). Schnabel et al. treat recommendations as treatments and use propensity weighting to make learning and evaluation unbiased under such selection [Schnabel et al., 2016].

**Online experiments.** An A/B test that randomizes users between systems measures the effect of the system on what you choose to track, which is why it is the final arbiter in industry. Its own problems are choosing a metric that tracks long-term value (clicks are easy to raise in ways users regret), novelty effects, and interference when users or items are shared between arms. Experiment design is L4-01 and L4-03 territory.

**Off-policy evaluation.** Between offline replay and a live test sits counterfactual estimation from logs. If the logging policy $\pi_0$ showed item $a_t$ to user $u_t$ with known probability $\pi_0(a_t \mid u_t)$ and observed reward $r_t$, the inverse propensity scoring (IPS) estimate of a new deterministic policy $\pi$'s value is

$$\hat V_{\text{IPS}}(\pi) = \frac{1}{N} \sum_{t=1}^{N} \frac{[\pi(u_t) = a_t]}{\pi_0(a_t \mid u_t)}\, r_t .$$

It is unbiased if propensities are logged correctly and $\pi_0$ gives positive probability to every action $\pi$ might take. The self-normalized version (SNIPS) divides by the sum of the weights instead of $N$, trading a little bias for much lower variance. Script 04 part 2 logs 60,000 events from a popularity-driven policy with 10% uniform exploration. The standard offline "hit rate" ranks popularity (0.457) above iALS (0.423) above the oracle (0.178), the reverse of the truth. IPS ranks them correctly. So does the simple "replay" average of clicks over matched events, but it overweights users whose recommended item the logger shows often; the bias is small here (0.812 against a true 0.829 for iALS) and grows the more the logging probability of each user's recommended item varies across users. The oracle's IPS estimate rests on only 545 matching events, and its 95% interval $[0.877, 1.340]$ exceeds 1, an impossible click rate; SNIPS gives 0.997 against a true 0.989. Off-policy estimates are only as good as the logging policy's exploration, which is one more reason production systems randomize a little.

## 7. Feedback loops and popularity bias

A deployed recommender trains on clicks it caused: model, exposure, clicks, training data, model. Chaney et al. simulated this loop and found that training on such confounded data homogenizes user behavior without increasing utility [Chaney et al., 2018]. Jiang et al. analyze "degenerate feedback loops" theoretically, separating echo-chamber and filter-bubble effects, and propose ways to slow them, such as exploration ([Jiang et al., AIES 2019](https://arxiv.org/abs/1902.10730)).

Script 04 part 1 runs the loop for 8 rounds. Each round every user is shown 10 items they have not clicked, clicks according to true preference, and the system retrains on all clicks so far:

| policy | clicks per user (8 rounds) | CTR round 8 | items shown in round 8 | exposure Gini | from top-20 items | list overlap between users |
|---|---|---|---|---|---|---|
| random | 15.60 | 0.182 | 100% | 0.06 | 4% | 0.014 |
| popularity | 8.02 | 0.057 | 11% | 0.95 | 65% | 0.241 |
| iALS | 27.65 | 0.223 | 44% | 0.75 | 29% | 0.051 |
| iALS + explore (2 of 10 slots random) | 27.79 | 0.260 | 98% | 0.58 | 22% | 0.035 |
| oracle | 59.41 | 0.562 | 100% | 0.28 | 4% | 0.016 |

Popularity is the degenerate case: it keeps showing the same popular items to users who did not click them, and ends with half as many clicks as random recommendation and the most homogeneous lists. iALS is far better but concentrates exposure (Gini 0.75) and shows only 44% of the catalogue in the last round. Giving up 2 of 10 slots to random items costs clicks in round 1 (CTR 0.534 against 0.602, in the full output) and pays them back by round 8, matching total clicks while keeping 98% of the catalogue in circulation. CTR falls for every policy over the rounds partly because the toy never re-shows clicked items, so each user's best items get used up; compare policies with each other, not across rounds.

**Popularity bias** is the general tendency of recommenders to over-recommend popular items relative to users' tastes. It has three sources visible in this module: exposure-shaped training data, metrics that reward predicting exposure (section 6), and losses whose negatives follow popularity (section 5). Remedies are exploration, propensity-weighted learning and evaluation, calibrated or diversified re-ranking, and measuring exposure distributions (coverage, Gini) next to accuracy.

## 8. Societal effects: what the evidence says

The worries are filter bubbles, polarization, radicalization, and winner-take-all exposure for creators. The evidence is mixed, and serious researchers disagree.

- **Audits of pathways.** Ribeiro et al. analyzed 330,925 videos on 349 YouTube channels and over 72 million comments, and reported that users consistently migrated from milder to more extreme content; from over 2 million recommendations collected in 2019, Alt-lite content was easily reachable from Intellectual Dark Web channels, while Alt-right videos were reachable only through channel recommendations ([Ribeiro et al., FAT* 2020](https://arxiv.org/abs/1908.08313)).
- **Panel data on consumption.** Hosseinmardi et al. studied a representative panel of over 300,000 Americans from 2016 to 2019 and found no evidence that YouTube's recommendations systematically drive far-right engagement; consumption largely reflected preferences that extend across the web ([Hosseinmardi et al., PNAS 2021](https://doi.org/10.1073/pnas.2101967118)).
- **A field experiment.** Guess et al. moved consenting Facebook and Instagram users to reverse-chronological feeds for three months during the 2020 US election. Users spent less time on the platforms, saw more political and untrustworthy content and, on Facebook, less uncivil content, but polarization, political knowledge and other key attitudes did not change significantly [Guess et al., 2023]. Bagchi et al. argued in a Science eLetter that emergency changes Facebook made to its algorithm around the election lowered misinformation for the control group during the study, so the comparison may mislead about the algorithm in normal times ([Bagchi et al., 2024](https://arxiv.org/abs/2409.18393)).

A fair reading: effects on *exposure* are large and easy to measure (they are what recommenders do); effects on *attitudes* in short experiments have been small; pathways to extreme content exist, but how much recommendation, as opposed to user demand, drives their use is contested; and long-run, population-level effects are hard to identify with any design. Regulation has moved anyway. Under the EU Digital Services Act (Regulation (EU) 2022/2065), Article 38 requires very large online platforms and search engines that use recommender systems to offer at least one option per system that is not based on profiling ([DSA, Article 38](https://eur-lex.europa.eu/eli/reg/2022/2065/oj); I checked the article's wording on a copy of the Official Journal text because the EUR-Lex page did not load for me); how well platforms implement this is itself disputed. L4-05 covers fairness and governance more broadly.

## Hands-on

Run from `code/L3-08-recommenders-and-ranking/` with the course Python (NumPy 2.5.4, PyTorch 2.15 CPU). All data are synthetic and generated by `recdata.py`; no downloads. Timings on this machine: script 01 about 20 to 25 seconds (pure-Python SGD), the others under 7 seconds. Results are seeded; small differences in trailing digits are possible on other platforms or library versions, especially for the PyTorch model.

### recdata.py: the synthetic world and shared helpers

True affinity is the dot product of 8-dimensional user and item factors plus user and item biases. Exposure follows a power law over items that is only loosely tied to quality (correlation of log exposure with item quality about 0.4), so "popular" partly means "shown a lot". Click probability is a logistic function of affinity. Side features are noisy linear views of the true factors, which is generous to the two-tower model; real features are less informative.

```python
"""A synthetic recommendation world with known ground truth, plus shared helpers.

Because we generate the data, we know every user's true preference for every item,
including the ones they never saw. Real logs never give you that; here it lets us
check what offline metrics get right and wrong.
"""
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def make_world(n_users=600, n_items=400, k=8, seed=0):
    rng = np.random.default_rng(seed)
    P = rng.normal(0, 1, (n_users, k)) / np.sqrt(k) * 1.6   # true user factors
    Q = rng.normal(0, 1, (n_items, k))                      # true item factors
    b_user = rng.normal(0, 0.3, n_users)
    b_item = rng.normal(0, 0.5, n_items)                    # item "quality"
    affinity = P @ Q.T + b_user[:, None] + b_item[None, :]
    # Exposure propensity: a power law over items, only loosely tied to quality.
    order = np.argsort(-(b_item + rng.normal(0, 1.0, n_items)))
    pop = np.empty(n_items)
    pop[order] = 1.0 / np.arange(1, n_items + 1) ** 1.0
    pop /= pop.sum()
    # Side features that a two-tower model can use: noisy linear views of the truth.
    user_feat = P @ rng.normal(0, 1, (k, 12)) + rng.normal(0, 0.5, (n_users, 12))
    item_feat = np.hstack([Q, b_item[:, None]]) @ rng.normal(0, 1, (k + 1, 12)) \
        + rng.normal(0, 0.5, (n_items, 12))
    return dict(P=P, Q=Q, affinity=affinity, pop=pop, b_item=b_item,
                user_feat=user_feat.astype(np.float32), item_feat=item_feat.astype(np.float32),
                click_prob=sigmoid(2.0 * (affinity - 1.5)))


def organic_clicks(world, exposures=60, seed=1):
    """Each user is shown `exposures` items drawn by popularity, and clicks some."""
    rng = np.random.default_rng(seed)
    n_users, n_items = world["click_prob"].shape
    X = np.zeros((n_users, n_items), np.float32)
    for u in range(n_users):
        shown = rng.choice(n_items, exposures, replace=False, p=world["pop"])
        clicks = shown[rng.random(exposures) < world["click_prob"][u, shown]]
        X[u, clicks] = 1
    return X


def split_holdout(X, frac=0.2, seed=2):
    """Hold out a fraction of each user's positives for testing."""
    rng = np.random.default_rng(seed)
    train, test = X.copy(), np.zeros_like(X)
    for u in range(X.shape[0]):
        pos = np.flatnonzero(X[u])
        if len(pos) >= 2:
            held = rng.choice(pos, max(1, int(frac * len(pos))), replace=False)
            train[u, held], test[u, held] = 0, 1
    return train, test


# ---------------------------------------------------------------- models
def ials(X, k=16, alpha=10.0, reg=1.0, iters=10, seed=0):
    """Implicit ALS (Hu, Koren and Volinsky, 2008): every cell is a 0/1 preference
    with confidence 1 + alpha * x. Dense solves are fine at this size."""
    rng = np.random.default_rng(seed)
    n_u, n_i = X.shape
    U, V = rng.normal(0, 0.1, (n_u, k)), rng.normal(0, 0.1, (n_i, k))
    C = 1.0 + alpha * X                              # confidence
    Pref = (X > 0).astype(float)                     # preference
    I = reg * np.eye(k)
    for _ in range(iters):
        for A, B, Cm, Pm in ((U, V, C, Pref), (V, U, C.T, Pref.T)):
            for r in range(A.shape[0]):
                Bc = B * Cm[r][:, None]              # rows of B weighted by confidence
                A[r] = np.linalg.solve(B.T @ Bc + I, Bc.T @ Pm[r])
    return U, V


def item_knn_scores(X, shrink=10.0):
    """Item-item cosine similarity; a user's score for i sums similarities to their items."""
    norms = np.sqrt(X.sum(0))
    S = (X.T @ X) / (np.outer(norms, norms) + shrink)
    np.fill_diagonal(S, 0)
    return X @ S


# ---------------------------------------------------------------- metrics
def topk_metrics(scores, train, test, k=10):
    """Full-catalogue Recall@k and NDCG@k, never recommending training items."""
    s = np.where(train > 0, -np.inf, scores)
    top = np.argsort(-s, axis=1)[:, :k]
    users = np.flatnonzero(test.sum(1) > 0)
    rec, ndcg = [], []
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    for u in users:
        hits = test[u, top[u]] > 0
        n_rel = int(test[u].sum())
        rec.append(hits.sum() / min(n_rel, k))
        ndcg.append((hits * discounts).sum() / discounts[:min(n_rel, k)].sum())
    return float(np.mean(rec)), float(np.mean(ndcg))


def sampled_hit_rate(scores, train, test, n_neg=100, k=10, seed=3):
    """The common shortcut: rank each test item against n_neg random negatives only."""
    rng = np.random.default_rng(seed)
    hits = []
    for u, i in zip(*np.nonzero(test)):
        cand = np.flatnonzero((train[u] == 0) & (test[u] == 0))
        neg = rng.choice(cand, n_neg, replace=False)
        hits.append((scores[u, neg] >= scores[u, i]).sum() < k)
    return float(np.mean(hits))


def gini(x):
    """Gini coefficient of exposure counts: 0 = equal, near 1 = concentrated."""
    x = np.sort(np.asarray(x, float))
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1) @ x / (n * x.sum()))
```

### Script 01: explicit matrix factorization

```python
"""Explicit-feedback matrix factorization (biased MF trained by SGD) on synthetic
ratings, compared with baselines and with SVD of an imputed matrix. Then: what the
observed-ratings test set hides when ratings are missing not at random.
"""
import numpy as np

from recdata import make_world

world = make_world()
A = world["affinity"]
n_u, n_i = A.shape
rng = np.random.default_rng(0)

# True expected rating on a 1-5 scale; an observed rating adds noise and rounds.
true_rating = np.clip(3.0 + 0.7 * (A - A.mean()), 1, 5)
# Missing not at random: people see popular items and rate what they like.
w = (world["pop"][None, :] ** 0.5) * np.exp(0.8 * A)
cells = rng.choice(n_u * n_i, 15_000, replace=False, p=(w / w.sum()).ravel())
u_all, i_all = np.divmod(cells, n_i)
r_all = np.clip(np.round(true_rating[u_all, i_all] + rng.normal(0, 0.5, len(cells))), 1, 5)
perm = rng.permutation(len(cells))
tr, va, te = np.split(perm, [int(0.8 * len(perm)), int(0.9 * len(perm))])
print(f"{len(cells)} ratings, density {len(cells) / (n_u * n_i):.1%}, "
      f"mean observed rating {r_all.mean():.2f} vs mean true rating {true_rating.mean():.2f}")


def rmse(pred, idx):
    return np.sqrt(np.mean((pred(u_all[idx], i_all[idx]) - r_all[idx]) ** 2))


def fit_mf(k, lam, epochs=40, lr=0.01, seed=0):
    """r_hat = mu + b_u + b_i + p_u . q_i, squared error + L2, plain SGD."""
    g = np.random.default_rng(seed)
    mu = r_all[tr].mean()
    bu, bi = np.zeros(n_u), np.zeros(n_i)
    Pu, Qi = g.normal(0, 0.1, (n_u, k)), g.normal(0, 0.1, (n_i, k))
    for _ in range(epochs):
        for t in g.permutation(tr):
            u, i, r = u_all[t], i_all[t], r_all[t]
            e = r - (mu + bu[u] + bi[i] + Pu[u] @ Qi[i])
            bu[u] += lr * (e - lam * bu[u])
            bi[i] += lr * (e - lam * bi[i])
            Pu[u], Qi[i] = (Pu[u] + lr * (e * Qi[i] - lam * Pu[u]),
                            Qi[i] + lr * (e * Pu[u] - lam * Qi[i]))
    return lambda u, i: np.clip(mu + bu[u] + bi[i] + np.sum(Pu[u] * Qi[i], axis=-1), 1, 5)


# ---- 1. Baselines and MF ------------------------------------------------------
mu = r_all[tr].mean()
print("\n1. RMSE on observed ratings              train    valid    test")
models = {"global mean": lambda u, i: np.full(len(np.atleast_1d(u)), mu),
          "oracle (true expectation)": lambda u, i: true_rating[u, i],
          "biases only (k=0)": fit_mf(0, 0.05)}
for k, lam in [(8, 0.0), (8, 0.05), (32, 0.0), (32, 0.05), (32, 0.2)]:
    models[f"MF k={k:<2d} lambda={lam}"] = fit_mf(k, lam)
for name, f in models.items():
    print(f"   {name:26s}       {rmse(f, tr):.3f}    {rmse(f, va):.3f}    {rmse(f, te):.3f}")

# ---- 2. Why not just take the SVD? -----------------------------------------------
# Plain SVD needs a full matrix. Filling the gaps with the mean treats "not rated"
# as "rated average", which is false for 94% of the cells.
print("\n2. Truncated SVD after filling missing cells with the mean")
M = np.full((n_u, n_i), mu)
M[u_all[tr], i_all[tr]] = r_all[tr]
Uu, s, Vt = np.linalg.svd(M - mu, full_matrices=False)
for k in (8, 32):
    Mk = mu + (Uu[:, :k] * s[:k]) @ Vt[:k]
    f = lambda u, i, Mk=Mk: np.clip(Mk[u, i], 1, 5)
    print(f"   {'SVD, mean-imputed, k=' + str(k):26s}       {rmse(f, tr):.3f}    "
          f"{rmse(f, va):.3f}    {rmse(f, te):.3f}")

# ---- 3. What the observed test set hides -------------------------------------
# Score every model on 20,000 random (user, item) cells against the TRUE expected
# rating. A real system can only approximate this with a randomized experiment.
ru, ri = rng.integers(0, n_u, 20_000), rng.integers(0, n_i, 20_000)
print("\n3. Observed test cells vs random cells (against the true expected rating)")
print("   model                         RMSE test   RMSE random   mean pred random")
for name in ("oracle (true expectation)", "global mean", "biases only (k=0)",
             "MF k=8  lambda=0.05", "MF k=32 lambda=0.2"):
    f = models[name]
    rand_err = np.sqrt(np.mean((f(ru, ri) - true_rating[ru, ri]) ** 2))
    print(f"   {name:26s}      {rmse(f, te):.3f}       {rand_err:.3f}         "
          f"{f(ru, ri).mean():.2f}")
print(f"   (mean true rating over random cells: {true_rating[ru, ri].mean():.2f})")
```

```text
15000 ratings, density 6.2%, mean observed rating 4.16 vs mean true rating 3.00

1. RMSE on observed ratings              train    valid    test
   global mean                      0.931    0.917    0.898
   oracle (true expectation)        0.517    0.505    0.498
   biases only (k=0)                0.819    0.882    0.852
   MF k=8  lambda=0.0               0.409    0.770    0.744
   MF k=8  lambda=0.05              0.484    0.779    0.751
   MF k=32 lambda=0.0               0.315    0.937    0.898
   MF k=32 lambda=0.05              0.472    0.889    0.857
   MF k=32 lambda=0.2               0.781    0.873    0.844

2. Truncated SVD after filling missing cells with the mean
   SVD, mean-imputed, k=8           0.842    0.899    0.880
   SVD, mean-imputed, k=32          0.683    0.898    0.881

3. Observed test cells vs random cells (against the true expected rating)
   model                         RMSE test   RMSE random   mean pred random
   oracle (true expectation)       0.498       0.000         3.00
   global mean                     0.898       1.576         4.15
   biases only (k=0)               0.852       1.464         3.95
   MF k=8  lambda=0.05             0.751       1.210         3.77
   MF k=32 lambda=0.2              0.844       1.468         3.98
   (mean true rating over random cells: 3.00)
```

### Script 02: implicit feedback, three ways to score

```python
"""Top-N recommendation from implicit clicks: popularity, item-kNN and implicit ALS,
scored three ways: full-catalogue metrics on held-out clicks, the sampled-negatives
shortcut, and (only possible because the data are synthetic) true user preference.
"""
import itertools

import numpy as np

from recdata import (ials, item_knn_scores, make_world, organic_clicks, sampled_hit_rate,
                     split_holdout, topk_metrics)

world = make_world()
X = organic_clicks(world)                 # exposure was driven by popularity
train, test = split_holdout(X)
cp = world["click_prob"]
top10_share = np.sort(X.sum(0))[-10:].sum() / X.sum()
print(f"{X.shape[0]} users, {X.shape[1]} items, {int(X.sum())} clicks "
      f"({X.sum(1).mean():.1f} per user); the top 10 items get {top10_share:.0%} of clicks")


def true_precision(scores, k=10):
    """Mean TRUE click probability of the k items recommended to each user."""
    s = np.where(train > 0, -np.inf, scores)
    top = np.argsort(-s, axis=1)[:, :k]
    return float(np.take_along_axis(cp, top, axis=1).mean())


rng = np.random.default_rng(0)
U, V = ials(train, k=8, alpha=10, reg=1)
U16, V16 = ials(train, k=16, alpha=10, reg=1)
models = {
    "random": rng.random(X.shape),
    "popularity": np.tile(train.sum(0), (X.shape[0], 1)),
    "item-kNN (cosine)": item_knn_scores(train),
    "iALS k=8": U @ V.T,
    "iALS k=16": U16 @ V16.T,
    "oracle: true prefs": cp,
}
print("\n1. Scores                Recall@10  NDCG@10  sampled HR@10  true prec@10")
for name, s in models.items():
    r, n = topk_metrics(s, train, test)
    print(f"   {name:20s}  {r:8.3f} {n:8.3f}  {sampled_hit_rate(s, train, test):12.3f}"
          f"  {true_precision(s):11.3f}")

# ---- 2. Tune iALS by the offline metric, or by what users actually want --------
print("\n2. iALS grid: choose by held-out Recall@10, or by true precision@10")
rows = []
for k, alpha, reg in itertools.product((8, 32), (2, 10, 40), (0.1, 10, 50)):
    U, V = ials(train, k=k, alpha=alpha, reg=reg, iters=8)
    s = U @ V.T
    rows.append((k, alpha, reg, topk_metrics(s, train, test)[0], true_precision(s)))
rows = np.array(rows)
by_offline, by_truth = rows[rows[:, 3].argmax()], rows[rows[:, 4].argmax()]
for label, r in [("best offline Recall", by_offline), ("best true precision", by_truth)]:
    print(f"   {label:20s} k={r[0]:<3.0f} alpha={r[1]:<4.0f} reg={r[2]:<5g} "
          f"Recall@10 {r[3]:.3f}  true prec@10 {r[4]:.3f}")
corr = np.corrcoef(rows[:, 3], rows[:, 4])[0, 1]
print(f"   correlation between the two criteria over {len(rows)} configs: {corr:+.2f}")
pop_share = []
for k, alpha, reg in (by_offline[:3], by_truth[:3]):
    U, V = ials(train, k=int(k), alpha=alpha, reg=reg, iters=8)
    top = np.argsort(-np.where(train > 0, -np.inf, U @ V.T), axis=1)[:, :10]
    pop_share.append(np.isin(top, np.argsort(-world["pop"])[:20]).mean())
print(f"   share of recommendations from the 20 most-exposed items: "
      f"{pop_share[0]:.0%} (offline pick) vs {pop_share[1]:.0%} (truth pick)")
```

```text
600 users, 400 items, 9153 clicks (15.3 per user); the top 10 items get 20% of clicks

1. Scores                Recall@10  NDCG@10  sampled HR@10  true prec@10
   random                   0.029    0.018         0.089        0.214
   popularity               0.240    0.162         0.503        0.236
   item-kNN (cosine)        0.297    0.228         0.546        0.340
   iALS k=8                 0.287    0.208         0.536        0.579
   iALS k=16                0.246    0.185         0.470        0.529
   oracle: true prefs       0.185    0.115         0.466        0.952

2. iALS grid: choose by held-out Recall@10, or by true precision@10
   best offline Recall  k=32  alpha=10   reg=50    Recall@10 0.320  true prec@10 0.450
   best true precision  k=8   alpha=40   reg=0.1   Recall@10 0.236  true prec@10 0.588
   correlation between the two criteria over 18 configs: +0.26
   share of recommendations from the 20 most-exposed items: 66% (offline pick) vs 28% (truth pick)
```

Popularity's held-out Recall@10 (0.240) is eight times random's, while its true precision (0.236) is barely above random's (0.214). The full table is discussed in section 6.

### Script 03: two-tower retrieval and cold start

```python
"""A two-tower retrieval model trained with in-batch softmax, with and without the
log-frequency ("logQ") correction, and a cold-start test on items never seen in
training. Compared with implicit ALS, which has no vector for an unseen item.
"""
import numpy as np
import torch
import torch.nn as nn

from recdata import ials, make_world, organic_clicks, split_holdout, topk_metrics

torch.manual_seed(0)
torch.set_num_threads(4)
world = make_world()
X = organic_clicks(world)
train, test = split_holdout(X)
cp = world["click_prob"]
n_u, n_i = X.shape

# Hold out 20% of items completely: no training clicks at all (new catalogue items).
rng = np.random.default_rng(0)
cold = rng.choice(n_i, n_i // 5, replace=False)
warm = np.setdiff1d(np.arange(n_i), cold)
train_warm = train.copy()
train_warm[:, cold] = 0
pairs = np.argwhere(train_warm > 0)                    # (user, item) positives
q = train_warm.sum(0) / train_warm.sum()               # item sampling frequency
print(f"{len(pairs)} training clicks on {len(warm)} warm items; {len(cold)} cold items")


class Tower(nn.Module):
    def __init__(self, d_in, d_out=16):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, 64), nn.ReLU(), nn.Linear(64, d_out))

    def forward(self, x):
        return nn.functional.normalize(self.net(x), dim=-1)


def train_two_tower(logq, epochs=40, batch=256, temp=0.1):
    uf, itf = torch.tensor(world["user_feat"]), torch.tensor(world["item_feat"])
    uf, itf = (uf - uf.mean(0)) / uf.std(0), (itf - itf.mean(0)) / itf.std(0)
    ut, it = Tower(uf.shape[1]), Tower(itf.shape[1])
    opt = torch.optim.Adam(list(ut.parameters()) + list(it.parameters()), lr=3e-3)
    logq_t = torch.log(torch.tensor(q, dtype=torch.float32) + 1e-12)
    g = torch.Generator().manual_seed(0)
    for _ in range(epochs):
        for idx in torch.randperm(len(pairs), generator=g).split(batch):
            u, i = torch.tensor(pairs[idx.numpy(), 0]), torch.tensor(pairs[idx.numpy(), 1])
            logits = ut(uf[u]) @ it(itf[i]).T / temp   # every other item in the batch
            if logq:                                   # is a negative; popular items
                logits = logits - logq_t[i][None, :]   # appear more often, so correct
            loss = nn.functional.cross_entropy(logits, torch.arange(len(u)))
            opt.zero_grad()
            loss.backward()
            opt.step()
    with torch.no_grad():
        return (ut(uf) @ it(itf).T).numpy()


def true_prec(scores, items, k=10):
    """Mean true click probability of the top-k among `items` (excluding train clicks)."""
    s = np.where(train[:, items] > 0, -np.inf, scores[:, items])
    top = items[np.argsort(-s, axis=1)[:, :k]]
    return float(np.take_along_axis(cp, top, axis=1).mean())


U, V = ials(train_warm, k=8, alpha=10, reg=1)
models = {"iALS (warm items only)": U @ V.T,
          "two-tower, no logQ": train_two_tower(logq=False),
          "two-tower, logQ": train_two_tower(logq=True)}
top20 = np.argsort(-world["pop"])[:20]
print("\n                          warm items:                         cold items:")
print("   model                   Recall@10  true prec@10  top-20 share   true prec@10")
for name, s in models.items():
    sw = np.where(np.isin(np.arange(n_i), cold)[None, :], -np.inf, s)   # rank warm only
    r, _ = topk_metrics(sw, train_warm, test * np.isin(np.arange(n_i), warm)[None, :])
    top = np.argsort(-np.where(train > 0, -np.inf, sw), axis=1)[:, :10]
    share = np.isin(top, top20).mean()
    print(f"   {name:24s} {r:8.3f}  {true_prec(s, warm):11.3f}  {share:11.0%}"
          f"   {true_prec(s, cold):12.3f}")
print(f"   {'random':24s} {'':8s}  {true_prec(rng.random((n_u, n_i)), warm):11.3f}  {'':11s}"
      f"   {true_prec(rng.random((n_u, n_i)), cold):12.3f}")
print(f"   {'oracle':24s} {'':8s}  {true_prec(cp, warm):11.3f}  {'':11s}"
      f"   {true_prec(cp, cold):12.3f}")
```

```text
5827 training clicks on 320 warm items; 80 cold items

                          warm items:                         cold items:
   model                   Recall@10  true prec@10  top-20 share   true prec@10
   iALS (warm items only)      0.295        0.554          28%          0.164
   two-tower, no logQ          0.096        0.716           1%          0.553
   two-tower, logQ             0.361        0.679          31%          0.499
   random                                   0.207                       0.203
   oracle                                   0.945                       0.766
```

### Script 04: feedback loops and off-policy evaluation

```python
"""Part 1: a recommender retrained on the clicks it caused (a feedback loop).
Part 2: evaluating new policies offline from biased logs: naive replay of clicks
versus inverse propensity scoring (IPS). Ground truth comes from the synthetic world.
"""
import numpy as np

from recdata import gini, ials, make_world, organic_clicks

world = make_world()
cp = world["click_prob"]
n_u, n_i = cp.shape
K, ROUNDS = 10, 8
top20 = np.argsort(-world["pop"])[:20]


def choose(policy, clicks, rng):
    """Return a (users x K) slate of items for each user."""
    seen = clicks > 0
    if policy == "random":
        s = rng.random((n_u, n_i))
    elif policy == "oracle":
        s = cp.copy()
    elif policy == "popularity":
        s = np.tile(clicks.sum(0), (n_u, 1)) + 1e-3 * rng.random((n_u, n_i))
    else:                                              # "iALS" or "iALS+explore"
        U, V = ials(clicks, k=8, alpha=10, reg=1, iters=6)
        s = U @ V.T
    s = np.where(seen, -np.inf, s)                     # do not re-show clicked items
    slate = np.argsort(-s, axis=1)[:, :K]
    if policy == "iALS+explore":                       # 2 of 10 slots: random unseen items
        for u in range(n_u):
            pool = np.setdiff1d(np.flatnonzero(~seen[u]), slate[u, :K - 2])
            slate[u, K - 2:] = rng.choice(pool, 2, replace=False)
    return slate


def jaccard_homogeneity(slates, rng, pairs=3000):
    a, b = rng.integers(0, n_u, pairs), rng.integers(0, n_u, pairs)
    return np.mean([len(set(slates[x]) & set(slates[y])) / len(set(slates[x]) | set(slates[y]))
                    for x, y in zip(a, b) if x != y])


print("Part 1: 8 rounds; each round every user sees 10 new items and clicks some.")
print("policy        clicks/user  CTR rd1  CTR rd8  coverage rd8  exposure Gini  top-20  homog.")
for policy in ("random", "popularity", "iALS", "iALS+explore", "oracle"):
    rng = np.random.default_rng(0)
    clicks = organic_clicks(world)                     # same starting log for everyone
    exposure = np.zeros(n_i)
    ctr = []
    for t in range(ROUNDS):
        slate = choose(policy, clicks, rng)
        p = np.take_along_axis(cp, slate, axis=1)
        new = rng.random(p.shape) < p
        np.put_along_axis(clicks, slate, np.maximum(np.take_along_axis(clicks, slate, 1), new), 1)
        np.add.at(exposure, slate.ravel(), 1)
        ctr.append(new.mean())
    print(f"{policy:14s}  {K * sum(ctr):9.2f}  {ctr[0]:7.3f}  {ctr[-1]:7.3f}"
          f"  {len(np.unique(slate)) / n_i:12.0%}  {gini(exposure):13.2f}"
          f"  {np.isin(slate, top20).mean():6.0%}  {jaccard_homogeneity(slate, rng):6.3f}")

# ---------------------------------------------------------------- Part 2
# A logging policy shows ONE item per event, mostly by popularity, with 10% uniform
# exploration, and records the probability of what it showed (the propensity).
rng = np.random.default_rng(1)
pi0 = 0.9 * world["pop"] + 0.1 / n_i
events = 60_000
users = rng.integers(0, n_u, events)
shown = rng.choice(n_i, events, p=pi0)
clicked = rng.random(events) < cp[users, shown]
prop = pi0[shown]

# Candidate policies to evaluate, each recommending one item per user.
U, V = ials(organic_clicks(world), k=8, alpha=10, reg=1)
targets = {"popularity top-1": np.full(n_u, np.argmax(world["pop"])),
           "iALS top-1": np.argmax(U @ V.T, axis=1),
           "oracle top-1": np.argmax(cp, axis=1)}

print("\nPart 2: offline estimates of each policy's click rate from the logged events")
print("policy             true   replay   IPS (95% CI)            SNIPS   matched events")
for name, rec in targets.items():
    truth = cp[np.arange(n_u), rec].mean()
    match = shown == rec[users]
    replay = clicked[match].mean()                     # average over matched events only
    # IPS: reweight matched events by 1 / propensity so rare actions count more.
    w = match * clicked / prop
    ips, se = w.mean(), w.std(ddof=1) / np.sqrt(events)
    snips = w.sum() / (match / prop).sum()             # self-normalized IPS
    print(f"{name:17s}  {truth:5.3f}  {replay:6.3f}   {ips:5.3f} [{ips - 1.96 * se:.3f}, "
          f"{ips + 1.96 * se:.3f}]   {snips:5.3f}  {match.sum():8d}")

# The most common offline protocol: did the policy's top-10 contain the logged clicks?
print("\n'Hit rate' of each policy's top-10 on logged clicks (the usual offline metric):")
clicked_pairs = (users[clicked], shown[clicked])
for name, scores in [("popularity", np.tile(world["pop"], (n_u, 1))), ("iALS", U @ V.T),
                     ("oracle", cp)]:
    top10 = np.argsort(-scores, axis=1)[:, :10]
    hit = (top10[clicked_pairs[0]] == clicked_pairs[1][:, None]).any(1).mean()
    print(f"   {name:10s} {hit:.3f}   true mean click prob of its top-10: "
          f"{np.take_along_axis(cp, top10, 1).mean():.3f}")
```

```text
Part 1: 8 rounds; each round every user sees 10 new items and clicks some.
policy        clicks/user  CTR rd1  CTR rd8  coverage rd8  exposure Gini  top-20  homog.
random              15.60    0.207    0.182          100%           0.06      4%   0.014
popularity           8.02    0.203    0.057           11%           0.95     65%   0.241
iALS                27.65    0.602    0.223           44%           0.75     29%   0.051
iALS+explore        27.79    0.534    0.260           98%           0.58     22%   0.035
oracle              59.41    0.951    0.562          100%           0.28      4%   0.016

Part 2: offline estimates of each policy's click rate from the logged events
policy             true   replay   IPS (95% CI)            SNIPS   matched events
popularity top-1   0.386   0.375   0.381 [0.368, 0.394]   0.375      8353
iALS top-1         0.829   0.812   0.813 [0.735, 0.891]   0.819      1321
oracle top-1       0.989   0.989   1.109 [0.877, 1.340]   0.997       545

'Hit rate' of each policy's top-10 on logged clicks (the usual offline metric):
   popularity 0.457   true mean click prob of its top-10: 0.289
   iALS       0.423   true mean click prob of its top-10: 0.714
   oracle     0.178   true mean click prob of its top-10: 0.960
```

The popularity target in part 2 recommends the single most-exposed item to everyone, so it matches many logged events (8,353) and its estimates are tight. The more a policy differs from the logging policy, the fewer matches and the wider the interval.

## Common mistakes

- **Treating missing as negative or as average.** Mean-imputed SVD and "unclicked means disliked" both ignore that most cells were never seen.
- **Sampled metrics.** Rank against the full catalogue, or correct the estimate.
- **Weak baselines.** Tune popularity, item-kNN and MF/iALS as carefully as the new model; many published gains vanished against tuned baselines.
- **Random splits for temporal data.** Split by time when you can.
- **Taking held-out clicks at face value.** They measure exposure as well as preference. Report coverage and exposure concentration, and validate important choices online or with off-policy estimates that use logged propensities.
- **Not logging propensities.** Without the probability of what was shown, unbiased off-policy evaluation is impossible.
- **Optimizing short-term clicks only.** It invites clickbait and degenerate loops; track longer-term and survey outcomes too.
- **Forgetting cold start.** ID-only models cannot place new items or users; plan for features, fallbacks and exploration.
- **Judging embeddings by cosine alone.** For learned dot-product models, cosines between item vectors can be arbitrary (L0-02).

## Exercises

1. [warm-up] A user's held-out set has 3 items, and the top-10 list has hits at ranks 2 and 7. Compute Recall@10 and NDCG@10.
2. [warm-up] In script 01, add $k = 4$ and $k = 16$ with $\lambda = 0.05$. Where does validation RMSE bottom out, and does it match the true rank?
3. [warm-up] Why does `split_holdout` skip users with fewer than two clicks? What happens to those users in evaluation?
4. [core] Derive the ALS update for $p_u$ in biased MF by setting the gradient of the objective with respect to $p_u$ to zero (biases fixed). Then implement an ALS version of `fit_mf` and compare speed and RMSE with SGD.
5. [core] Implement BPR-MF with uniform negative sampling and add it to script 02. Compare Recall@10 and true precision with iALS.
6. [core] In `make_world`, make popularity strongly tied to quality (replace the noise std 1.0 by 0.1). Rerun scripts 02 and 03. Do the offline-versus-truth gaps and the logQ comparison change, and why?
7. [core] In script 04 part 2, lower the logging policy's uniform exploration from 10% to 1%. What happens to the number of matched events and the IPS intervals for the iALS and oracle policies?
8. [core] Add a popularity penalty to the iALS scores in script 04 part 1, $s_{ui} - \gamma \log(\text{exposure}_i + 1)$, for a few values of $\gamma$. Plot or tabulate clicks per user against exposure Gini.
9. [stretch] Add an item ID embedding to the item tower in script 03 (with dropout on the ID so the model still learns to use features). Does warm performance improve, and what happens on cold items?
10. [stretch] Implement a simple two-stage system: retrieve 50 candidates per user with the logQ two-tower model, then rank them with iALS scores (warm items) or the tower score (cold items). Measure candidate recall (the fraction of each user's true top 10 that survives retrieval) and the final true precision.

### Hints and answers

1. Recall@10 $= 2/3$. DCG $= 1/\log_2 3 + 1/\log_2 8 = 0.631 + 0.333 = 0.964$; ideal DCG for 3 items $= 1 + 0.631 + 0.5 = 2.131$; NDCG $\approx 0.452$.
2. Expect the minimum near $k = 8$, but with 15,000 noisy ratings the curve is flat around it and the exact minimum depends on $\lambda$.
3. One click cannot be split into training and test. Those users have no test items and are excluded from metric averages, a common and rarely reported filter.
4. Gradient: $-2\sum_{i \in \mathcal{K}_u} (r_{ui} - \mu - b_u - b_i - p_u^\top q_i) q_i + 2\lambda p_u = 0$, so $p_u = (Q_u^\top Q_u + \lambda I)^{-1} Q_u^\top \tilde r_u$ with $\tilde r_u$ the bias-removed ratings. ALS converges in a few sweeps and, vectorized, is far faster than the Python SGD loop.
5. Sample $(u, i, j)$, compute $d = \hat x_{ui} - \hat x_{uj}$, and step along $\sigma(-d)$ times the gradient of $d$. Expect BPR with uniform negatives to favor popular items, since they are sampled as negatives no more often than any other item but appear often as positives.
6. When exposure tracks quality, held-out clicks become a better proxy for preference, the offline and true rankings agree more, and the logQ correction stops looking harmful.
7. Matches for policies unlike the logger fall and the weights $1/\pi_0$ for rarely shown items grow, so intervals widen sharply; at zero exploration, items with $\pi_0 = 0$ cannot be evaluated at all.
8. Expect a trade-off curve: modest $\gamma$ lowers Gini with little loss in clicks, large $\gamma$ hurts clicks.
9. IDs help on warm items with enough data; on cold items the ID embedding is untrained, so without dropout the model can lean on it and degrade.
10. Candidate recall bounds what the ranker can achieve; measure both to know which stage limits the system.

## Check your understanding

**1. Why can't you just take the SVD of the rating matrix?** The SVD needs every entry. With 94% missing, you must impute, and imputing a constant asserts that unseen items would be rated average, which is false and drags the factors toward the fill value. MF instead fits only observed entries, with regularization, and is solved by SGD or ALS.

**2. In iALS, what do the confidence weights do?** Every cell enters the loss with target 1 (interacted) or 0 (not), weighted by $c_{ui} = 1 + \alpha x_{ui}$. Observed interactions count strongly, unobserved ones weakly, so "not clicked" is treated as weak evidence of dislike rather than as a hard negative or as nothing.

**3. Why does a two-tower model need the logQ correction, and when might it hurt?** In-batch negatives are sampled roughly in proportion to popularity, so popular items are over-penalized; subtracting $\log p_j$ makes training estimate the full softmax over the logged data. If the logged popularity mostly reflects past exposure rather than preference, correctly fitting the log reproduces that exposure bias, as script 03 shows.

**4. Your new model beats the old one on offline Recall@10. Why might it lose an A/B test?** The held-out clicks were produced by the old system's exposure, so offline metrics reward predicting what was shown; the new model may simply be closer to the old one. Sampled metrics, random splits and untuned baselines can add further distortion. Off-policy estimates with logged propensities, or an online test, measure what users would actually do.

**5. Do recommender systems polarize people?** The evidence is mixed. Recommenders clearly shape exposure. Field experiments on feed ranking have found small short-run effects on attitudes, audits have found pathways to extreme content, and panel studies attribute much consumption to user demand. Long-run, population-level effects are hard to identify, and studies have been contested on design grounds, so a confident answer either way goes beyond the evidence.

## Further reading

- **Google for Developers, "Recommendation Systems"** (course, last updated August 2025). https://developers.google.com/machine-learning/recommendation
  *Read for:* a short, free introduction to candidate generation, scoring and re-ranking, matrix factorization and softmax models. *Level:* intro.
- **Yehuda Koren, Robert Bell, Chris Volinsky, "Matrix Factorization Techniques for Recommender Systems"** (2009, IEEE Computer 42(8), 30 to 37). https://doi.org/10.1109/MC.2009.263
  *Read for:* biased MF, SGD and ALS, and lessons from the Netflix Prize, in a readable magazine article. Publisher version may be paywalled; author copies circulate. *Level:* intro to intermediate.
- **Yifan Hu, Yehuda Koren, Chris Volinsky, "Collaborative Filtering for Implicit Feedback Datasets"** (2008, IEEE ICDM, 263 to 272). https://doi.org/10.1109/ICDM.2008.22
  *Read for:* preference and confidence, the weighted objective and the efficient ALS trick. Publisher version may be paywalled. *Level:* intermediate.
- **Steffen Rendle, Christoph Freudenthaler, Zeno Gantner, Lars Schmidt-Thieme, "BPR: Bayesian Personalized Ranking from Implicit Feedback"** (2009, UAI; arXiv:1205.2618). https://arxiv.org/abs/1205.2618
  *Read for:* the pairwise ranking objective for implicit data. *Level:* intermediate.
- **Paul Covington, Jay Adams, Emre Sargin, "Deep Neural Networks for YouTube Recommendations"** (2016, ACM RecSys, 191 to 198). https://doi.org/10.1145/2959100.2959190
  *Read for:* the candidate generation plus ranking design and many practical lessons. Free PDF on Google Research. *Level:* intermediate.
- **Xinyang Yi, Ji Yang, Lichan Hong, Derek Zhiyuan Cheng, Lukasz Heldt, Aditee Kumthekar, Zhe Zhao, Li Wei, Ed Chi, "Sampling-Bias-Corrected Neural Modeling for Large Corpus Item Recommendations"** (2019, ACM RecSys, 269 to 277). https://research.google/pubs/sampling-bias-corrected-neural-modeling-for-large-corpus-item-recommendations/
  *Read for:* two-tower retrieval with in-batch softmax, the logQ correction and streaming frequency estimation. *Level:* advanced.
- **Walid Krichene and Steffen Rendle, "On Sampled Metrics for Item Recommendation"** (2020, ACM KDD, 1748 to 1757). https://doi.org/10.1145/3394486.3403226
  *Read for:* why sampled ranking metrics mislead, and corrections. *Level:* intermediate.
- **Maurizio Ferrari Dacrema, Paolo Cremonesi, Dietmar Jannach, "Are We Really Making Much Progress? A Worrying Analysis of Recent Neural Recommendation Approaches"** (2019, ACM RecSys; arXiv:1907.06902). https://arxiv.org/abs/1907.06902
  *Read for:* a reproducibility study showing the value of tuned simple baselines. *Level:* intermediate.
- **Steffen Rendle, Walid Krichene, Li Zhang, John Anderson, "Neural Collaborative Filtering vs. Matrix Factorization Revisited"** (2020, arXiv:2005.09683). https://arxiv.org/abs/2005.09683
  *Read for:* why a tuned dot product beats learned MLP similarity and why dot products matter for retrieval. *Level:* intermediate.
- **Tobias Schnabel, Adith Swaminathan, Ashudeep Singh, Navin Chandak, Thorsten Joachims, "Recommendations as Treatments: Debiasing Learning and Evaluation"** (2016, ICML; arXiv:1602.05352). https://arxiv.org/abs/1602.05352
  *Read for:* propensity-weighted learning and evaluation under selection bias. *Level:* advanced.
- **Allison J.B. Chaney, Brandon M. Stewart, Barbara E. Engelhardt, "How Algorithmic Confounding in Recommendation Systems Increases Homogeneity and Decreases Utility"** (2018, ACM RecSys; arXiv:1710.11214). https://arxiv.org/abs/1710.11214
  *Read for:* simulations of the feedback loop that script 04 imitates. *Level:* intermediate.
- **Andrew M. Guess, Neil Malhotra, Jennifer Pan, Pablo Barberá, et al., "How do social media feed algorithms affect attitudes and behavior in an election campaign?"** (2023, Science 381, 398 to 404). https://doi.org/10.1126/science.abp9364
  *Read for:* a large field experiment on algorithmic versus chronological feeds; read with the Bagchi et al. eLetter. May be paywalled. *Level:* intermediate.
