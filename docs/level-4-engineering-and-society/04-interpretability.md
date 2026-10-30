---
title: "Interpretability"
---

> **Level 4 · Module 4** · Prerequisites: L1-02, L2-01, L2-02, L2-05, L4-03 · Time: 7 to 9 hours · Code: `code/L4-04-interpretability/`

## What you will be able to do

- Explain what different interpretability methods claim to show, and separate *information is present* (correlation) from *the model uses it* (causation).
- Train and evaluate probes with proper baselines and control tasks, and test whether a probed direction matters by removing it.
- Compute gradient, gradient-times-input, integrated-gradients and guided-backprop attributions, and run the sanity checks that some of them fail.
- Train a two-layer attention-only transformer on CPU, watch induction heads form in a sudden phase change, and confirm their role with ablations.
- Reproduce superposition in a toy model, train a sparse autoencoder on data with known features, and measure what it recovers.
- Assess claims in mechanistic interpretability, including the critiques of sparse autoencoders and circuit analyses, as of October 2026.

## 1. Why interpret, and what counts as an explanation

A trained network is a precise but unreadable description of a function. We open it up for several different reasons, and they need different evidence:

- **Debugging.** A classifier that detects pneumonia by reading the hospital's watermark is accurate on the test set and useless elsewhere. Explanations can reveal such shortcuts before deployment.
- **Science.** What algorithm did the network learn? Is it the one we expected?
- **Safety and auditing.** Behavioral tests (L4-03) only cover inputs we thought to try. If we could read what a model represents, we might detect problems, such as a model recognizing that it is being evaluated, that behavior alone does not reveal (L4-05).
- **Accountability.** People affected by a decision may be owed a reason.

Two properties of an explanation are easy to confuse. **Plausibility** is whether it looks sensible to a human. **Faithfulness** is whether it reflects what the model actually computes. A saliency map that outlines the object is plausible; it is faithful only if the model's output really depends on those pixels in the way the map says. Much of this module is about methods that are plausible and less faithful than they look.

An alternative to explaining a black box is not building one. Rudin argues that for high-stakes decisions, people should use inherently interpretable models (sparse linear models, small rule lists, scoring systems) rather than post-hoc explanations of black boxes, because the explanations can be wrong in ways nobody notices and because the accuracy cost of interpretable models is often smaller than assumed [Rudin, 2019]. Coefficients of linear models and feature importances of tree ensembles, with their own pitfalls, are in L1-02 and L1-03. This module is about deep networks, where that option is usually not available.

Methods divide roughly into three families. **Probing** asks what information a representation contains. **Attribution** asks which parts of the input a particular output depends on. **Mechanistic interpretability** tries to reverse-engineer the computation into human-understandable parts (features and circuits) and to verify the account by intervening on the model.

## 2. Probing

A **probe** is a simple classifier trained to predict a property (part of speech, the previous token, whether a digit is even) from a model's internal activations ([Alain and Bengio, 2016](https://arxiv.org/abs/1610.01644)). If a linear probe succeeds on held-out data, the property is linearly decodable from that layer. Probing is cheap and has produced many findings about what language models represent. Belinkov's review lists its shortcomings [Belinkov, 2022]; three of them matter most in practice.

**Decodable compared with what?** Representations of almost anything contain a lot of information about the input, including from untrained networks. Script 1 trains an MLP on scikit-learn's 8x8 digits (96.8% test accuracy) and probes its second hidden layer for "is the digit even?" with logistic regression: 0.958. That sounds like the network learned parity. But the same probe on a *randomly initialized* network gets 0.815, and on raw pixels 0.922. Always report a random-network baseline and an input baseline.

**Control tasks.** Hewitt and Liang proposed comparing a probe's accuracy on the real property with its accuracy on a **control task**: a property with the same structure but no linguistic meaning, such as assigning each word type a random label. A good probe should be **selective**, achieving high accuracy on the real task and low accuracy on the control. They found that popular probes on ELMo were not very selective, and that the choice of probe and its regularization changed the conclusions [Hewitt and Liang, 2019]. Script 1's control is a random split of the ten digit classes into two groups of five, repeated 20 times. The trained layer decodes random splits as well as it decodes parity (0.960 against 0.958, selectivity $-0.002$). The probe found that the layer encodes the digit's identity, and any function of the identity comes along for free. "The network represents parity" is not supported.

**Decodable is not used.** A probe shows that information is present, not that the model relies on it. **Amnesic probing** tests use instead: remove the property from the representation and measure how the model's behavior changes; Elazar et al. found that probing accuracy did not track how much the model relied on a property ([Elazar et al., 2021](https://aclanthology.org/2021.tacl-1.10/), TACL). Script 1 does a one-step version. Image brightness ("ink") is decodable from layer 2 with accuracy 0.922, although the network was never trained on it. Projecting the probe's direction out of the layer leaves classification accuracy unchanged (0.967 against 0.968), so the network's output does not depend on that direction. A second ink probe trained after the removal still reaches 0.900, so the information was spread over many directions; methods like amnesic probing iterate the removal for this reason. Removing the ten directions found by a *class* probe drops accuracy to 0.207, while ten random directions leave it at 0.960. That is causal evidence that the class-probe subspace matters, with a caveat: the final layer reads its outputs from essentially that subspace, so the result is close to tautological here. In a real model the interesting cases are the intermediate ones.

## 3. Attribution methods and their sanity checks

**Attribution** assigns each input feature a score for its contribution to one output, here the logit $f_c(x)$ of class $c$ for an input $x \in \mathbb{R}^d$:

- **Gradient (saliency)**: $\partial f_c / \partial x_i$, the sensitivity of the output to an infinitesimal change in feature $i$.
- **Gradient times input**: $x_i \, \partial f_c / \partial x_i$, a first-order estimate of the effect of setting $x_i$ to zero.
- **Integrated gradients (IG)** [Sundararajan et al., 2017]: integrate the gradient along a straight path from a baseline $x'$ (here an all-black image) to $x$:
  $$\mathrm{IG}_i(x) = (x_i - x'_i) \int_0^1 \frac{\partial f_c\big(x' + \alpha (x - x')\big)}{\partial x_i}\, d\alpha.$$
  The authors derive IG from axioms, including *sensitivity* and *implementation invariance*. It also satisfies **completeness**: by the fundamental theorem of calculus along the path, the attributions sum exactly to $f_c(x) - f_c(x')$. Script 2 checks this numerically: with a 64-step midpoint rule the median relative error is 0.0007.
- **Guided backpropagation**: the gradient, but every ReLU also blocks negative gradients on the way back. It produces crisp, image-like maps.

**Attention weights are not explanations by default.** Jain and Wallace found that attention weights often correlate poorly with gradient-based importance and that very different attention distributions can give the same prediction ([Jain and Wallace, 2019](https://arxiv.org/abs/1902.10186), NAACL). Wiegreffe and Pinter replied that whether attention "is explanation" depends on the definition and proposed tests for when it is informative ([Wiegreffe and Pinter, 2019](https://arxiv.org/abs/1908.04626), EMNLP). In a transformer the value vectors and later layers transform what attention selects, so a weight is at most one part of the story. Section 5 uses attention patterns, but backs them with ablations.

**Sanity checks.** Adebayo et al. proposed two tests that any explanation of a trained model should pass [Adebayo et al., 2018]. In the **model parameter randomization test**, re-initialize the model's weights, layer by layer from the top; a map that barely changes cannot be explaining what the model learned. In the **data randomization test**, train on randomly permuted labels; the map should differ from the map for the real model. They reported that plain gradients and GradCAM passed, while guided backpropagation and guided GradCAM were invariant to the higher layers' parameters and failed. Some methods were acting partly as edge detectors, producing outputs that resemble the input whatever the model.

Script 2 runs the parameter randomization test on a small CNN for 8x8 digits (98.1% test accuracy), measuring the Spearman rank correlation between each method's map before and after randomization (1 means unchanged). After all four layers are re-initialized:

- the **gradient** map is unrelated to the original (0.086), as it should be;
- **guided backprop** keeps a correlation of 0.553, consistent with Adebayo et al.;
- **gradient times input** and **IG** keep 0.906 and 0.903.

The last result needs care. Both methods multiply by $x$, and about half the pixels of a digit are zero, so both maps are zero there in any model. The rank correlation is then largely a correlation with the input itself: the trained maps correlate 0.934 and 0.936 with $|x|$. Restricting the comparison to the pixels that are on lowers the correlation to 0.388 and 0.374, still well above the gradient's 0.031. So these methods *do* depend on the learned weights, but a large part of what they show is the input. That is fine for some purposes and misleading if you read the map as "where the model looks".

**Faithfulness tests have their own problems.** A common check deletes the most-attributed pixels and measures how fast the output falls. In script 2, zeroing the top 8 pixels by IG lowers the class logit by 12.96 on average, against 3.78 for 8 random "on" pixels, and gradient times input does nearly as well (12.70). But deletion favors methods that point at bright pixels, which is what multiplying by $x$ does, and deleted images are off the training distribution, so the model's response may say more about the artifact than about the original decision. Use several checks, and treat agreement between them as evidence, not proof.

## 4. Mechanistic interpretability: features, circuits and interventions

Mechanistic interpretability tries to explain a network's computation in terms of parts. The program was set out for vision models in "Zoom In" as three claims, framed by the authors as speculative: **features** are the fundamental units of networks and correspond to directions in activation space; features are connected by weights into **circuits** that can be studied rigorously; and analogous features and circuits form across different models and tasks (**universality**) [Olah et al., 2020].

**The residual stream view of transformers.** Elhage et al. analysed small attention-only transformers (L2-05's architecture without MLPs) by treating the residual stream as a shared channel that every layer reads from and writes to linearly [Elhage et al., 2021]. A model with no layers can only use the current token, so it models bigram statistics. A one-layer attention-only model behaves like an ensemble of bigram and "skip-trigram" rules (if token A appears somewhere earlier and the current token is B, predict C). Two layers allow **composition**: a head in layer 1 can use, as its query or key, information that a layer-0 head wrote. One composition is especially important, the **induction head** (section 5).

**Interventions make claims causal.** Watching activations gives correlations. To show that a component *does* something, change it and see what happens:

- **Ablation**: replace a component's output with zero or with its mean over a dataset, and measure the change in behavior.
- **Activation patching**: run the model on a "clean" and a "corrupted" input, copy one component's activation from one run into the other, and measure how much behavior moves.
- **Circuit discovery**: find the minimal set of components whose patching accounts for a behavior. Wang et al. explained how GPT-2 small completes sentences such as "When Mary and John went to the store, John gave a drink to" with "Mary": a circuit of 26 attention heads in 7 classes in the arXiv version (28 heads in the ICLR 2023 version), evaluated with criteria of faithfulness, completeness and minimality [Wang et al., 2023]. Nanda et al. fully reverse-engineered small transformers trained on modular addition, which use discrete Fourier transforms and trigonometric identities to turn addition into rotation around a circle ([Nanda et al., 2023](https://arxiv.org/abs/2301.05217), ICLR).

Interventions can mislead too. Meng et al. used causal tracing (patching hidden states into a corrupted run) to locate factual associations in particular MLP layers of GPT and edited facts there (ROME; [Meng et al., 2022](https://arxiv.org/abs/2202.05262), NeurIPS). Hase et al. then found that where causal tracing localizes a fact gives no insight into which layer is best to edit; the choice of edited layer predicted editing success far better ([Hase et al., 2023](https://arxiv.org/abs/2301.04213), NeurIPS). Makelov et al. showed that patching a *subspace* can change behavior by activating a dormant pathway rather than the mechanism the model normally uses, an "interpretability illusion" ([Makelov et al., 2024](https://arxiv.org/abs/2311.17030), ICLR). A position paper from May 2026 argues that the field often presents validation metrics (faithfulness scores, ablation effects) as if they identified causal mechanisms, and proposes that papers state their identification assumptions explicitly ([Lin and Liu, 2026](https://arxiv.org/abs/2605.08012), arXiv preprint).

## 5. Induction heads

An **induction head** implements "if the current token is A, find an earlier A in the context and predict the token that came after it": on `... A B ... A`, predict `B`. It needs two steps, so two heads in different layers:

1. A **previous-token head** (layer 0) copies, at each position $j$, information about the token at $j - 1$ into the residual stream.
2. The **induction head** (layer 1) at query position $i$ (current token A) attends to keys whose *previous-token* information says "A": exactly the positions right after earlier copies of A. It then copies the token found there (B) to the output.

The key of step 2 is built from the output of step 1, which is K-composition in Elhage et al.'s terms. One layer cannot do this, because a single attention layer's keys contain only the token and position at each key position.

Olsson et al. argued that induction heads may be the mechanism behind most **in-context learning** in transformers. They defined an in-context learning score (the loss on the 500th token of a context minus the average loss on the 50th), found that induction heads form in a sudden **phase change** early in training that coincides with a jump in this score, and gave six lines of evidence. They describe the evidence as causal for small attention-only models and correlational for large models with MLPs [Olsson et al., 2022]. That distinction is the honest summary: for big models, the claim is plausible and supported, not proven.

**Script 3** reproduces the small-model case on CPU in about 12 seconds. The data are random tokens from a 64-token vocabulary: a random segment of length $L$ between 8 and 20, repeated to fill 40 positions. The first copy is unpredictable (loss $\ln 64 = 4.159$). After it, every token is predictable by induction. Because $L$ varies, a head cannot solve the task by attending to a fixed offset. The model is attention-only with 2 layers, 4 heads per layer and width 64, written out in full (rather than imported from L2-05's `gpt.py`) so that it can store attention patterns and switch heads off.

Two measurements per head:

- **Previous-token score**: average attention from position $i$ to $i - 1$.
- **Induction score**: at positions whose next token is a repeat, the total attention on keys $j$ whose previous token equals the current token, $x_{j-1} = x_i$.

What the run shows:

- **A phase change.** Loss on the repeated part sits near chance for 300 steps (3.648 at step 300), then falls to 1.599 by step 400 and 0.916 by step 450. The best induction score jumps from 0.124 to 0.743 over the same 100 steps. Rerunning with seeds 1 to 3 put the drop between steps 350 and 500 every time (I checked this separately; the script runs seed 0).
- **The expected division of labour.** All four layer-1 heads become induction heads (scores 0.80 to 0.84). In layer 0, head 1 is mainly a previous-token head (0.49), with head 0 helping (0.25).
- **Ablations confirm the roles.** Zeroing all four layer-1 heads sends the repeated-part loss from 0.336 to 4.692, worse than chance. Zeroing the single best previous-token head sends it to 3.891: the induction heads still exist, but without their input they cannot find the right position. Zeroing one induction head only raises the loss to 0.603, because the other three are redundant copies. This redundancy is common in real models and is one reason single-component ablations can understate importance.
- **A one-layer control.** The same training for a one-layer model reaches 2.932 on the repeated part, which is close to the 2.577 a model would get by spreading its bet evenly over the segment's tokens. It learns "predict something that appeared earlier" (a skip-trigram style rule) but cannot learn "predict what came *after* the earlier copy".

The loss on the non-repeated part rises slightly above chance (4.304 against 4.159): having learned to copy, the model wrongly expects repeats inside the first copy too. A small model trained on a narrow task can show a mechanism cleanly; that is the strength and the limit of this kind of experiment.

## 6. Superposition and sparse autoencoders

If features were neurons, interpretability would be easier: read each neuron. In practice many neurons are **polysemantic**, responding to several unrelated things. Elhage et al. proposed **superposition** as an explanation: a layer with $m$ dimensions can represent more than $m$ features as non-orthogonal directions if the features are **sparse** (rarely active at the same time), accepting a little interference when they co-occur. In toy models they found that whether features are stored in superposition depends on sparsity and importance, with phase changes between regimes and links to the geometry of uniform polytopes [Elhage et al., 2022].

Script 4 part 1 trains their toy setup: 20 features with geometrically decreasing importance, compressed into 5 dimensions by $h = Wx$ and reconstructed as $\hat x = \mathrm{ReLU}(W^\top h + b)$ with importance-weighted squared error. When every feature is always active, the model represents exactly 5 features, orthogonally (interference 0.000), and ignores the rest, as PCA would. As features become sparser it packs in more: 9 at activation probability 0.3, 11 at 0.1 and 13 at 0.03, with growing interference (the summed squared overlap of a represented feature with the others, 1.868 at 0.03).

**Sparse autoencoders (SAEs)** try to undo superposition. Given activations $a \in \mathbb{R}^D$, learn an overcomplete dictionary of $N > D$ directions and sparse nonnegative codes:

$$f = \mathrm{ReLU}\big(W_e (a - b_d) + b_e\big), \qquad \hat a = W_d f + b_d, \qquad \mathcal{L} = \lVert a - \hat a \rVert^2 + \lambda \sum_k f_k \lVert W_{d,k} \rVert.$$

The hope is that each column of $W_d$ is one feature the model uses. (The penalty multiplies by the decoder norm so the SAE cannot shrink codes by inflating decoder columns.) Cunningham et al. reported that SAE features on language models were more interpretable and monosemantic than neurons or other decompositions ([Cunningham et al., 2024](https://arxiv.org/abs/2309.08600), ICLR). Bricken et al. applied SAEs to a one-layer transformer's 512-neuron MLP layer, with dictionaries from 512 to 131,072 features [Bricken et al., 2023]. Templeton et al. scaled this to Claude 3 Sonnet with SAEs of about 1, 4 and 34 million features, and showed that clamping a feature associated with the Golden Gate Bridge to a high value made the model talk as if it were the bridge [Templeton et al., 2024]. Gao et al. trained a 16-million-latent TopK SAE on GPT-4 activations ([Gao et al., 2024](https://arxiv.org/abs/2406.04093)), and Google DeepMind released Gemma Scope, open SAEs on every layer of Gemma 2 2B and 9B ([Lieberum et al., 2024](https://arxiv.org/abs/2408.05147)).

On a real model nobody knows the true features, so it is hard to tell whether an SAE found them. Script 4 part 2 uses data where we do know: 96 random unit directions in 32 dimensions, with about 3 active per sample. An SAE with 192 latents is trained at several penalties $\lambda$:

- With no penalty it reconstructs perfectly (fraction of variance unexplained, FVU, 0.000) and matches **none** of the true features: a dense code is just another basis.
- At $\lambda = 0.3$ it recovers all 96 features with cosine above 0.9 (mean best cosine 0.985), at the cost of FVU 0.121. At $\lambda = 1.0$ recovery is slightly cleaner and reconstruction worse (FVU 0.320): the penalty shrinks the codes.
- PCA, limited to 32 orthogonal directions, matches none.
- The average number of active latents (L0) is 13.8 at $\lambda = 0.3$, against 3 true features per sample, and 150 of the 192 latents match some true feature, so many features are covered by more than one latent. Reconstruction, sparsity and interpretability pull against each other, and the "best" SAE depends on which you value.
- Two SAEs from different seeds at $\lambda = 0.3$ share 153 of 192 latents. On this easy synthetic problem SAEs are fairly consistent; section 7 reports that on real models they are much less so.

## 7. What has and has not been shown

A fair summary as of October 2026, based on the sources below:

**Established, at least in small or specific settings.** Small models implement identifiable algorithms that can be verified causally: induction heads in attention-only models, modular addition, the indirect-object circuit in GPT-2 small. Polysemanticity is real, superposition is a good explanation of it in toy models, and SAEs and related dictionary methods find many features that humans judge interpretable, some of which steer behavior when intervened on. Probing and attribution are useful when used with baselines and causal checks.

**Contested or not shown.** That these methods scale to complete explanations of large models, and that SAE features are the model's "true" features:

- **SAEs on downstream tasks.** Kantamneni et al. tested SAE-based probes against strong baselines across data scarcity, class imbalance, label noise and covariate shift, and found that adding SAE probes did not consistently improve over baselines ([Kantamneni et al., 2025](https://arxiv.org/abs/2502.16681), ICML). In March 2025 the Google DeepMind interpretability team reported that SAEs underperformed simple linear probes at detecting harmful intent out of distribution, and announced that it was deprioritizing fundamental SAE research ([Smith et al., 2025](https://www.alignmentforum.org/posts/4uXCAJNuPKtKBsi28), Alignment Forum).
- **Stability.** Paulo and Belrose found that in an SAE with 131K latents on a Llama 3 8B MLP layer, only 30% of features were shared across random seeds ([Paulo and Belrose, 2025](https://arxiv.org/abs/2501.16615)). Compare 153 of 192 in script 4's easy synthetic case.
- **Baselines.** Heap et al. found that SAEs trained on *randomly initialized* transformers score similarly to those on trained models on automated interpretability and reconstruction metrics; the paper's current title is "Automated Interpretability Metrics Do Not Distinguish Trained and Random Transformers" (2025, revised January 2026; [arXiv:2501.17727](https://arxiv.org/abs/2501.17727)). Korznikov et al. report a synthetic setting where SAEs explain 71% of the variance yet recover only 9% of the true features, and find simple random baselines matching trained SAEs on interpretability, sparse probing and causal editing metrics ([Korznikov et al., February 2026](https://arxiv.org/abs/2602.14111), preprint). The probing lesson of section 2 applies: without a random baseline, a high score means little. Script 4's no-penalty row is a small version (perfect reconstruction, no features).
- **Coverage.** Anthropic's circuit-tracing work replaces a model's MLPs with cross-layer transcoders and draws "attribution graphs" of the computation on a prompt ([Ameisen et al., 2025](https://transformer-circuits.pub/2025/attribution-graphs/methods.html)). Applying it to Claude 3.5 Haiku, Lindsey et al. found mechanisms for multi-step reasoning and planning ahead in poems, and say their attribution graphs gave satisfying insight for about a quarter of the prompts they tried ([Lindsey et al., 2025](https://transformer-circuits.pub/2025/attribution-graphs/biology.html)). The replacement model is an approximation, and its errors are part of what is unexplained.

**The direction of travel, as of October 2026.** A 29-author review of open problems lists the need for better methods and foundations, for connecting methods to concrete goals such as auditing, and for socio-technical work [Sharkey et al., 2025]. Some recent work moves away from dictionaries toward models that explain activations in words: in May 2026 Anthropic described natural language autoencoders, which train one language model to verbalize an activation and another to reconstruct it from the text, used them in auditing (for example, to surface unverbalized awareness of being evaluated), and listed confabulated explanations among the limitations ([Fraser-Taliente et al., 2026](https://transformer-circuits.pub/2026/nla/)). Whether interpretability will be reliable enough to support safety claims about frontier models is an open question on which researchers disagree; the critiques above are by people working in the field, not outsiders.

## Hands-on

The scripts are in `code/L4-04-interpretability/`. Run them from the repository root, for example `.venv/bin/python code/L4-04-interpretability/03_induction_heads.py`. They need PyTorch (CPU), NumPy, SciPy and scikit-learn; the digits data ship with scikit-learn, nothing is downloaded, and no pretrained weights are used. Timings on this machine (24 cores, scripts using 4 threads): 4 s, 5 s, 12 s and 30 s. Outputs are from PyTorch 2.15.0 (CPU), NumPy 2.5.4 and scikit-learn 1.9.1 on Linux; floating-point results from PyTorch can differ slightly across versions and thread counts.

### Script 1: probing pitfalls, `01_probing.py`

```python
"""Three probing pitfalls on a small MLP trained on scikit-learn's 8x8 digits.

1. A probe needs baselines: random networks and raw inputs are decodable too.
2. A probe for a concept that is a function of the class label is easy for any
   representation that encodes the class; control tasks expose this.
3. Decodable is not the same as used: remove a probe's direction and see
   whether the model's behavior changes.
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.datasets import load_digits
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

torch.manual_seed(0)
rng = np.random.default_rng(0)
X, y = load_digits(return_X_y=True)
X = (X / 16.0).astype(np.float32)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.4, random_state=0, stratify=y)


def make_mlp():
    return nn.Sequential(nn.Linear(64, 64), nn.ReLU(), nn.Linear(64, 64), nn.ReLU(),
                         nn.Linear(64, 10))


def hidden(net, x):
    """Activations after the second ReLU, the layer we probe."""
    with torch.no_grad():
        return net[:4](torch.from_numpy(x)).numpy()


trained, random_net = make_mlp(), make_mlp()
opt = torch.optim.Adam(trained.parameters(), lr=3e-3)
xt, yt = torch.from_numpy(Xtr), torch.from_numpy(ytr)
for epoch in range(300):
    loss = nn.functional.cross_entropy(trained(xt), yt)
    opt.zero_grad()
    loss.backward()
    opt.step()
with torch.no_grad():
    acc = (trained(torch.from_numpy(Xte)).argmax(1).numpy() == yte).mean()
print(f"trained MLP test accuracy: {acc:.3f}\n")


def probe_acc(Htr, Hte, ltr, lte):
    clf = LogisticRegression(max_iter=3000, C=1.0).fit(Htr, ltr)
    return clf.score(Hte, lte), clf


reps = {"trained MLP layer 2": (hidden(trained, Xtr), hidden(trained, Xte)),
        "random MLP layer 2": (hidden(random_net, Xtr), hidden(random_net, Xte)),
        "raw pixels": (Xtr, Xte)}

# ---- 1 and 2: the concept "digit is even", against control partitions -------
even = np.array([0, 2, 4, 6, 8])
controls = [rng.permutation(10)[:5] for _ in range(20)]   # random 5-vs-5 class splits
print("probe accuracy (logistic regression, held-out digits)")
print(f"{'representation':22s} {'is-even':>8s} {'control (mean of 20)':>21s} {'selectivity':>12s}")
for name, (Htr, Hte) in reps.items():
    a_even, _ = probe_acc(Htr, Hte, np.isin(ytr, even), np.isin(yte, even))
    a_ctrl = np.mean([probe_acc(Htr, Hte, np.isin(ytr, c), np.isin(yte, c))[0]
                      for c in controls])
    print(f"{name:22s} {a_even:8.3f} {a_ctrl:21.3f} {a_even - a_ctrl:12.3f}")

# ---- 3: decodable but not used ------------------------------------------------
# "Ink": is the image's total brightness above the median? The classifier was
# never asked about ink. Find the probe direction, project it out of the layer-2
# activations, and see what the rest of the network does.
Htr, Hte = reps["trained MLP layer 2"]
ink_tr = Xtr.sum(1) > np.median(Xtr.sum(1))
ink_te = Xte.sum(1) > np.median(Xtr.sum(1))
a_ink, clf_ink = probe_acc(Htr, Hte, ink_tr, ink_te)
_, clf_cls = probe_acc(Htr, Hte, ytr, yte)


def accuracy_after_removing(directions):
    """Project the directions out of layer-2 activations, then run the last layer."""
    H = Hte
    if len(directions):
        Q, _ = np.linalg.qr(directions.T)      # orthonormal basis, (64, k)
        H = Hte - (Hte @ Q) @ Q.T
    with torch.no_grad():
        out = trained[4](torch.from_numpy(H.astype(np.float32)))
    return (out.argmax(1).numpy() == yte).mean()


print(f"\nink probe accuracy on layer 2: {a_ink:.3f}")
print("classifier accuracy after projecting directions out of layer 2:")
print(f"  nothing removed                     {accuracy_after_removing(np.zeros((0, 64))):.3f}")
print(f"  ink probe direction (1 dim)         {accuracy_after_removing(clf_ink.coef_):.3f}")
rand = rng.standard_normal((1, 64))
print(f"  random direction (1 dim)            {accuracy_after_removing(rand):.3f}")
print(f"  class-probe directions (10 dims)    {accuracy_after_removing(clf_cls.coef_):.3f}")
rand10 = rng.standard_normal((10, 64))
print(f"  random directions (10 dims)         {accuracy_after_removing(rand10):.3f}")
# Iterate: after removing one ink direction, is ink still decodable? (A single
# linear direction rarely removes all the information.)
Q, _ = np.linalg.qr(clf_ink.coef_.T)
a_ink2, _ = probe_acc(Htr - (Htr @ Q) @ Q.T, Hte - (Hte @ Q) @ Q.T, ink_tr, ink_te)
print(f"ink probe retrained after removal:    {a_ink2:.3f}")
```

Output:

```text
trained MLP test accuracy: 0.968

probe accuracy (logistic regression, held-out digits)
representation          is-even  control (mean of 20)  selectivity
trained MLP layer 2       0.958                 0.960       -0.002
random MLP layer 2        0.815                 0.820       -0.005
raw pixels                0.922                 0.914        0.008

ink probe accuracy on layer 2: 0.922
classifier accuracy after projecting directions out of layer 2:
  nothing removed                     0.968
  ink probe direction (1 dim)         0.967
  random direction (1 dim)            0.968
  class-probe directions (10 dims)    0.207
  random directions (10 dims)         0.960
ink probe retrained after removal:    0.900
```

Discussed in section 2. The random-network row is the baseline most papers omit.

### Script 2: attribution and sanity checks, `02_attribution_sanity.py`

```python
"""Attribution maps on a small CNN for 8x8 digits, and two checks they can fail.

Methods: gradient (saliency), gradient x input, integrated gradients (IG),
guided backpropagation. Checks:
  * completeness: IG attributions should sum to f(x) - f(baseline);
  * model randomization (Adebayo et al., 2018): re-initialize the layers from
    the top down; a map that barely changes cannot be explaining the learned model;
  * deletion: zero the most-attributed pixels and watch the logit fall.
"""
import copy

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split

torch.manual_seed(0)
X, y = load_digits(return_X_y=True)
X = torch.tensor(X / 16.0, dtype=torch.float32).view(-1, 1, 8, 8)
y = torch.tensor(y)
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)


class GuidedReLU(torch.autograd.Function):
    """ReLU whose backward pass also zeroes negative incoming gradients."""
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.clamp_min(0)

    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        return g * (x > 0) * (g > 0)


class Act(nn.Module):
    guided = False          # class-wide switch for guided backprop

    def forward(self, x):
        return GuidedReLU.apply(x) if Act.guided else torch.relu(x)


def make_cnn():
    return nn.Sequential(
        nn.Conv2d(1, 16, 3, padding=1), Act(),
        nn.Conv2d(16, 32, 3, padding=1), Act(), nn.MaxPool2d(2),
        nn.Flatten(), nn.Linear(32 * 16, 64), Act(), nn.Linear(64, 10))


model = make_cnn()
opt = torch.optim.Adam(model.parameters(), lr=2e-3)
for epoch in range(15):
    for idx in torch.randperm(len(Xtr)).split(64):
        loss = nn.functional.cross_entropy(model(Xtr[idx]), ytr[idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
with torch.no_grad():
    print(f"CNN test accuracy: {(model(Xte).argmax(1) == yte).float().mean():.3f}")


def grad_of_logit(net, x, cls):
    x = x.clone().requires_grad_()
    net(x).gather(1, cls[:, None]).sum().backward()
    return x.grad.detach()


def attributions(net, x, cls, steps=64):
    out = {"gradient": grad_of_logit(net, x, cls)}
    out["grad x input"] = out["gradient"] * x
    # IG with a black-image baseline: average gradient along the straight path.
    total = torch.zeros_like(x)
    for a in (torch.arange(steps) + 0.5) / steps:
        total += grad_of_logit(net, a * x, cls)
    out["integrated grad"] = total / steps * x
    Act.guided = True
    out["guided backprop"] = grad_of_logit(net, x, cls)
    Act.guided = False
    return out


x, cls = Xte[:200], yte[:200]
maps = attributions(model, x, cls)

# ---- completeness of IG --------------------------------------------------------
with torch.no_grad():
    f_x = model(x).gather(1, cls[:, None]).squeeze(1)
    f_0 = model(torch.zeros_like(x)).gather(1, cls[:, None]).squeeze(1)
gap = (maps["integrated grad"].flatten(1).sum(1) - (f_x - f_0)).abs() / (f_x - f_0).abs()
print(f"IG completeness: median relative error {gap.median():.4f} with 64 steps")


def similarity(a, b, on_only=False):
    """Mean Spearman rank correlation of |attribution| maps, per image.
    on_only: use only pixels where the input is nonzero (about half of them)."""
    rs = []
    for i in range(len(a)):
        keep = (x[i].flatten() > 0) if on_only else torch.ones(64, dtype=torch.bool)
        rs.append(spearmanr(a[i].abs().flatten()[keep], b[i].abs().flatten()[keep])[0])
    return np.mean(rs)


# ---- cascading model randomization -------------------------------------------
layers = [i for i, m in enumerate(model) if isinstance(m, (nn.Conv2d, nn.Linear))]
print("\nRank correlation with the trained model's maps after re-initializing layers")
print("from the top down (1.0 = map unchanged; low = map depends on learned weights)")
print(f"{'layers randomized':18s}" + "".join(f"{k:>17s}" for k in maps))
rand_model = copy.deepcopy(model)
torch.manual_seed(1)
for depth, li in enumerate(reversed(layers), start=1):
    rand_model[li].reset_parameters()
    rmaps = attributions(rand_model, x, cls)
    print(f"top {depth} of {len(layers):<10d}" +
          "".join(f"{similarity(maps[k], rmaps[k]):17.3f}" for k in maps))
print(f"{'all 4, on pixels':18s}" +
      "".join(f"{similarity(maps[k], rmaps[k], on_only=True):17.3f}" for k in maps))

# What does the input alone look like as an "explanation"?
edges = x.abs()
print(f"\nrank correlation of each trained map with the raw input |x|:")
print("".join(f"{k:>17s}" for k in maps))
print("".join(f"{similarity(maps[k], edges):17.3f}" for k in maps))

# ---- deletion test ---------------------------------------------------------------
# Zero the k most-attributed pixels (by signed attribution) and measure the
# drop in the class logit; compare with deleting k random "on" pixels.
print("\nmean drop in class logit after zeroing the top-8 pixels")
with torch.no_grad():
    for k_name, m in list(maps.items()) + [("random", torch.rand_like(x) * (x > 0))]:
        top = m.flatten(1).topk(8, dim=1).indices
        xd = x.flatten(1).clone().scatter_(1, top, 0).view_as(x)
        drop = f_x - model(xd).gather(1, cls[:, None]).squeeze(1)
        print(f"  {k_name:16s} {drop.mean():6.2f}")
```

Output:

```text
CNN test accuracy: 0.981
IG completeness: median relative error 0.0007 with 64 steps

Rank correlation with the trained model's maps after re-initializing layers
from the top down (1.0 = map unchanged; low = map depends on learned weights)
layers randomized          gradient     grad x input  integrated grad  guided backprop
top 1 of 4                     0.301            0.913            0.917            0.783
top 2 of 4                     0.155            0.905            0.907            0.758
top 3 of 4                     0.143            0.909            0.911            0.686
top 4 of 4                     0.086            0.906            0.903            0.553
all 4, on pixels              0.031            0.388            0.374            0.513

rank correlation of each trained map with the raw input |x|:
         gradient     grad x input  integrated grad  guided backprop
            0.225            0.934            0.936            0.652

mean drop in class logit after zeroing the top-8 pixels
  gradient          10.42
  grad x input      12.70
  integrated grad   12.96
  guided backprop   10.70
  random             3.78
```

Discussed in section 3. The guided-backprop correlation falls as more layers are randomized but stays high; Adebayo et al. found it largely invariant to the higher layers in much larger networks.

### Script 3: induction heads, `03_induction_heads.py`

```python
"""Watch induction heads form in a tiny attention-only transformer, then test them.

Data: a random segment of random length, repeated. The first copy cannot be
predicted; later copies can, by "find where the current token appeared before
and copy the token that followed it", which is what an induction head does.
"""
import math
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

V, T = 64, 40            # vocabulary size, sequence length
LMIN, LMAX = 8, 20       # segment length varies, so position alone cannot solve the task


def make_batch(n, gen):
    """A random segment of length L, tiled to fill T positions (2 to 5 copies).
    The first copy is unpredictable; after it, every next token is predictable
    by "find the earlier occurrence of the current token, copy its successor"."""
    seg = torch.randint(V, (n, T), generator=gen)
    L = torch.randint(LMIN, LMAX + 1, (n,), generator=gen)
    ar = torch.arange(T)
    x = torch.gather(seg, 1, ar[None, :] % L[:, None])
    pred = ar[None, :] >= (L[:, None] - 1)   # positions whose next token repeats
    return x, L, pred


class Attention(nn.Module):
    def __init__(self, d, h):
        super().__init__()
        self.h, self.dh = h, d // h
        self.qkv, self.out = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.head_mask = None    # set to a (h,) tensor of 0/1 to ablate heads
        self.pattern = None      # last attention pattern, (B, h, T, T)

    def forward(self, x):
        B, Tn, d = x.shape
        q, k, v = self.qkv(x).view(B, Tn, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        att = (q @ k.transpose(-1, -2)) / math.sqrt(self.dh)
        causal = torch.ones(Tn, Tn, dtype=torch.bool).tril()
        att = att.masked_fill(~causal, float("-inf")).softmax(-1)
        self.pattern = att.detach()
        y = att @ v                                    # (B, h, T, dh)
        if self.head_mask is not None:
            y = y * self.head_mask.view(1, -1, 1, 1)   # zero-ablate chosen heads
        return self.out(y.transpose(1, 2).reshape(B, Tn, d))


class AttnOnly(nn.Module):
    """Embedding -> n_layer attention layers on the residual stream -> unembed. No MLPs."""

    def __init__(self, n_layer, d=64, h=4):
        super().__init__()
        self.emb, self.pos = nn.Embedding(V, d), nn.Embedding(T, d)
        self.layers = nn.ModuleList(Attention(d, h) for _ in range(n_layer))
        self.lns = nn.ModuleList(nn.LayerNorm(d) for _ in range(n_layer))
        self.ln_f, self.unemb = nn.LayerNorm(d), nn.Linear(d, V)

    def forward(self, x):
        r = self.emb(x) + self.pos(torch.arange(x.shape[1]))
        for ln, att in zip(self.lns, self.layers):
            r = r + att(ln(r))                 # each layer adds to the residual stream
        return self.unemb(self.ln_f(r))


def losses(model, x, pred):
    """Mean next-token loss on induction-predictable positions and on all others."""
    logits = model(x)[:, :-1]
    nll = F.cross_entropy(logits.reshape(-1, V), x[:, 1:].reshape(-1), reduction="none")
    nll = nll.view(x.shape[0], -1)
    m = pred[:, :-1]
    return nll[m].mean().item(), nll[~m].mean().item()


def head_scores(model, x, pred):
    """Per head, two attention scores averaged over query positions.
    induction: attention mass on keys j whose PREVIOUS token equals the query's
               current token (the successors of earlier occurrences), measured
               at positions where the next token is a repeat.
    prev-token: attention mass on the immediately preceding position."""
    model(x)
    Tn = x.shape[1]
    match = torch.zeros(x.shape[0], Tn, Tn, dtype=torch.bool)
    match[:, :, 1:] = x[:, :, None] == x[:, None, :-1]   # x[j-1] == x[i]
    match &= torch.ones(Tn, Tn, dtype=torch.bool).tril()  # keys j <= i only
    out = []
    for att in model.layers:
        P = att.pattern                                   # (B, h, T, T)
        ind = (P * match[:, None]).sum(-1)                # (B, h, T)
        ind = ind.permute(0, 2, 1)[pred].mean(0)          # (h,)
        q = torch.arange(1, Tn)
        prev = P[:, :, q, q - 1].mean((0, 2))             # (h,)
        out.append((ind, prev))
    return out


def train(n_layer, steps, seed, log_every=None):
    torch.manual_seed(seed)
    gen = torch.Generator().manual_seed(seed)
    model = AttnOnly(n_layer)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    xe, Le, pe = make_batch(256, torch.Generator().manual_seed(999))  # fixed eval batch
    history = []
    for step in range(steps + 1):
        if log_every and step % log_every == 0:
            with torch.no_grad():
                rep, other = losses(model, xe, pe)
                ind = max(s[0].max().item() for s in head_scores(model, xe, pe))
            history.append((step, rep, other, ind))
        x, _, _ = make_batch(64, gen)
        logits = model(x)[:, :-1]
        loss = F.cross_entropy(logits.reshape(-1, V), x[:, 1:].reshape(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()
    return model, history, (xe, Le, pe)


if __name__ == "__main__":
    torch.set_num_threads(4)
    t0 = time.time()
    print(f"chance loss = ln({V}) = {math.log(V):.3f} nats\n")

    model, hist, (xe, Le, pe) = train(n_layer=2, steps=1000, seed=0, log_every=50)
    print("2-layer attention-only model during training")
    print(" step  loss(repeat)  loss(other)  best induction score")
    for step, rep, other, ind in hist:
        print(f"{step:5d}  {rep:12.3f}  {other:11.3f}  {ind:20.3f}")

    with torch.no_grad():
        scores = head_scores(model, xe, pe)
        print("\nPer-head scores after training (4 heads per layer)")
        for li, (ind, prev) in enumerate(scores):
            print(f"layer {li}: induction  " + " ".join(f"{v:.2f}" for v in ind))
            print(f"layer {li}: prev-token " + " ".join(f"{v:.2f}" for v in prev))
        prev_heads = scores[0][1].argsort(descending=True).tolist()

        def ablate(layer, heads):
            mask = torch.ones(4)
            mask[heads] = 0
            model.layers[layer].head_mask = mask
            r = losses(model, xe, pe)
            model.layers[layer].head_mask = None
            return r

        # Causal test: zero some heads' outputs and see what breaks.
        print("\nAblations: loss on repeated part / elsewhere")
        tests = [("nothing", 0, []),
                 ("layer 1, best induction head", 1, [int(scores[1][0].argmax())]),
                 ("layer 1, all four heads", 1, [0, 1, 2, 3]),
                 (f"layer 0, best prev-token head ({prev_heads[0]})", 0, prev_heads[:1]),
                 (f"layer 0, second prev-token head ({prev_heads[1]})", 0, prev_heads[1:2]),
                 ("layer 0, the other two heads", 0, prev_heads[2:])]
        for name, layer, heads in tests:
            r = ablate(layer, heads)
            print(f"  {name:38s} {r[0]:.3f} / {r[1]:.3f}")
        # A model that only knew "the next token is one of the segment's L
        # tokens" and spread its bet evenly would score mean(ln L).
        Lq = Le[:, None].expand_as(pe)[pe].float()
        print(f"  reference: uniform over the segment    {Lq.log().mean():.3f}")

    # Control: one attention layer cannot compose a previous-token head with a
    # matching head, so it should not learn induction (Elhage et al., 2021).
    m1, h1, _ = train(n_layer=1, steps=1000, seed=0, log_every=1000)
    print(f"\n1-layer control after 1000 steps: loss(repeat) {h1[-1][1]:.3f}, "
          f"loss(other) {h1[-1][2]:.3f}, best induction score {h1[-1][3]:.3f}")
    print(f"\ntotal time {time.time() - t0:.1f} s")
```

Output:

```text
chance loss = ln(64) = 4.159 nats

2-layer attention-only model during training
 step  loss(repeat)  loss(other)  best induction score
    0         4.335        4.316                 0.072
   50         4.203        4.222                 0.073
  100         4.148        4.174                 0.078
  150         4.088        4.204                 0.088
  200         3.926        4.253                 0.095
  250         3.789        4.247                 0.105
  300         3.648        4.258                 0.124
  350         3.225        4.302                 0.317
  400         1.599        4.442                 0.743
  450         0.916        4.395                 0.841
  500         0.762        4.365                 0.833
  550         0.695        4.321                 0.813
  600         0.559        4.363                 0.827
  650         0.461        4.342                 0.851
  700         0.437        4.292                 0.835
  750         0.414        4.288                 0.841
  800         0.390        4.286                 0.835
  850         0.365        4.299                 0.843
  900         0.348        4.323                 0.839
  950         0.338        4.318                 0.828
 1000         0.338        4.304                 0.833

Per-head scores after training (4 heads per layer)
layer 0: induction  0.07 0.06 0.08 0.07
layer 0: prev-token 0.25 0.49 0.08 0.03
layer 1: induction  0.84 0.82 0.80 0.81
layer 1: prev-token 0.03 0.03 0.03 0.03

Ablations: loss on repeated part / elsewhere
  nothing                                0.336 / 4.309
  layer 1, best induction head           0.603 / 4.277
  layer 1, all four heads                4.692 / 5.017
  layer 0, best prev-token head (1)      3.891 / 4.191
  layer 0, second prev-token head (0)    0.547 / 4.343
  layer 0, the other two heads           0.587 / 4.412
  reference: uniform over the segment    2.577

1-layer control after 1000 steps: loss(repeat) 2.932, loss(other) 4.315, best induction score 0.114

total time 11.1 s
```

Discussed in section 5. The induction score at initialization is already 0.072 because near-uniform attention puts some mass on the matching keys by chance; the jump, not the level, is the signal.

### Script 4: superposition and a sparse autoencoder, `04_superposition_sae.py`

```python
"""Superposition in a toy model, then a sparse autoencoder on data whose true
features we know, so we can check what the SAE recovers instead of guessing.

Part 1 follows the setup of Elhage et al. (2022), "Toy Models of Superposition":
n sparse features, squeezed through m < n hidden dimensions, x' = ReLU(W^T W x + b).
"""
import time

import torch

torch.set_num_threads(4)


# ---------------------------------------------------------------- part 1
def toy_superposition(p_active, n=20, m=5, steps=4000, seed=0):
    """Train x' = ReLU(W^T W x + b) with importance-weighted squared error.
    Each feature is nonzero with probability p_active, uniform on [0, 1]."""
    g = torch.Generator().manual_seed(seed)
    W = (torch.randn(m, n, generator=g) * 0.3).requires_grad_()
    b = torch.zeros(n, requires_grad=True)
    imp = 0.8 ** torch.arange(n)            # feature i matters 0.8^i as much as feature 0
    opt = torch.optim.Adam([W, b], lr=3e-3)
    for _ in range(steps):
        x = torch.rand(1024, n, generator=g) * (torch.rand(1024, n, generator=g) < p_active)
        out = torch.relu(x @ W.T @ W + b)
        loss = (imp * (out - x) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    W = W.detach()
    norms = W.norm(dim=0)                     # ||W_i||: is feature i represented?
    unit = W / norms.clamp_min(1e-9)
    overlap = (unit.T @ W) ** 2               # (W_i_hat . W_j)^2
    interference = overlap.sum(1) - overlap.diagonal()  # sum over j != i
    return norms, interference


print("Part 1: 20 features, 5 hidden dimensions")
print(" P(feature active)  features with |W_i|>0.5  mean interference of those")
for p in [1.0, 0.3, 0.1, 0.03]:
    norms, interf = toy_superposition(p)
    rep = norms > 0.5
    print(f"   {p:5.2f}             {int(rep.sum()):2d}                      {interf[rep].mean():.3f}")


# ---------------------------------------------------------------- part 2
D, N_TRUE, N_LAT = 32, 96, 192   # activation width, true features, SAE latents


def make_data(n_samples, seed):
    """Activations built from 96 known unit directions in 32 dimensions:
    each sample uses about 3 of them with random positive magnitudes."""
    g = torch.Generator().manual_seed(seed)
    dirs = torch.randn(N_TRUE, D, generator=g)
    dirs /= dirs.norm(dim=1, keepdim=True)
    active = torch.rand(n_samples, N_TRUE, generator=g) < 3 / N_TRUE
    coef = active * (0.5 + torch.rand(n_samples, N_TRUE, generator=g))
    return coef @ dirs, dirs, active


def train_sae(acts, l1, seed, steps=3000):
    """ReLU SAE: f = ReLU(W_e (a - b_d) + b_e), a_hat = W_d f + b_d.
    Loss = ||a - a_hat||^2 + l1 * sum_k f_k ||W_d[:, k]||."""
    g = torch.Generator().manual_seed(seed)
    W_d = torch.randn(D, N_LAT, generator=g)
    W_d = (W_d / W_d.norm(dim=0)).requires_grad_()
    W_e = W_d.detach().T.clone().requires_grad_()      # common init: encoder = decoder^T
    b_e = torch.zeros(N_LAT, requires_grad=True)
    b_d = acts.mean(0).clone().requires_grad_()
    opt = torch.optim.Adam([W_d, W_e, b_e, b_d], lr=2e-3)
    for _ in range(steps):
        a = acts[torch.randint(len(acts), (1024,), generator=g)]
        f = torch.relu((a - b_d) @ W_e.T + b_e)
        a_hat = f @ W_d.T + b_d
        loss = ((a - a_hat) ** 2).sum(1).mean() + l1 * (f * W_d.norm(dim=0)).sum(1).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        f = torch.relu((acts - b_d) @ W_e.T + b_e)
        a_hat = f @ W_d.T + b_d
        fvu = ((acts - a_hat) ** 2).sum() / ((acts - acts.mean(0)) ** 2).sum()
        l0 = (f > 0).float().sum(1).mean()
        dead = int(((f > 0).sum(0) == 0).sum())
    return (W_d / W_d.norm(dim=0)).detach(), fvu.item(), l0.item(), dead


t0 = time.time()
acts, dirs, active = make_data(20000, seed=0)
print(f"\nPart 2: {N_TRUE} true features in {D} dimensions, SAE with {N_LAT} latents")
print(f"true L0 (features active per sample): {active.float().sum(1).mean():.2f}")
print("   l1    FVU    L0   dead  true feats matched  mean best cos  latents near a true feat")
for l1 in [0.0, 0.03, 0.1, 0.3, 1.0]:
    dec, fvu, l0, dead = train_sae(acts, l1, seed=0)
    sims = dirs @ dec                           # (true feature, latent) cosines
    best = sims.max(1).values                   # best latent for each true feature
    near = int((sims.max(0).values > 0.9).sum())  # latents that match some feature
    print(f"  {l1:4.2f}  {fvu:5.3f}  {l0:5.1f}  {dead:3d}    {int((best > 0.9).sum()):3d} / {N_TRUE}"
          f"          {best.mean():.3f}        {near:3d} / {N_LAT}")

# PCA can only give 32 orthogonal directions for 96 features.
_, _, Vt = torch.linalg.svd(acts - acts.mean(0), full_matrices=False)
best_pca = (dirs @ Vt.T).abs().max(1).values
print(f"  PCA (32 components): true features matched {int((best_pca > 0.9).sum())} / {N_TRUE}, "
      f"mean best cos {best_pca.mean():.3f}")

# Same data, same l1, different seed: do two SAEs find the same latents?
decA, *_ = train_sae(acts, 0.3, seed=1)
decB, *_ = train_sae(acts, 0.3, seed=2)
cross = (decA.T @ decB).max(1).values
print(f"\nseed 1 vs seed 2 at l1=0.3: {int((cross > 0.9).sum())} / {N_LAT} latents of A "
      f"have a twin in B (cos>0.9)")
print(f"\npart 2 time {time.time() - t0:.1f} s")
```

Output:

```text
Part 1: 20 features, 5 hidden dimensions
 P(feature active)  features with |W_i|>0.5  mean interference of those
    1.00              5                      0.000
    0.30              9                      0.872
    0.10             11                      1.292
    0.03             13                      1.868

Part 2: 96 true features in 32 dimensions, SAE with 192 latents
true L0 (features active per sample): 3.00
   l1    FVU    L0   dead  true feats matched  mean best cos  latents near a true feat
  0.00  0.000   82.1    0      0 / 96          0.488          0 / 192
  0.03  0.005   73.5    0      0 / 96          0.634          0 / 192
  0.10  0.038   43.3    0     21 / 96          0.847         21 / 192
  0.30  0.121   13.8    0     96 / 96          0.985        150 / 192
  1.00  0.320    4.7    0     96 / 96          0.997        133 / 192
  PCA (32 components): true features matched 0 / 96, mean best cos 0.456

seed 1 vs seed 2 at l1=0.3: 153 / 192 latents of A have a twin in B (cos>0.9)

part 2 time 20.6 s
```

Discussed in section 6. No latents died in these runs; on real activations dead latents (never active) are a common problem, and SAE papers use resampling or other tricks to revive them.

## Common mistakes

- **Reporting probe accuracy without baselines.** Always add a random-network baseline, an input baseline and a control task.
- **Reading a probe as evidence of use.** Remove the direction or subspace and measure behavior; if nothing changes, the model may not use it, or may use the information elsewhere.
- **Trusting a saliency map because it looks right.** Run the randomization test. Maps that look like the input may be showing the input.
- **Treating attention weights as importance.** Back any attention-based story with ablation or patching.
- **Single-component ablations only.** Redundant components (script 3's four induction heads) hide each other's importance. Ablate groups, too.
- **Judging an SAE by reconstruction alone.** Dense codes reconstruct perfectly and explain nothing. Report sparsity, reconstruction and some measure of usefulness (a downstream task or known features), against random baselines.
- **Generalizing from toy models to frontier models** without saying so. Toy results show that a mechanism is possible and how to detect it, not that a large model uses it.
- **Calling a result causal** because it came from an intervention, without stating what the intervention assumes (section 4).

## Exercises

1. [warm-up] In script 3, print the attention pattern of the best induction head for one sequence (as a matrix of numbers or an image). Where are the stripes, and why are they offset by one from the earlier copies of each token?
2. [warm-up] In script 2, verify completeness for gradient times input. Does it hold? Why does IG satisfy it and gradient times input not?
3. [core] In script 1, replace the logistic-regression probe with an MLP probe (scikit-learn's `MLPClassifier`). What happens to the random-network row and to selectivity?
4. [core] In script 3, set `LMIN = LMAX = 12`. Does the model still form induction heads, or can it find a cheaper solution? Check the per-head scores and the one-layer control.
5. [core] In script 3, ablate by replacing a head's output with its mean over the evaluation batch rather than zero. Do the conclusions change?
6. [core] In script 4 part 1, change the importance decay from 0.8 to 1.0 (all features equally important). How does the number of represented features change with sparsity?
7. [core] In script 2, implement the data randomization test: train the same CNN on shuffled labels (it will need more epochs to fit them) and compare its maps with the real model's.
8. [core] In script 4 part 2, raise the true number of active features per sample from 3 to 10. How do recovery and the best $\lambda$ change?
9. [stretch] Train an SAE on the layer-0 output (residual stream) of script 3's trained model. Do any latents correspond to "the previous token was $t$"? How would you check?
10. [stretch] Add MLP blocks to script 3's model (or import `GPT` from `code/L2-05-transformers/gpt.py` and change it to expose attention patterns). Does the phase change still happen, and at which step?

### Hints and answers

1. Stripes appear one position after each earlier occurrence of the current token, because the head attends to the *successor* of the matching token; the previous-token head has written "my predecessor was A" there.
2. It does not hold in general: gradient times input is a first-order approximation, exact only for linear functions (and, with a zero baseline, for functions that are positively homogeneous, such as ReLU networks without biases). IG integrates along the path, so the fundamental theorem of calculus gives completeness.
3. An MLP probe gets higher accuracy on every representation, including the random network and the control task, so selectivity does not improve. More expressive probes measure the probe's capacity as much as the representation, which is Hewitt and Liang's point.
4. With a fixed period, a head can attend to position $i - L + 1$ using positional embeddings alone, so the model can solve the task without content-based induction. Look at whether the one-layer control now does much better; if it does, the task no longer requires composition.
5. Mean ablation usually changes the numbers (often less damaging than zero ablation, because zero may be far off-distribution), but the ranking of components should hold. If it does not, that is worth reporting.
6. With equal importance, more features are represented at a given sparsity, because there is no reason to sacrifice the low-importance ones first.
7. A method that passes should produce maps for the random-label model that differ clearly from the real model's maps; expect the gradient to pass and guided backprop to look similar in both.
8. Denser codes make the problem harder: recovery drops, a different $\lambda$ trades off better, and at some density the SAE cannot separate features in 32 dimensions.
9. Collect layer-0 outputs over many sequences, train the SAE, then for each latent find the previous-token value that maximizes its mean activation, and measure how selective it is. Intervening on the latent and watching the induction head's attention is the causal check.
10. Expect the phase change to persist with some shift in timing; Olsson et al. report induction heads in models with MLPs, with the evidence for their role being correlational at scale.

## Check your understanding

1. **A linear probe decodes part of speech from layer 6 with 97% accuracy. What does that show, and what does it not?** It shows that part of speech is linearly decodable from that layer, on that data. Without a control task it does not show the representation encodes part of speech specifically rather than word identity; without a random-network baseline it does not show training created it; and without an intervention it does not show the model uses it.
2. **Why does gradient times input survive the model randomization test so well on digits?** It multiplies by the input, so wherever the input is zero the attribution is zero for every model, and elsewhere its magnitude tracks pixel intensity. Much of the map is the input. Restricting the comparison to nonzero pixels shows a smaller but real dependence on the weights.
3. **Why can a one-layer attention-only transformer not implement an induction head?** The head must attend to the position *after* an earlier copy of the current token, so its keys must contain information about each position's predecessor. In one layer, keys see only the token and position at that key position. A previous-token head in an earlier layer must first write the predecessor's identity into the residual stream: composition.
4. **What is superposition, and why does sparsity make it possible?** Representing more features than dimensions as non-orthogonal directions. If features are rarely active together, the interference between their overlapping directions seldom occurs, and a nonlinearity such as ReLU can filter out the small interference that does, so the benefit of representing extra features outweighs the cost.
5. **An SAE reconstructs activations with 95% variance explained and its features look interpretable to an automated rater. What else would you want before believing it found the model's features?** A comparison with SAEs trained on a randomly initialized model and other random baselines; stability across seeds; evidence that features matter causally (intervening on them changes behavior as predicted); and usefulness on a downstream task compared with simpler methods such as linear probes.

## Further reading

- **Cynthia Rudin, "Stop Explaining Black Box Machine Learning Models for High Stakes Decisions and Use Interpretable Models Instead"** (2019, Nature Machine Intelligence 1:206-215; arXiv:1811.10154). https://arxiv.org/abs/1811.10154
  *Read for:* the case against post-hoc explanation in high-stakes settings. Open version on arXiv. *Level:* intro.
- **Yonatan Belinkov, "Probing Classifiers: Promises, Shortcomings, and Advances"** (2022, Computational Linguistics 48(1); arXiv:2102.12452). https://arxiv.org/abs/2102.12452
  *Read for:* a short, balanced review of what probing can and cannot show. *Level:* intro.
- **John Hewitt, Percy Liang, "Designing and Interpreting Probes with Control Tasks"** (2019, EMNLP; arXiv:1909.03368). https://aclanthology.org/D19-1275/
  *Read for:* control tasks and selectivity, the baseline every probing study needs. *Level:* intermediate.
- **Julius Adebayo, Justin Gilmer, Michael Muelly, Ian Goodfellow, Moritz Hardt, Been Kim, "Sanity Checks for Saliency Maps"** (2018, NeurIPS; arXiv:1810.03292). https://arxiv.org/abs/1810.03292
  *Read for:* the randomization tests and which popular saliency methods fail them. *Level:* intro.
- **Mukund Sundararajan, Ankur Taly, Qiqi Yan, "Axiomatic Attribution for Deep Networks"** (2017, ICML; arXiv:1703.01365). https://arxiv.org/abs/1703.01365
  *Read for:* integrated gradients and the axiomatic approach to attribution. *Level:* intermediate.
- **Chris Olah, Nick Cammarata, Ludwig Schubert, Gabriel Goh, Michael Petrov, Shan Carter, "Zoom In: An Introduction to Circuits"** (2020, Distill). https://distill.pub/2020/circuits/zoom-in/
  *Read for:* the features-and-circuits program and its three claims, with vision examples. *Level:* intro.
- **Nelson Elhage, Neel Nanda, Catherine Olsson, et al. (25 authors), "A Mathematical Framework for Transformer Circuits"** (2021, Transformer Circuits Thread). https://transformer-circuits.pub/2021/framework/index.html
  *Read for:* the residual-stream view, zero-, one- and two-layer attention-only models, and composition. *Level:* advanced.
- **Catherine Olsson, Nelson Elhage, Neel Nanda, et al. (26 authors), "In-context Learning and Induction Heads"** (2022, arXiv:2209.11895). https://arxiv.org/abs/2209.11895
  *Read for:* the phase change, the six lines of evidence, and how strong each is at different scales. *Level:* intermediate.
- **Nelson Elhage, Tristan Hume, Catherine Olsson, et al. (16 authors), "Toy Models of Superposition"** (2022, arXiv:2209.10652). https://transformer-circuits.pub/2022/toy_model/index.html
  *Read for:* when and how networks store more features than dimensions. *Level:* intermediate.
- **Trenton Bricken, Adly Templeton, Joshua Batson, et al. (24 authors), "Towards Monosemanticity: Decomposing Language Models With Dictionary Learning"** (2023, Transformer Circuits Thread), and **Adly Templeton, Tom Conerly, Jonathan Marcus, et al. (22 authors), "Scaling Monosemanticity: Extracting Interpretable Features from Claude 3 Sonnet"** (2024). https://transformer-circuits.pub/2023/monosemantic-features/index.html and https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html
  *Read for:* sparse autoencoders from a one-layer model to a production model, with many worked features. *Level:* intermediate.
- **Kevin Wang, Alexandre Variengien, Arthur Conmy, Buck Shlegeris, Jacob Steinhardt, "Interpretability in the Wild: a Circuit for Indirect Object Identification in GPT-2 small"** (2023, ICLR; arXiv:2211.00593). https://arxiv.org/abs/2211.00593
  *Read for:* a full circuit analysis in a real language model, with criteria for evaluating it. *Level:* advanced.
- **Lee Sharkey, Bilal Chughtai, et al. (29 authors), "Open Problems in Mechanistic Interpretability"** (2025, arXiv:2501.16496). https://arxiv.org/abs/2501.16496
  *Read for:* the field's own map of what is unsolved; read with the critiques in section 7, especially Kantamneni et al. ([arXiv:2502.16681](https://arxiv.org/abs/2502.16681)) and Paulo and Belrose ([arXiv:2501.16615](https://arxiv.org/abs/2501.16615)). *Level:* advanced.
