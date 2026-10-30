---
title: "Generative Models and Diffusion"
---

> **Level 3 · Module 3** · Prerequisites: L0-04, L0-05, L2-02, L2-05, L2-06 · Time: 8 to 10 hours · Code: `code/L3-03-generative-models-and-diffusion/`

## What you will be able to do

- State what a generative model is for, and compare VAEs, GANs, diffusion and flow models by what they optimize and how they fail.
- Derive the DDPM training loss from the forward noising process and connect noise prediction to the score $\nabla_x \log q_t(x)$.
- Implement and train a diffusion model and a flow-matching model on CPU, and sample from them with DDPM, DDIM and Euler steps.
- Explain classifier-free guidance as a reweighting of scores, and measure the trade-off it makes between fidelity and diversity.
- Explain why modern image and video systems run diffusion or flow matching in a learned latent space with a transformer backbone.
- Compute a Fréchet distance and judge FID claims: what it measures, how it can be gamed, and what it cannot see.

## 1. What a generative model is for

A **generative model** learns a distribution $p_\theta(x)$ close to the data distribution $p_{\text{data}}(x)$, from samples alone. We may want **samples** (new images, audio, molecules), **densities** (evaluate $p_\theta(x)$ for compression or anomaly detection) or **conditional samples** (given a label, a prompt or a partial input, which is what most products need). No single family is best at all three. Autoregressive models factor $p(x) = \prod_i p(x_i \mid x_{<i})$ and give exact likelihoods; that is the language-model story of L2-05 and L3-01. This module covers the other main families:

| family | what it learns | likelihood | typical failure |
|---|---|---|---|
| VAE | encoder and decoder, trained on a lower bound | lower bound only | blurry or off-manifold samples |
| GAN | generator trained against a discriminator | none | unstable training, dropped modes |
| diffusion | denoiser at every noise level | bound, or exact via an ODE | slow sampling (many network calls) |
| flow matching | velocity field from noise to data | exact via the ODE | same as diffusion, often fewer steps |

A model can produce perfect samples from half the modes of the data, or cover every mode with blurry samples, so you must say which property you care about. Throughout this module the toy data is a mixture of 8 Gaussians on a circle, so we can count modes and measure exactly what goes wrong. The shared helper `toy2d.py` (in the Hands-on section) defines four measurements: **in_mode** (fraction of samples within $3\sigma$ of a mode centre; 0.989 for real data), **modes** (how many of the 8 modes receive at least a quarter of their fair share), **mode_kl** (imbalance between modes, 0 if balanced) and **spread** (width of the samples within a mode, 1.00 for real data).

## 2. Variational autoencoders

L2-06 trained autoencoders whose codes were useful but which could not generate: there was no distribution to sample codes from. A **variational autoencoder (VAE)** fixes this with a latent-variable model: draw $z \sim p(z) = \mathcal{N}(0, I)$, then $x \sim p_\theta(x \mid z)$ from a decoder network. The likelihood $p_\theta(x) = \int p_\theta(x \mid z) p(z)\, dz$ is intractable, so an encoder $q_\phi(z \mid x)$ approximates the posterior, and we maximize the **evidence lower bound (ELBO)** [Kingma and Welling, 2014]:

$$\log p_\theta(x) = \underbrace{\mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x \mid z)] - \mathrm{KL}\big(q_\phi(z \mid x)\,\|\,p(z)\big)}_{\text{ELBO}} + \mathrm{KL}\big(q_\phi(z \mid x)\,\|\,p_\theta(z \mid x)\big).$$

The last term is a KL divergence (L0-04), so it is never negative and the ELBO is a lower bound. The first term rewards reconstruction; the second keeps each code distribution close to the prior so that prior samples decode to something sensible. With a Gaussian encoder, $z = \mu_\phi(x) + \sigma_\phi(x) \odot \epsilon$ with $\epsilon \sim \mathcal{N}(0, I)$, so gradients flow through $\mu$ and $\sigma$: this is the **reparameterization trick** that made the method practical. With a Gaussian decoder of fixed standard deviation $\sigma_x$, the reconstruction term is $-\lVert x - g_\theta(z) \rVert^2 / (2\sigma_x^2)$ plus a constant, so $\sigma_x$ is really a weight on reconstruction against the KL term.

Script 04 trains VAEs with a 2D latent on the 8-mode data. With $\sigma_x = 0.05$ reconstructions are nearly perfect (RMSE 0.029) but only 16% of prior samples land near a mode: the codes no longer look like $\mathcal{N}(0, I)$ in aggregate, and prior samples fall in "holes" that decode to points between modes. With $\sigma_x = 1$ the KL term wins, codes carry 0.84 nats, and samples collapse toward the middle (7% near a mode). The best setting, $\sigma_x = 0.3$, reaches 54%: a continuous decoder must map a connected Gaussian onto 8 separated blobs, so the regions in between get decoded too. On images the same tension shows up as blur. VAEs remain important as **compressors**: latent diffusion (section 7) runs inside a VAE-style latent space.

## 3. GANs, and why they were hard to train

A **generative adversarial network** trains a generator $G$ that maps noise $z$ to samples and a discriminator $D$ that outputs the probability that its input is real, in the minimax game [Goodfellow et al., 2014]

$$\min_G \max_D \; \mathbb{E}_{x \sim p_{\text{data}}}[\log D(x)] + \mathbb{E}_{z}[\log(1 - D(G(z)))].$$

For a fixed $G$ the best discriminator is $D^*(x) = p_{\text{data}}(x) / (p_{\text{data}}(x) + p_G(x))$, and with it the objective equals $-\log 4 + 2\,\mathrm{JSD}(p_{\text{data}} \,\|\, p_G)$, so the global optimum is $p_G = p_{\text{data}}$. Goodfellow et al. also noted that early in training $\log(1 - D(G(z)))$ saturates, and proposed training $G$ to maximize $\log D(G(z))$ instead (the "non-saturating" loss used in script 04).

GANs were the leading image generators for several years because samples are sharp: nothing in the loss rewards averaging. They were also notoriously hard to train, for structural reasons:

- **It is a game, not a minimization.** Each player's loss changes whenever the other updates. Gradient descent on a game can cycle instead of converging, and the losses printed during training do not tell you whether samples are improving.
- **Mode collapse.** The generator is rewarded for samples the current discriminator accepts, not for covering the data. Mapping many $z$ to a few convincing outputs is a cheap win until the discriminator catches up, and then the generator can hop to other modes.
- **Weak gradients.** When real and generated distributions barely overlap, a good discriminator separates them almost perfectly and the generator's gradient carries little information. Arjovsky, Chintala and Bottou's Wasserstein GAN (2017, arXiv:1701.07875) replaced the divergence for this reason and claimed better stability and less mode collapse.

Script 04 trains the same small GAN with three seeds. All three cover 8/8 modes with blurry samples (in_mode about 0.36) for the first 1,000 steps; by step 3,000 they cover 3, 6 and 3 modes, while in_mode rises to 0.60 to 0.78. The generator trades coverage for sharpness, the ending depends on the seed, and a single "sample quality" number would report this as progress. Taming it took a long list of architectures, losses and regularizers; diffusion models sidestepped the game with a plain regression loss.

## 4. Diffusion: noising is easy, so learn to undo it

**The forward process.** Fix a schedule $\beta_1, \dots, \beta_T$ and add a little Gaussian noise at each step:

$$q(x_t \mid x_{t-1}) = \mathcal{N}\big(\sqrt{1 - \beta_t}\, x_{t-1},\; \beta_t I\big).$$

With $\alpha_t = 1 - \beta_t$ and $\bar\alpha_t = \prod_{s \le t} \alpha_s$, composing Gaussians gives a closed form for any $t$:

$$x_t = \sqrt{\bar\alpha_t}\, x_0 + \sqrt{1 - \bar\alpha_t}\, \epsilon, \qquad \epsilon \sim \mathcal{N}(0, I).$$

DDPM used $T = 1000$ with $\beta_t$ rising linearly from $10^{-4}$ to $0.02$ [Ho et al., 2020]. Script 01 prints the **signal-to-noise ratio** $\mathrm{SNR}(t) = \bar\alpha_t / (1 - \bar\alpha_t)$: 8.7 at $t = 100$, 1.1 at $t = 250$, 0.085 at $t = 500$ and $4 \times 10^{-5}$ at $t = 1000$. By step 1000, $x_T$ is standard normal (variances 1.005 and 1.004), and running the chain step by step agrees with the closed form. The schedule spends its steps unevenly: after $t \approx 500$ the SNR is below 0.1, so half the steps act on near-pure noise, which matters for sampling (section 6).

**The reverse process.** If we could sample $q(x_{t-1} \mid x_t)$ we could walk from noise back to data. That distribution depends on the whole data distribution, but conditioned on $x_0$ it is a Gaussian with mean

$$\tilde\mu_t(x_t, x_0) = \frac{1}{\sqrt{\alpha_t}}\Big(x_t - \frac{\beta_t}{\sqrt{1 - \bar\alpha_t}}\, \epsilon\Big), \qquad \text{where } x_t = \sqrt{\bar\alpha_t} x_0 + \sqrt{1 - \bar\alpha_t}\,\epsilon.$$

So a network that predicts the noise $\epsilon$ from $(x_t, t)$ is enough to define the reverse steps. Ho et al. derived the ELBO for this hierarchical latent-variable model and found that a simplified, reweighted version works better in practice:

$$L_{\text{simple}} = \mathbb{E}_{x_0, t, \epsilon}\, \big\lVert \epsilon - \epsilon_\theta\big(\sqrt{\bar\alpha_t} x_0 + \sqrt{1 - \bar\alpha_t}\,\epsilon,\; t\big) \big\rVert^2.$$

Training is supervised regression with free labels: corrupt a data point with known noise, predict the noise. No adversary, no posterior to approximate. Sampling (DDPM "ancestral" sampling) starts at $x_T \sim \mathcal{N}(0, I)$ and repeats $x_{t-1} = \tilde\mu_t(x_t, \hat x_0) + \sigma_t z$ with the predicted noise, here with $\sigma_t^2 = \beta_t$; Ho et al. found $\beta_t$ and the posterior variance $\tilde\beta_t$ gave similar results. They reported an FID of 3.17 on unconditional CIFAR-10 and were candid that their log-likelihoods (3.70 bits per dimension) were not competitive with other likelihood-based models; most bits go to imperceptible detail. For a slower derivation, see [Weng, 2021] and [Luo, 2022].

## 5. The score view

The **score** of a distribution is $\nabla_x \log p(x)$, a vector field pointing toward higher density. For Gaussian corruption, Tweedie's formula links the best denoiser to the score of the noisy distribution $q_t$ [Luo, 2022]:

$$\mathbb{E}[x_0 \mid x_t] = \frac{x_t + (1 - \bar\alpha_t)\,\nabla_{x_t} \log q_t(x_t)}{\sqrt{\bar\alpha_t}} \quad\Longrightarrow\quad \epsilon^*(x_t, t) = \mathbb{E}[\epsilon \mid x_t] = -\sqrt{1 - \bar\alpha_t}\; \nabla_{x_t} \log q_t(x_t).$$

The minimizer of $L_{\text{simple}}$ is that conditional mean, so **a noise predictor is a scaled score estimator**. Ho et al. pointed out that their objective resembles denoising score matching over many noise levels. Song et al. made the picture continuous [Song et al., 2021]: the forward process becomes a stochastic differential equation, its time reversal needs only the score, and there is a deterministic **probability flow ODE** with the same marginals at every $t$. The ODE gives exact likelihoods and allows fewer, larger steps with standard solvers. "Diffusion" and "score-based" models are therefore the same model seen from two sides.

Our toy makes this concrete. For a Gaussian mixture, $q_t$ is again a Gaussian mixture, so $\epsilon^*$ has a closed form (`exact_eps` in `toy2d.py`), which separates sampler error from learning error. DDPM sampling with the exact $\epsilon^*$ reproduces the data (in_mode 0.986, against 0.989 for real data). The MLP trained for 3,000 steps reaches 0.950, and script 01 shows where it loses: its squared error against $\epsilon^*$ is 0.035 at $t = 100$ and below 0.006 elsewhere. Around $t = 100$ the noise is just small enough for the modes to separate, and the network must decide which mode a point belongs to; that sharp transition is the hardest part of the field to fit.

> **Note:** The exact denoiser for a *finite* training set (a mixture of point masses) is also closed-form, and sampling with it reproduces training points: in script 05 the median distance from such a sample to its nearest training point is 0.0025, against 0.022 for fresh data. A perfectly minimized diffusion loss memorizes; trained networks generalize partly *because* they miss that optimum. Stein et al. report memorization on small, simple datasets like CIFAR-10 but not necessarily on ImageNet [Stein et al., 2023]. Why networks generalize here is an open research question.

## 6. Samplers: trading network calls for quality

DDPM needs one network evaluation per step, 1000 per sample. **DDIM** (Song, Meng and Ermon, ICLR 2021, arXiv:2010.02502) keeps the same trained model but uses a deterministic, non-Markovian update that can skip steps; they reported 10 to 50 times faster sampling. With $\hat x_0 = (x_t - \sqrt{1 - \bar\alpha_t}\,\hat\epsilon)/\sqrt{\bar\alpha_t}$, one DDIM step to an earlier time $s$ is

$$x_s = \sqrt{\bar\alpha_s}\,\hat x_0 + \sqrt{1 - \bar\alpha_s}\,\hat\epsilon,$$

which is a discretization of the probability flow ODE. In script 01, DDIM with the learned network scores in_mode 0.903, 0.840, 0.693 and 0.117 at 100, 20, 10 and 5 steps (DDPM with 1000 steps: 0.950). With the *exact* score it scores 0.974, 0.773 and 0.151 at 20, 10 and 5 steps. So the collapse at 5 steps is discretization error of the solver, not a modelling error, and evenly spaced steps on this schedule waste most of the budget at near-zero SNR (section 8 tests this). Faster sampling since then has come from better ODE solvers, better step placement and **distillation**, which trains a student to reproduce many teacher steps in a few (progressive distillation, consistency models; see [Weng, 2021]).

## 7. Conditioning and classifier-free guidance

A conditional model $\epsilon_\theta(x_t, t, c)$ takes the condition $c$ (a class, a text embedding) as an extra input. Its samples tend to follow the condition loosely, so samplers apply **guidance**. **Classifier-free guidance** trains one network both with $c$ and with a null token $\varnothing$ (dropping the label with probability $p_{\text{uncond}}$; Ho and Salimans found 0.1 and 0.2 about equally good and 0.5 worse) and, at each sampling step, uses [Ho and Salimans, 2022]

$$\tilde\epsilon = (1 + w)\,\epsilon_\theta(x_t, t, c) - w\,\epsilon_\theta(x_t, t, \varnothing).$$

Since $\epsilon \propto -\nabla \log q_t$, the guided score is

$$\nabla \log q_t(x \mid c) + w\big(\nabla \log q_t(x \mid c) - \nabla \log q_t(x)\big) = \nabla \log q_t(x \mid c) + w\,\nabla \log q_t(c \mid x),$$

using Bayes' rule. Guidance pushes samples toward regions where an *implicit classifier* is confident about $c$, roughly sampling from $p(x \mid c)\,p(c \mid x)^w$. It is not exactly that distribution (the noised version of a product is not the product of noised versions), and Ho and Salimans note the guided score is not the gradient of any classifier. They report a clean trade-off: small $w$ (0.1 or 0.3) gave the best FID, strong guidance ($w \ge 4$) the best Inception score.

Script 02 conditions on the parity of the mode (class 0 is the four even modes) and samples class 0. Almost every sample lands in the right class at any $w$ (0.98 to 1.00). Mild guidance sharpens samples (in_mode 0.922 at $w = 0$, 0.959 at $w = 0.5$) and narrows them (spread 1.17 to 1.04). Strong guidance overshoots: in_mode falls to 0.729 at $w = 3$ and 0.220 at $w = 10$, where the median sample sits at radius 3.84 instead of 2: the toy analogue of the oversaturated, over-contrasted images you get with very high guidance scales. Part of the reason is in the first output line: the unconditional branch, trained on 10% of examples, is poor (in_mode 0.406), and guidance amplifies the *difference* between two imperfect predictions.

**Latent diffusion.** Pixel-space diffusion at high resolution is expensive. Rombach et al. first train an autoencoder that compresses images by a factor $f$ per side, then run diffusion in its latent space and decode at the end [Rombach et al., 2022]. They studied $f$ from 1 to 32 and found 4 to 16 a good balance between cost and fidelity, used a lightly KL-regularized (VAE-like) autoencoder for text-to-image, and conditioned on text through cross-attention layers inside the denoiser. The official CompVis repository describes Stable Diffusion v1 as such a latent text-to-image diffusion model, with a factor-8 autoencoder, an 860M-parameter UNet and a frozen CLIP ViT-L/14 text encoder (CLIP is covered in L3-04). All the publicly documented systems in section 9 also work in a learned latent space.

## 8. Flow matching

Flow matching starts from the ODE view instead of a noising chain [Lipman et al., 2023]. Choose a path from noise $z \sim \mathcal{N}(0, I)$ to a data point $x_1$; the simplest is a straight line,

$$x_t = (1 - t)\, z + t\, x_1, \qquad \frac{dx_t}{dt} = x_1 - z, \qquad t \in [0, 1].$$

Train a network $v_\theta(x, t)$ by regression onto the velocity of these per-sample paths:

$$L_{\text{FM}} = \mathbb{E}_{t, z, x_1}\, \big\lVert v_\theta(x_t, t) - (x_1 - z) \big\rVert^2.$$

The minimizer is $v^*(x, t) = \mathbb{E}[x_1 - z \mid x_t = x]$, and Lipman et al. show that integrating $dx/dt = v^*(x, t)$ from $t = 0$ carries the noise distribution to the data distribution. Training needs no simulation. Diffusion paths are a special case of their Gaussian paths, and the straight line is the path of **rectified flow** (Liu, Gong and Liu, 2022, arXiv:2209.03003). The loss cannot reach zero (script 03 ends at 1.96): many $(z, x_1)$ pairs pass through the same $x_t$ and the network learns their average direction, so the per-sample paths are straight but the learned marginal paths are not.

Script 03 trains both models with the same MLP and budget. With flow matching and Euler steps, in_mode is 0.776, 0.891 and 0.920 at 5, 10 and 20 network calls; DDIM on evenly spaced steps gets 0.117, 0.689 and 0.844. But the linear path has $\mathrm{SNR}(t) = t^2/(1 - t)^2$, and running DDIM on the DDPM steps with those same SNRs closes most of the gap at 5 steps (0.117 to 0.674), although flow matching stays ahead at 10 and 20 steps. So much of the few-step advantage in this toy is *where the steps sit in noise level*, not straight paths as such; the rest comes from parameterization and how well each network fit. Esser et al. likewise found that the distribution of training noise levels mattered, and improved rectified flow by sampling times that favour perceptually relevant scales [Esser et al., 2024]. When a paper credits a speed-up to a new formulation, check whether the baseline got the same step schedule.

## 9. What large systems do (as of October 2026)

Publicly documented open models in 2024 to 2026 converge on one recipe: **a transformer denoiser, trained with flow matching or a diffusion objective, in the latent space of a learned autoencoder, conditioned on text embeddings**.

- Stable Diffusion 3's paper reports that rectified flow with their improved time sampling outperformed established diffusion formulations in a large study, introduces a transformer with separate weights for image and text tokens, and reports predictable scaling, with lower validation loss tracking better text-to-image results [Esser et al., 2024].
- The Wan video models (Wan team, 2025, arXiv:2503.20314) describe a diffusion-transformer built on flow matching, a video autoencoder that compresses 4× in time and 8× in each spatial dimension, and open weights at 1.3B and 14B parameters.
- The Z-Image report (arXiv:2511.22699, revised July 2026) describes a 6B-parameter single-stream diffusion transformer trained for about 314,000 H800 GPU hours (about 630,000 US dollars), by the authors' own accounting.

Closed commercial systems document far less; architecture statements about them come from blog posts or system cards and should be treated as claims. Generating images with the same autoregressive token models used for text (early-fusion multimodal models) is a competing approach, covered in L3-04.

## 10. Evaluation: FID and its limits

**FID** embeds real and generated images with an Inception-v3 network trained on ImageNet, fits a Gaussian to each set of features, and reports the Fréchet distance between the two Gaussians [Heusel et al., 2017]:

$$\mathrm{FID} = \lVert \mu_r - \mu_g \rVert^2 + \mathrm{Tr}\big(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2}\big).$$

Heusel et al. introduced it as capturing similarity to real images better than the earlier Inception score, and it became the standard number in papers. Script 05 computes the same formula on the toy, both on raw 2D coordinates and on 64 fixed random features (a stand-in for the feature network), for 2,000 samples against 10,000 real test points:

| samples | FD, raw 2D | FD, features | in_mode | modes |
|---|---|---|---|---|
| fresh real data | 0.0029 | 0.084 | 0.985 | 8/8 |
| Gaussian with matched mean and covariance | 0.0073 | 24.9 | 0.071 | 0/8 |
| one mode dropped | 0.1142 | 3.69 | 0.988 | 7/8 |
| modes 2× too wide | 0.0021 | 4.13 | 0.652 | 8/8 |
| modes 2× too narrow | 0.0014 | 1.51 | 1.000 | 8/8 |
| copies of 500 training points | 0.0336 | 0.52 | 0.983 | 8/8 |

What this shows, and what the literature confirms on real images:

- **It sees only two moments of the features.** In raw coordinates the wrong-width mixtures score *better than real data*, and a single Gaussian with no modes scores 0.007. Everything depends on the feature extractor. Kynkäänniemi et al. (ICLR 2023, arXiv:2203.06026) showed that Inception features are so tied to ImageNet classes that matching the distribution of top class predictions lowers FID substantially without improving images, so FID can be gamed deliberately or by accident.
- **It does not detect memorization.** On features, copying training points beats every flawed generator, and nothing flags it. Stein et al. found that no metric they examined separates memorization from underfitting or mode shrinkage [Stein et al., 2023].
- **It is biased by sample size.** Two independent sets of *real* data give FD 6.2 at $n = 50$, 1.37 at $n = 200$, 0.24 at $n = 1{,}000$ and 0.052 at $n = 5{,}000$ (script 05). Scores are comparable only with the same $n$, the same feature network and the same preprocessing.
- **It disagrees with people.** In a large human study, Stein et al. found that no existing metric correlated strongly with human judgments of realism. Humans rated diffusion models most realistic and FID did not reflect it; they blame Inception-v3 features in part and recommend self-supervised DINOv2 features instead.

Report FID to compare with prior work, but pair it with separate fidelity and coverage measures (precision and recall style), nearest-neighbour checks against training data, task-specific measures, and a human evaluation with a stated protocol.

## Hands-on

Run each script from the repository root, for example `.venv/bin/python code/L3-03-generative-models-and-diffusion/01_ddpm_2d.py`; the scripts import `toy2d.py`, which Python finds because it sits next to them. Outputs below come from Python 3.12, PyTorch 2.15.0 (CPU, 4 threads), NumPy 2.5.4 and SciPy 1.18.1 on Linux (WSL2), on a machine shared with other jobs. Wall times were 5 to 48 seconds per script (04 is the slowest); printed timings vary with load, but seeded results reproduced exactly across repeated runs.

### Shared helpers, `toy2d.py` (imported by the scripts; prints nothing)

```python
"""Shared pieces for the 2D generative-model toys (imported by the scripts; prints nothing).

Data: a mixture of 8 Gaussians on a circle of radius 2, each with std SIGMA.
Because the modes are known, we can measure what a sample-quality number hides:
  in_mode  fraction of samples within 3 SIGMA of some mode centre (real data: 0.989)
  modes    how many of the 8 modes get at least a quarter of their fair share
  mode_kl  KL(mode histogram || uniform) in nats; 0 means perfectly balanced
  spread   RMS distance to the mode centre for in-mode samples, relative to real data
"""
import math
import torch
import torch.nn as nn

# These models are tiny; beyond a few threads PyTorch spends more time coordinating than computing.
torch.set_num_threads(4)

SIGMA = 0.1
K = 8
CENTERS = 2.0 * torch.stack([torch.cos(torch.arange(K) * 2 * math.pi / K),
                             torch.sin(torch.arange(K) * 2 * math.pi / K)], dim=1)
# E[r^2 | r < 3 sigma] / sigma^2 for a 2D Gaussian, so real data has spread 1.00.
_U_TRUNC = 2 - 9 * math.exp(-4.5) / (1 - math.exp(-4.5))


def sample_data(n, gen=None, return_labels=False):
    k = torch.randint(0, K, (n,), generator=gen)
    x = CENTERS[k] + SIGMA * torch.randn(n, 2, generator=gen)
    return (x, k) if return_labels else x


def evaluate(x):
    d = torch.cdist(x, CENTERS)                      # (n, 8) distances to centres
    dmin, k = d.min(dim=1)
    ok = dmin < 3 * SIGMA
    counts = torch.bincount(k[ok], minlength=K).float()
    p = counts / counts.sum().clamp(min=1)
    mode_kl = (p[p > 0] * (p[p > 0] * K).log()).sum().item()
    spread = (dmin[ok].pow(2).mean() / (SIGMA**2 * _U_TRUNC)).sqrt().item() if ok.any() else float("nan")
    return dict(in_mode=ok.float().mean().item(),
                modes=int((counts >= len(x) / (4 * K)).sum()),
                mode_kl=mode_kl, spread=spread)


def fmt(m):
    return f"in_mode {m['in_mode']:.3f}  modes {m['modes']}/8  mode_kl {m['mode_kl']:.3f}  spread {m['spread']:.2f}"


# ---- Noise schedule (DDPM: T = 1000, beta linear from 1e-4 to 0.02) -------------
T = 1000
BETAS = torch.linspace(1e-4, 0.02, T)
ALPHAS = 1 - BETAS
ABAR = torch.cumprod(ALPHAS, dim=0)                  # ABAR[i] is alpha-bar at t = i + 1


def time_features(t01, n_freq=16):
    """Sinusoidal features of a time in [0, 1], so the MLP can tell noise levels apart."""
    f = torch.exp(torch.linspace(0, math.log(200.0), n_freq))
    a = t01[:, None] * f[None, :]
    return torch.cat([a.sin(), a.cos()], dim=1)


class Net(nn.Module):
    """MLP that maps (x_t, t, optional class) to a 2D output: noise or velocity."""

    def __init__(self, n_classes=0, hidden=128):
        super().__init__()
        self.emb = nn.Embedding(n_classes + 1, 16) if n_classes else None   # last index = "no class"
        d_in = 2 + 32 + (16 if n_classes else 0)
        self.mlp = nn.Sequential(nn.Linear(d_in, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, hidden), nn.SiLU(),
                                 nn.Linear(hidden, 2))

    def forward(self, x, t01, c=None):
        h = [x, time_features(t01)]
        if self.emb is not None:
            h.append(self.emb(c))
        return self.mlp(torch.cat(h, dim=1))


def exact_eps(x, i, centers=CENTERS, sigma0=SIGMA):
    """E[noise | x_t] for Gaussian-mixture data with equal weights, in closed form.

    q_t is again a mixture: components N(a * mu_k, v I) with a = sqrt(abar), v = a^2 sigma0^2 + 1 - abar.
    Its score is sum_k r_k(x) (a mu_k - x) / v, and E[eps | x_t] = -sqrt(1 - abar) * score.
    With sigma0 = 0 and centers = training points this is the optimum for an empirical dataset.
    """
    a, s2 = ABAR[i].sqrt(), 1 - ABAR[i]
    v = a**2 * sigma0**2 + s2
    logits = -torch.cdist(x, a * centers).pow(2) / (2 * v)
    r = torch.softmax(logits, dim=1)                 # responsibilities r_k(x)
    score = (r @ (a * centers) - x) / v
    return -s2.sqrt() * score


@torch.no_grad()
def ddpm_sample(eps_fn, n, gen):
    """Ancestral sampling, T steps, sigma_t^2 = beta_t."""
    x = torch.randn(n, 2, generator=gen)
    for i in reversed(range(T)):
        eps = eps_fn(x, i)
        x = (x - BETAS[i] / (1 - ABAR[i]).sqrt() * eps) / ALPHAS[i].sqrt()
        if i > 0:
            x = x + BETAS[i].sqrt() * torch.randn(n, 2, generator=gen)
    return x


@torch.no_grad()
def ddim_sample(eps_fn, n, gen, steps, idx=None):
    """Deterministic DDIM (eta = 0) on an evenly spaced subset of the T noise levels,
    or on the noise levels given by the decreasing index list idx."""
    idx = idx or torch.linspace(T - 1, 0, steps).round().long().tolist()
    x = torch.randn(n, 2, generator=gen)
    for j, i in enumerate(idx):
        eps = eps_fn(x, i)
        x0_hat = (x - (1 - ABAR[i]).sqrt() * eps) / ABAR[i].sqrt()
        abar_prev = ABAR[idx[j + 1]] if j + 1 < len(idx) else torch.tensor(1.0)
        x = abar_prev.sqrt() * x0_hat + (1 - abar_prev).sqrt() * eps
    return x


def train_eps_model(steps=3000, batch=1024, n_classes=0, p_uncond=0.1, seed=0, label_fn=None):
    """Train the noise predictor with the DDPM 'simple' loss ||eps - eps_theta(x_t, t)||^2."""
    torch.manual_seed(seed)
    net = Net(n_classes)
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(steps):
        x0, c = sample_data(batch, return_labels=True)
        i = torch.randint(0, T, (batch,))
        eps = torch.randn_like(x0)
        xt = ABAR[i].sqrt()[:, None] * x0 + (1 - ABAR[i]).sqrt()[:, None] * eps
        if n_classes:   # classifier-free guidance training: sometimes hide the label
            c = label_fn(c) if label_fn else c
            c = torch.where(torch.rand(batch) < p_uncond, torch.full_like(c, n_classes), c)
            pred = net(xt, i.float() / T, c)
        else:
            pred = net(xt, i.float() / T)
        loss = (pred - eps).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    return net, loss.item()
```

### Script 01: DDPM, the exact score, and DDIM

```python
"""DDPM on a 2D mixture of 8 Gaussians, trained on CPU in seconds.

1. The forward process: closed form versus running the noising chain.
2. A sampler check with the exact (analytic) noise predictor, before any learning.
3. Train an MLP noise predictor; compare it with the exact one.
4. Sample with DDPM (1000 stochastic steps) and DDIM (deterministic, fewer steps).
"""
import time
import torch
from toy2d import (ABAR, ALPHAS, T, evaluate, exact_eps, fmt, sample_data,
                   ddpm_sample, ddim_sample, train_eps_model)

torch.manual_seed(0)
gen = torch.Generator().manual_seed(1)
N = 10000

# ---- 1. Forward process ------------------------------------------------------
print("alpha_bar and signal-to-noise ratio along the schedule:")
for t in (1, 100, 250, 500, 750, 1000):
    ab = ABAR[t - 1].item()
    print(f"  t={t:4d}  alpha_bar={ab:.5f}  SNR={ab / (1 - ab):.2e}")

x0 = sample_data(100000, gen)
x = x0.clone()
for i in range(500):                  # run the chain x_t = sqrt(1-b) x_{t-1} + sqrt(b) z for 500 steps
    x = ALPHAS[i].sqrt() * x + (1 - ALPHAS[i]).sqrt() * torch.randn_like(x)
resid = x - ABAR[499].sqrt() * x0     # closed form says this is N(0, (1 - alpha_bar) I)
print(f"chain vs closed form at t=500 (100,000 points): residual variance per axis"
      f" {resid.var(0).numpy().round(4)}, formula {1 - ABAR[499]:.4f}")
xT = ABAR[-1].sqrt() * x0 + (1 - ABAR[-1]).sqrt() * torch.randn_like(x0)
c = torch.cov(xT.T)
print(f"x_T: mean {xT.mean(0).numpy().round(3)}, var {c.diag().numpy().round(3)}, cov_xy {c[0, 1]:.3f}")

# ---- 2. Reference rows ---------------------------------------------------------
print("\nreference rows (10,000 samples each):")
print(f"  real data                    {fmt(evaluate(sample_data(N, gen)))}")
print(f"  Gaussian, same mean and cov  {fmt(evaluate(2**0.5 * torch.randn(N, 2, generator=gen)))}")
t0 = time.time()
x_exact = ddpm_sample(exact_eps, N, gen)
print(f"  DDPM with the exact eps      {fmt(evaluate(x_exact))}  ({time.time() - t0:.1f}s)")
for steps in (20, 10, 5):
    print(f"  DDIM {steps:2d} steps, exact eps   {fmt(evaluate(ddim_sample(exact_eps, N, gen, steps)))}")

# ---- 3. Train -------------------------------------------------------------------
t0 = time.time()
net, last_loss = train_eps_model(steps=3000)
print(f"\ntrained 3000 steps in {time.time() - t0:.1f}s, last batch loss {last_loss:.3f}")
eps_net = lambda x, i: net(x, torch.full((len(x),), i / T))

with torch.no_grad():
    print("learned vs exact noise prediction, E||eps_net - eps_exact||^2 at x_t ~ q_t:")
    for t in (10, 100, 300, 600, 1000):
        i = t - 1
        xt = ABAR[i].sqrt() * sample_data(N, gen) + (1 - ABAR[i]).sqrt() * torch.randn(N, 2, generator=gen)
        err = (eps_net(xt, i) - exact_eps(xt, i)).pow(2).sum(1).mean().item()
        print(f"  t={t:4d}  error {err:.4f}")

# ---- 4. Samplers -----------------------------------------------------------------
print("\nsamples from the trained network:")
t0 = time.time()
x = ddpm_sample(eps_net, N, gen)
print(f"  DDPM, 1000 steps   {fmt(evaluate(x))}  ({time.time() - t0:.1f}s)")
for steps in (100, 20, 10, 5):
    t0 = time.time()
    x = ddim_sample(eps_net, N, gen, steps)
    print(f"  DDIM, {steps:4d} steps   {fmt(evaluate(x))}  ({time.time() - t0:.2f}s)")
```

```text
alpha_bar and signal-to-noise ratio along the schedule:
  t=   1  alpha_bar=0.99990  SNR=1.00e+04
  t= 100  alpha_bar=0.89702  SNR=8.71e+00
  t= 250  alpha_bar=0.52409  SNR=1.10e+00
  t= 500  alpha_bar=0.07859  SNR=8.53e-02
  t= 750  alpha_bar=0.00335  SNR=3.36e-03
  t=1000  alpha_bar=0.00004  SNR=4.04e-05
chain vs closed form at t=500 (100,000 points): residual variance per axis [0.927  0.9228], formula 0.9214
x_T: mean [ 0.002 -0.   ], var [1.005 1.004], cov_xy -0.005

reference rows (10,000 samples each):
  real data                    in_mode 0.989  modes 8/8  mode_kl 0.000  spread 1.00
  Gaussian, same mean and cov  in_mode 0.068  modes 0/8  mode_kl 0.004  spread 1.53
  DDPM with the exact eps      in_mode 0.986  modes 8/8  mode_kl 0.000  spread 1.01  (0.3s)
  DDIM 20 steps, exact eps   in_mode 0.974  modes 8/8  mode_kl 0.000  spread 0.71
  DDIM 10 steps, exact eps   in_mode 0.773  modes 8/8  mode_kl 0.001  spread 0.83
  DDIM  5 steps, exact eps   in_mode 0.151  modes 0/8  mode_kl 0.003  spread 1.63

trained 3000 steps in 5.0s, last batch loss 0.233
learned vs exact noise prediction, E||eps_net - eps_exact||^2 at x_t ~ q_t:
  t=  10  error 0.0052
  t= 100  error 0.0352
  t= 300  error 0.0023
  t= 600  error 0.0012
  t=1000  error 0.0013

samples from the trained network:
  DDPM, 1000 steps   in_mode 0.950  modes 8/8  mode_kl 0.001  spread 1.05  (5.4s)
  DDIM,  100 steps   in_mode 0.903  modes 8/8  mode_kl 0.003  spread 0.97  (0.51s)
  DDIM,   20 steps   in_mode 0.840  modes 8/8  mode_kl 0.004  spread 0.79  (0.11s)
  DDIM,   10 steps   in_mode 0.693  modes 8/8  mode_kl 0.007  spread 0.84  (0.06s)
  DDIM,    5 steps   in_mode 0.117  modes 1/8  mode_kl 0.175  spread 1.55  (0.03s)
```

### Script 02: classifier-free guidance

```python
"""Classifier-free guidance on the 8-mode toy.

The label is the parity of the mode: class 0 = the 4 even modes, class 1 = the 4 odd ones,
so each class is itself a 4-mode distribution. One network is trained both conditionally
and unconditionally (the label is replaced by a "no class" token 10% of the time).
At sampling time
    eps = (1 + w) * eps(x, t, c) - w * eps(x, t, none)        (Ho and Salimans' convention)
so w = 0 is the plain conditional model. We ask for class 0 and measure:
  hit     fraction of samples whose nearest mode belongs to class 0
  in_mode fraction within 3 sigma of some mode
  kl4     KL(histogram over the 4 requested modes || uniform), mode balance
  spread  within-mode RMS distance relative to real data (diversity)
  radius  median distance from the origin (the real modes sit at 2.00)
"""
import torch
from toy2d import CENTERS, T, ddpm_sample, evaluate, train_eps_model

gen = torch.Generator().manual_seed(2)
NONE = 2                                          # index of the "no class" token
net, _ = train_eps_model(steps=1500, n_classes=2, p_uncond=0.1, label_fn=lambda k: k % 2)
N = 4000


def guided(w, c):
    def eps_fn(x, i):
        t01 = torch.full((len(x),), i / T)
        e_c = net(x, t01, torch.full((len(x),), c))
        if w == 0:
            return e_c
        return (1 + w) * e_c - w * net(x, t01, torch.full((len(x),), NONE))
    return eps_fn


x = ddpm_sample(lambda x, i: net(x, torch.full((len(x),), i / T), torch.full((len(x),), NONE)), N, gen)
m = evaluate(x)
print(f"unconditional branch: in_mode {m['in_mode']:.3f}  modes {m['modes']}/8  mode_kl {m['mode_kl']:.3f}\n")
print("asking for class 0 (modes 0, 2, 4, 6):")
print(f"{'w':>5} {'hit':>6} {'in_mode':>8} {'kl4':>6} {'spread':>7} {'radius':>7}")
for w in (0.0, 0.5, 1.0, 3.0, 10.0):
    x = ddpm_sample(guided(w, 0), N, gen)
    near = torch.cdist(x, CENTERS).argmin(1)
    hit = near % 2 == 0
    p = torch.bincount(near[hit] // 2, minlength=4).float()
    p = p / p.sum()
    kl4 = (p[p > 0] * (4 * p[p > 0]).log()).sum()
    m = evaluate(x)
    print(f"{w:>5.1f} {hit.float().mean():>6.3f} {m['in_mode']:>8.3f} {kl4:>6.3f} {m['spread']:>7.2f} {x.norm(dim=1).median():>7.3f}")
```

```text
unconditional branch: in_mode 0.406  modes 7/8  mode_kl 0.038

asking for class 0 (modes 0, 2, 4, 6):
    w    hit  in_mode    kl4  spread  radius
  0.0  0.993    0.922  0.000    1.17   1.993
  0.5  0.998    0.959  0.000    1.04   2.014
  1.0  0.997    0.952  0.002    0.99   2.031
  3.0  0.984    0.729  0.005    1.08   2.131
 10.0  0.991    0.220  0.040    1.02   3.837
```

### Script 03: flow matching versus DDIM

```python
"""Flow matching (rectified-flow path) versus DDIM on the same 8-mode toy.

Flow matching: x_t = (1 - t) * z + t * x1, z ~ N(0, I), x1 ~ data; regress v_theta(x_t, t)
onto x1 - z. Sampling integrates dx/dt = v_theta(x, t) from t = 0 (noise) to t = 1 (data)
with Euler steps. Both models use the same MLP, optimizer and number of training steps.
DDIM is run twice: on evenly spaced DDPM steps, and on the DDPM steps whose signal-to-noise
ratio matches the flow's evenly spaced times (alpha_bar / (1 - alpha_bar) = t^2 / (1 - t)^2).
"""
import time
import torch
from toy2d import ABAR, T, Net, evaluate, fmt, sample_data, ddim_sample, train_eps_model


def train_flow(steps=3000, batch=1024, seed=0):
    torch.manual_seed(seed)
    net = Net()
    opt = torch.optim.Adam(net.parameters(), lr=2e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(steps):
        x1, z = sample_data(batch), torch.randn(batch, 2)
        t = torch.rand(batch)
        xt = (1 - t[:, None]) * z + t[:, None] * x1
        loss = (net(xt, t) - (x1 - z)).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
    return net, loss.item()


@torch.no_grad()
def euler_sample(net, n, gen, steps):
    x = torch.randn(n, 2, generator=gen)
    for j in range(steps):
        x = x + net(x, torch.full((n,), j / steps)) / steps
    return x


def snr_matched_steps(steps):
    """DDPM indices whose SNR is closest to that of the flow times 0, 1/steps, 2/steps, ..."""
    log_snr = (ABAR / (1 - ABAR)).log()
    idx = [T - 1]                                   # t = 0 is pure noise: start at the last step
    for j in range(1, steps):
        t = j / steps
        idx.append(int((log_snr - 2 * torch.log(torch.tensor(t / (1 - t)))).abs().argmin()))
    return idx


gen = torch.Generator().manual_seed(3)
N = 10000
t0 = time.time()
fm, fm_loss = train_flow()
print(f"flow matching: trained in {time.time() - t0:.1f}s, last batch loss {fm_loss:.3f}"
      " (cannot reach 0: many (z, x1) pairs pass through the same x_t)")
t0 = time.time()
dm, _ = train_eps_model()
print(f"DDPM noise model: trained in {time.time() - t0:.1f}s")
eps_fn = lambda x, i: dm(x, torch.full((len(x),), i / T))

print("\nsteps  sampler")
for steps in (2, 5, 10, 20, 100):
    print(f"  {steps:3d}  flow + Euler       {fmt(evaluate(euler_sample(fm, N, gen, steps)))}")
    print(f"  {steps:3d}  DDIM, even steps   {fmt(evaluate(ddim_sample(eps_fn, N, gen, steps)))}")
    idx = snr_matched_steps(steps)
    print(f"  {steps:3d}  DDIM, matched SNR  {fmt(evaluate(ddim_sample(eps_fn, N, gen, steps, idx)))}")
print("\nDDPM indices used by 'matched SNR' at 5 steps:", snr_matched_steps(5))
```

```text
flow matching: trained in 5.3s, last batch loss 1.960 (cannot reach 0: many (z, x1) pairs pass through the same x_t)
DDPM noise model: trained in 5.2s

steps  sampler
    2  flow + Euler       in_mode 0.066  modes 0/8  mode_kl 0.027  spread 1.75
    2  DDIM, even steps   in_mode 0.018  modes 0/8  mode_kl 0.209  spread 1.55
    2  DDIM, matched SNR  in_mode 0.383  modes 5/8  mode_kl 0.188  spread 1.48
    5  flow + Euler       in_mode 0.776  modes 8/8  mode_kl 0.000  spread 0.73
    5  DDIM, even steps   in_mode 0.117  modes 0/8  mode_kl 0.150  spread 1.56
    5  DDIM, matched SNR  in_mode 0.674  modes 8/8  mode_kl 0.081  spread 0.89
   10  flow + Euler       in_mode 0.891  modes 8/8  mode_kl 0.001  spread 0.84
   10  DDIM, even steps   in_mode 0.689  modes 8/8  mode_kl 0.009  spread 0.82
   10  DDIM, matched SNR  in_mode 0.756  modes 8/8  mode_kl 0.148  spread 1.17
   20  flow + Euler       in_mode 0.920  modes 8/8  mode_kl 0.000  spread 0.90
   20  DDIM, even steps   in_mode 0.844  modes 8/8  mode_kl 0.002  spread 0.79
   20  DDIM, matched SNR  in_mode 0.836  modes 8/8  mode_kl 0.019  spread 1.04
  100  flow + Euler       in_mode 0.949  modes 8/8  mode_kl 0.001  spread 0.93
  100  DDIM, even steps   in_mode 0.904  modes 8/8  mode_kl 0.004  spread 0.96
  100  DDIM, matched SNR  in_mode 0.905  modes 8/8  mode_kl 0.002  spread 1.00

DDPM indices used by 'matched SNR' at 5 steps: [999, 527, 338, 187, 73]
```

### Script 04: a VAE and a GAN on the same data

```python
"""A VAE and a GAN on the same 8-mode toy, scored with the same metrics as the diffusion model.

VAE: 2D latent, Gaussian decoder with a FIXED output std sigma_x. The negative ELBO is
    ||x - decoder(z)||^2 / (2 sigma_x^2) + KL(q(z|x) || N(0, I))   (+ a constant),
so sigma_x sets how much reconstruction counts relative to the KL term.
GAN: non-saturating loss, same training budget for 3 seeds, checked during training.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from toy2d import evaluate, fmt, sample_data


def mlp(d_in, d_out, h=128):
    return nn.Sequential(nn.Linear(d_in, h), nn.SiLU(), nn.Linear(h, h), nn.SiLU(), nn.Linear(h, d_out))


def train_vae(sigma_x, steps=3000, batch=512, seed=0):
    torch.manual_seed(seed)
    enc, dec = mlp(2, 4), mlp(2, 2)
    opt = torch.optim.Adam(list(enc.parameters()) + list(dec.parameters()), lr=1e-3)
    for _ in range(steps):
        x = sample_data(batch)
        mu, logvar = enc(x).chunk(2, dim=1)
        z = mu + (0.5 * logvar).exp() * torch.randn_like(mu)        # reparameterization trick
        rec = (x - dec(z)).pow(2).sum(1) / (2 * sigma_x**2)
        kl = 0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar).sum(1)  # KL to N(0, I), closed form
        loss = (rec + kl).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        x = sample_data(10000)
        mu, logvar = enc(x).chunk(2, dim=1)
        kl = 0.5 * (mu.pow(2) + logvar.exp() - 1 - logvar).sum(1).mean().item()
        rmse = (x - dec(mu)).pow(2).sum(1).mean().sqrt().item()
        samples = dec(torch.randn(10000, 2))     # decoder means; adding sigma_x noise only blurs more
    return kl, rmse, samples


def train_gan(seed, steps=3000, batch=256, checkpoints=(500, 1000, 2000, 3000)):
    torch.manual_seed(seed)
    G, D = mlp(2, 2), mlp(2, 1)
    opt_g = torch.optim.Adam(G.parameters(), lr=1e-3, betas=(0.5, 0.999))
    opt_d = torch.optim.Adam(D.parameters(), lr=1e-3, betas=(0.5, 0.999))
    report = []
    for step in range(1, steps + 1):
        real, fake = sample_data(batch), G(torch.randn(batch, 2))
        # Discriminator: real -> 1, fake -> 0.
        d_loss = (F.binary_cross_entropy_with_logits(D(real), torch.ones(batch, 1))
                  + F.binary_cross_entropy_with_logits(D(fake.detach()), torch.zeros(batch, 1)))
        opt_d.zero_grad()
        d_loss.backward()
        opt_d.step()
        # Generator, non-saturating form: maximize log D(G(z)).
        g_loss = F.binary_cross_entropy_with_logits(D(fake), torch.ones(batch, 1))
        opt_g.zero_grad()
        g_loss.backward()
        opt_g.step()
        if step in checkpoints:
            with torch.no_grad():
                m = evaluate(G(torch.randn(10000, 2)))
            report.append(f"{step}: {m['modes']}/8 modes, in_mode {m['in_mode']:.2f}")
    return report


print("VAE (decoder std sigma_x fixed):")
for sigma_x in (0.05, 0.1, 0.3, 1.0):
    kl, rmse, x = train_vae(sigma_x)
    print(f"  sigma_x={sigma_x:<4}  KL {kl:5.2f} nats  recon RMSE {rmse:.3f}  samples: {fmt(evaluate(x))}")

print("\nGAN (modes covered and in_mode at training checkpoints):")
for seed in range(3):
    print(f"  seed {seed}:  " + " | ".join(train_gan(seed)))
```

```text
VAE (decoder std sigma_x fixed):
  sigma_x=0.05  KL  6.11 nats  recon RMSE 0.029  samples: in_mode 0.156  modes 0/8  mode_kl 0.006  spread 1.58
  sigma_x=0.1   KL  4.29 nats  recon RMSE 0.090  samples: in_mode 0.230  modes 1/8  mode_kl 0.014  spread 1.37
  sigma_x=0.3   KL  2.57 nats  recon RMSE 0.137  samples: in_mode 0.542  modes 8/8  mode_kl 0.023  spread 1.04
  sigma_x=1.0   KL  0.84 nats  recon RMSE 0.451  samples: in_mode 0.070  modes 0/8  mode_kl 0.132  spread 1.76

GAN (modes covered and in_mode at training checkpoints):
  seed 0:  500: 8/8 modes, in_mode 0.36 | 1000: 8/8 modes, in_mode 0.37 | 2000: 8/8 modes, in_mode 0.36 | 3000: 3/8 modes, in_mode 0.60
  seed 1:  500: 8/8 modes, in_mode 0.36 | 1000: 8/8 modes, in_mode 0.39 | 2000: 6/8 modes, in_mode 0.59 | 3000: 6/8 modes, in_mode 0.78
  seed 2:  500: 8/8 modes, in_mode 0.35 | 1000: 8/8 modes, in_mode 0.36 | 2000: 8/8 modes, in_mode 0.39 | 3000: 3/8 modes, in_mode 0.68
```

### Script 05: what a Fréchet distance can see

```python
"""What a Frechet distance (the 'FD' in FID) can and cannot see, on the 8-mode toy.

FD between two sample sets = ||mu1 - mu2||^2 + Tr(S1 + S2 - 2 (S1 S2)^(1/2)), computed on
some feature vector of each sample. FID uses Inception-v3 features of images; here we use
either the raw 2D coordinates or 64 fixed random Fourier features (a stand-in extractor).
"""
import numpy as np
import torch
from scipy.linalg import sqrtm
from toy2d import CENTERS, SIGMA, K, ddpm_sample, evaluate, exact_eps, sample_data

gen = torch.Generator().manual_seed(5)
W = torch.randn(2, 64, generator=gen) / 0.3      # frequencies: sensitive to structure at scale ~0.3
B = 2 * np.pi * torch.rand(64, generator=gen)


def features(x, kind):
    return x if kind == "raw" else torch.cos(x @ W + B)


def fd(x, y, kind):
    fx, fy = features(x, kind).double().numpy(), features(y, kind).double().numpy()
    m1, m2 = fx.mean(0), fy.mean(0)
    s1, s2 = np.cov(fx, rowvar=False), np.cov(fy, rowvar=False)
    covmean = sqrtm(s1 @ s2).real
    return float(((m1 - m2) ** 2).sum() + np.trace(s1 + s2 - 2 * covmean))


def mixture(n, sigma=SIGMA, modes=range(K)):
    modes = torch.tensor(list(modes))
    k = modes[torch.randint(0, len(modes), (n,), generator=gen)]
    return CENTERS[k] + sigma * torch.randn(n, 2, generator=gen)


n = 2000
test = sample_data(10000, gen)                   # the 'real' reference set
train = sample_data(500, gen)                    # a small training set
copies = train[torch.randint(0, len(train), (n,), generator=gen)]
exact_empirical = ddpm_sample(lambda x, i: exact_eps(x, i, centers=train, sigma0=0.0), n, gen)

candidates = {
    "fresh real data": sample_data(n, gen),
    "Gaussian, matched mean/cov": torch.randn(n, 2, generator=gen) * 2.005**0.5,
    "7 of 8 modes (one dropped)": mixture(n, modes=range(7)),
    "modes 2x too wide": mixture(n, sigma=2 * SIGMA),
    "modes 2x too narrow": mixture(n, sigma=SIGMA / 2),
    "copies of 500 train points": copies,
    "diffusion, exact score of train set": exact_empirical,
}
print(f"{'samples (n=2000) vs 10,000 test points':<38} {'FD raw':>8} {'FD feat':>8} {'in_mode':>8}"
      f" {'modes':>6} {'NN dist':>8}")
for name, x in candidates.items():
    nn_dist = torch.cdist(x, train).min(1).values.median().item()   # distance to closest train point
    m = evaluate(x)
    print(f"{name:<38} {fd(x, test, 'raw'):>8.4f} {fd(x, test, 'feat'):>8.4f} {m['in_mode']:>8.3f}"
          f" {m['modes']:>4}/8 {nn_dist:>8.4f}")

print("\nFD between two independent sets of REAL data, by sample size (features):")
for m_size in (50, 200, 1000, 5000):
    vals = [fd(sample_data(m_size, gen), sample_data(m_size, gen), "feat") for _ in range(5)]
    print(f"  n={m_size:<5} mean {np.mean(vals):.4f}  (min {min(vals):.4f}, max {max(vals):.4f})")
```

```text
samples (n=2000) vs 10,000 test points   FD raw  FD feat  in_mode  modes  NN dist
fresh real data                          0.0029   0.0844    0.985    8/8   0.0216
Gaussian, matched mean/cov               0.0073  24.9292    0.071    0/8   0.6119
7 of 8 modes (one dropped)               0.1142   3.6864    0.988    7/8   0.0223
modes 2x too wide                        0.0021   4.1339    0.652    8/8   0.0593
modes 2x too narrow                      0.0014   1.5125    1.000    8/8   0.0163
copies of 500 train points               0.0336   0.5212    0.983    8/8   0.0000
diffusion, exact score of train set      0.0222   0.3414    0.976    8/8   0.0025

FD between two independent sets of REAL data, by sample size (features):
  n=50    mean 6.2450  (min 4.4468, max 9.2826)
  n=200   mean 1.3666  (min 0.9610, max 1.8078)
  n=1000  mean 0.2402  (min 0.1836, max 0.2702)
  n=5000  mean 0.0522  (min 0.0342, max 0.0759)
```

## Common mistakes

- **Trusting one quality number.** The GAN in script 04 "improved" while losing five of eight modes. Measure fidelity and coverage separately.
- **Comparing FIDs across setups.** Sample count, feature network, resizing and reference set all change the number.
- **Blaming the model for sampler error.** At 5 DDIM steps even the exact score fails. Before retraining, test the sampler with more steps or, on a toy, with an exact score.
- **Evenly spaced steps on a schedule that spends half its range on noise.** Place steps by noise level (SNR), not by index.
- **Cranking guidance.** High $w$ eventually leaves the data manifold. Sweep it and look at samples.
- **Train/sample mismatch in time conventions.** DDPM indexes $t$ from data (0) to noise ($T$); flow matching here goes from noise (0) to data (1). Mixing them silently produces garbage.
- **Not checking for memorization.** Compare samples with their nearest training examples.

## Exercises

1. [warm-up] Using $\bar\alpha_t$ from script 01, at what $t$ is the SNR closest to 1? What fraction of the 1000 steps have SNR below 0.01?
2. [warm-up] Show that $x_t = \sqrt{\bar\alpha_t} x_0 + \sqrt{1 - \bar\alpha_t}\,\epsilon$ follows from two applications of the one-step forward process.
3. [warm-up] In the guidance formula, what does $w = -1$ give? What about $w = 0$?
4. [core] Derive $D^*(x)$ for the GAN objective by maximizing the integrand pointwise in $D(x)$.
5. [core] In script 01, train for 10,000 steps instead of 3,000. Does the gap between learned and exact DDPM samples close? Does the error at $t = 100$ fall most?
6. [core] Replace the linear $\beta$ schedule with one where $\log \mathrm{SNR}$ falls linearly in $t$ from $+9$ to $-10$. Rerun DDIM at 5 and 10 evenly spaced steps.
7. [core] In script 02, set `p_uncond` to 0.5 and rerun. How do the unconditional branch and the $w = 3$ row change?
8. [core] In script 05, add a model that outputs only 4 of the 8 modes with correct widths. Where does it rank on each FD, and why does in_mode not catch it?
9. [stretch] Train the diffusion MLP on only 64 training points for many steps. Track the median nearest-training-point distance of samples over training. When does it start to memorize?
10. [stretch] Add a "reflow" step to script 03: generate (noise, sample) pairs with the trained flow, retrain on those pairs, and measure 1- and 2-step quality.

### Hints and answers

1. $t = 259$. SNR is below 0.01 from $t = 675$ on, 32.6% of the steps; compute it with `(ABAR / (1 - ABAR) < 0.01).float().mean()`.
2. $x_2 = \sqrt{\alpha_2}(\sqrt{\alpha_1} x_0 + \sqrt{1-\alpha_1}\epsilon_1) + \sqrt{1-\alpha_2}\epsilon_2$; the two noise terms add to a Gaussian with variance $\alpha_2(1 - \alpha_1) + 1 - \alpha_2 = 1 - \alpha_1\alpha_2$. Induct.
3. $w = -1$ gives the unconditional model; $w = 0$ the plain conditional model.
4. Maximize $p\log D + q \log(1 - D)$ over $D \in (0,1)$: the derivative $p/D - q/(1 - D) = 0$ gives $D = p/(p+q)$.
5. Compare the printed errors at each $t$ before and after; the $t = 100$ row is the one to watch. (Not run for this page; report what you find.)
6. Use `ABAR = torch.sigmoid(log_snr)`, since $\bar\alpha = \mathrm{SNR}/(1 + \mathrm{SNR})$. Evenly spaced steps on such a schedule are spread evenly in log-SNR, so expect results closer to the "SNR-matched" column than to the even-steps one; check it.
7. The unconditional branch should improve, the conditional one may worsen slightly, and large-$w$ overshoot should shrink. Ho and Salimans found 0.5 worse on images; check whether the toy agrees.
8. in_mode is computed per sample, so it cannot see missing modes; `modes` and `mode_kl` can. FD on features should penalize it heavily, raw FD less so.
9. The empirical optimum of section 5 memorizes, so the nearest-neighbour distance should fall as the network fits better; how fast depends on network size and training length.
10. Liu et al. report that rectifying makes paths straighter and gives good results with very few steps; measure whether the toy shows a gain at 1 and 2 steps (not run for this page).

## Check your understanding

1. **Why can a noise predictor generate data?** The optimal noise prediction is a scaled score, $\epsilon^* = -\sqrt{1-\bar\alpha_t}\,\nabla \log q_t$, and the score at every noise level is all the reverse-time SDE or probability flow ODE needs to carry Gaussian noise back to the data distribution.

2. **Why do VAE samples land between modes?** A continuous decoder maps a connected Gaussian, so some prior mass decodes between clusters; with small $\sigma_x$ the codes also stop matching the prior. Script 04 put at most 54% of samples near a mode.

3. **What does classifier-free guidance do to the distribution?** It adds $w\,\nabla\log q_t(c \mid x)$ to the conditional score, pushing samples toward regions where the condition is unambiguous. This raises fidelity and prompt adherence, lowers diversity, and at large $w$ drives samples off the data (radius 3.84 in script 02).

4. **Is flow matching "better" than diffusion?** It is a clean objective with simple ODE sampling and often needs fewer steps, but diffusion is a special case of its Gaussian paths, and in script 03 much of the few-step gap closed once DDIM used steps at matching noise levels. Compare with matched schedules, architectures and budgets.

5. **A paper reports a lower FID than the previous best. What do you check?** Same sample count, feature network, preprocessing and reference set; whether matching ImageNet class statistics could explain it; precision and recall separately; memorization; and human evaluation, since Stein et al. found no metric correlated strongly with human judgment.

## Further reading

- **Lilian Weng, "What are Diffusion Models?"** (2021, updated 2022 and 2024, blog). https://lilianweng.github.io/posts/2021-07-11-diffusion-models/
  *Read for:* a careful, derivation-level tour of DDPM, score matching connections, DDIM, guidance, latent diffusion, distillation and consistency models. *Level:* intro.
- **Calvin Luo, "Understanding Diffusion Models: A Unified Perspective"** (2022, arXiv:2208.11970). https://arxiv.org/abs/2208.11970
  *Read for:* the ELBO view of diffusion as a hierarchical VAE, the three equivalent prediction targets, and Tweedie's formula linking them to scores. *Level:* intro to intermediate.
- **Diederik P. Kingma and Max Welling, "Auto-Encoding Variational Bayes"** (2014, ICLR; arXiv:1312.6114). https://arxiv.org/abs/1312.6114
  *Read for:* the ELBO, the reparameterization trick and the original VAE. *Level:* intermediate.
- **Ian J. Goodfellow, Jean Pouget-Abadie, Mehdi Mirza, Bing Xu, David Warde-Farley, Sherjil Ozair, Aaron Courville, Yoshua Bengio, "Generative Adversarial Nets"** (2014, NeurIPS 27; arXiv:1406.2661). https://arxiv.org/abs/1406.2661
  *Read for:* the minimax game, the optimal discriminator and the non-saturating generator loss. *Level:* intermediate.
- **Jonathan Ho, Ajay Jain, Pieter Abbeel, "Denoising Diffusion Probabilistic Models"** (2020, NeurIPS 33; arXiv:2006.11239). https://arxiv.org/abs/2006.11239
  *Read for:* the forward process, the simplified noise-prediction loss, the schedule used in this module, and an honest discussion of likelihoods. *Level:* intermediate.
- **Yang Song, Jascha Sohl-Dickstein, Diederik P. Kingma, Abhishek Kumar, Stefano Ermon, Ben Poole, "Score-Based Generative Modeling through Stochastic Differential Equations"** (2021, ICLR; arXiv:2011.13456). https://arxiv.org/abs/2011.13456
  *Read for:* the SDE view, reverse-time sampling with the score, and the probability flow ODE. *Level:* advanced.
- **Jonathan Ho and Tim Salimans, "Classifier-Free Diffusion Guidance"** (2022, arXiv:2207.12598; short version at a NeurIPS 2021 workshop). https://arxiv.org/abs/2207.12598
  *Read for:* the guidance formula, label dropout, and the FID versus Inception-score trade-off. Short and readable. *Level:* intermediate.
- **Robin Rombach, Andreas Blattmann, Dominik Lorenz, Patrick Esser, Björn Ommer, "High-Resolution Image Synthesis with Latent Diffusion Models"** (2022, CVPR; arXiv:2112.10752). https://arxiv.org/abs/2112.10752
  *Read for:* diffusion in an autoencoder's latent space, the choice of compression factor, and cross-attention conditioning. *Level:* intermediate.
- **Yaron Lipman, Ricky T. Q. Chen, Heli Ben-Hamu, Maximilian Nickel, Matt Le, "Flow Matching for Generative Modeling"** (2023, ICLR; arXiv:2210.02747). https://arxiv.org/abs/2210.02747
  *Read for:* the conditional flow matching objective, why it recovers the marginal flow, and diffusion as a special case. *Level:* advanced.
- **Patrick Esser, Sumith Kulal, Andreas Blattmann, et al. (17 authors), "Scaling Rectified Flow Transformers for High-Resolution Image Synthesis"** (2024, ICML; arXiv:2403.03206). https://arxiv.org/abs/2403.03206
  *Read for:* a large comparison of diffusion and rectified-flow formulations, time sampling for training, and the transformer used in Stable Diffusion 3. *Level:* advanced.
- **Martin Heusel, Hubert Ramsauer, Thomas Unterthiner, Bernhard Nessler, Sepp Hochreiter, "GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium"** (2017, NeurIPS 30; arXiv:1706.08500). https://arxiv.org/abs/1706.08500
  *Read for:* the definition of FID and the case made for it. *Level:* intermediate.
- **George Stein, Jesse C. Cresswell, Rasa Hosseinzadeh, Yi Sui, Brendan Leigh Ross, Valentin Villecroze, Zhaoyan Liu, Anthony L. Caterini, J. Eric T. Taylor, Gabriel Loaiza-Ganem, "Exposing flaws of generative model evaluation metrics and their unfair treatment of diffusion models"** (2023, NeurIPS; arXiv:2306.04675). https://arxiv.org/abs/2306.04675
  *Read for:* a large human study against 17 metrics, the case against Inception features, and why memorization is hard to measure. *Level:* intermediate.
