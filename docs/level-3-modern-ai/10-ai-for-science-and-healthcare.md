# AI for Science and Healthcare

> **Level 3 · Module 10** · Prerequisites: L1-01, L1-05, L2-03, L3-09 · Time: 6 to 8 hours · Code: `code/L3-10-ai-for-science-and-healthcare/`

> **Note:** Every number in this module about a real system was checked against a primary or authoritative source in October 2026 and is dated where it can change. The code uses **synthetic data only**. It teaches evaluation methods, it does not describe any real patient population, and nothing here is medical advice.

## What you will be able to do

- Place a published claim on an evidence ladder, from retrospective benchmark to randomised trial, and say who validated it and how.
- Summarise what has been demonstrated, and what is still claimed or contested, for protein structure prediction, machine-learning weather forecasting, materials and drug discovery, and medical imaging and clinical prediction (as of October 2026).
- Show with code how shortcut learning and site differences inflate internal validation, and why external validation is necessary but not sufficient.
- Audit a model by subgroup with confidence intervals, and trace a disparity to the choice of label or feature.
- Translate a model's metrics into deployment quantities (alerts, positive predictive value, net benefit) and monitor calibration over time.
- Explain the main regulatory facts (US FDA, EU AI Act) well enough to ask the right questions.

## 1. How to read a claim

Science and medicine are where machine learning has some of its clearest successes and some of its most expensive failures. The difference is usually in the validation, not the architecture. Rank evidence roughly like this:

1. **Retrospective, random split.** Train and test rows come from the same pool. Good for development, weak as evidence (L1-01).
2. **Held out by group or time.** Test on new protein families, new chemical scaffolds, new hospitals, or later years. This asks whether the model generalises the way deployment needs.
3. **Blind or prospective.** Predictions are made before the answer exists: a protein structure not yet released, tomorrow's weather, a compound not yet synthesised.
4. **External and independent.** Someone other than the developer evaluates it on their own data.
5. **Randomised trial with an outcome that matters**, such as cancers missed, not only an accuracy score.
6. **Post-deployment monitoring**, because the world keeps changing.

For each case below, ask: what exactly was shown, by whom, at which rung, and against what comparator?

## 2. Protein structure prediction

A protein's function depends on the 3D shape its amino-acid chain folds into. Experimental methods (X-ray crystallography, cryo-electron microscopy) are slow and expensive. The **CASP** experiment has long tested predictors blind on structures that were solved but not yet public, which puts it at rung 3.

**Demonstrated.** In CASP14, AlphaFold2's median backbone accuracy was 0.96 Å r.m.s.d.$_{95}$ (root-mean-square deviation over the best-fitting 95% of residues), against 2.8 Å for the next best method; all-atom accuracy was 1.5 Å against 3.5 Å [Jumper et al., 2021]. The model also outputs a per-residue confidence, **pLDDT**, which the authors report reliably predicts its local accuracy. The same paper reports the limits: accuracy drops substantially when the multiple sequence alignment has fewer than about 30 sequences, and is weaker for chains whose shape depends mostly on contacts with other chains. The AlphaFold database now holds over 214 million predicted structures [Varadi et al., 2024]. In 2024 the Nobel Prize in Chemistry went half to David Baker "for computational protein design" and half jointly to Demis Hassabis and John Jumper "for protein structure prediction" [Nobel Prize press release, 2024].

**Independently checked.** Terwilliger and colleagues compared AlphaFold predictions with experimental electron-density maps for 102 crystal structures [Terwilliger et al., 2024]. In the highest-confidence regions (pLDDT above 90) the median error of the Cα (backbone) atom positions was 0.6 Å, but about 10% of those residues were off by more than 2 Å. For comparison, the same protein crystallised in two different crystal forms differs by a median of 0.3 Å. Their conclusion is in the title: predictions are valuable hypotheses that accelerate, but do not replace, experimental structure determination.

**Extended, with stated limits.** AlphaFold 3 predicts complexes of proteins with nucleic acids, small molecules and ions, and its authors report far greater accuracy on protein-ligand complexes than state-of-the-art docking tools [Abramson et al., 2024]. The paper lists its own limitations: chirality violations in 4.4% of a ligand benchmark, plausible-looking but invented structure in disordered regions, static structures only, and missed conformational states. As of October 2026, the inference code is on GitHub under Apache 2.0, the model weights are available from Google under separate terms of use, and the repository states that outputs are not validated or approved for clinical use.

**Still a separate claim.** Accurate static structures do not by themselves show that drug discovery gets faster or cheaper. That needs its own evidence (section 4).

## 3. Weather and climate

Weather is perhaps the cleanest success, for structural reasons: decades of consistent training data (the ERA5 **reanalysis**, a physically consistent reconstruction of past weather), clear metrics, and verification that is automatically prospective, since every forecast is scored against weather that had not yet happened.

**Demonstrated (developer evaluations, peer-reviewed).** GraphCast, a graph neural network (L3-09) trained on ERA5 from 1979 to 2017, forecasts hundreds of variables 10 days ahead at 0.25° resolution in under a minute, and significantly outperformed ECMWF's deterministic HRES forecast on 90% of 1,380 verification targets in 2018 test data [Lam et al., 2023]. GenCast, a diffusion model (L3-03) that produces 15-day ensembles, had greater skill than ECMWF's ensemble (ENS) on 97.2% of 1,320 targets [Price et al., 2025]. Its authors list the limits: it starts from ERA5 rather than the operational analysis, it depends on traditional data assimilation for its inputs, it runs at 0.25° while ENS has moved to 0.1°, and precipitation was left out of the headline evaluation because of doubts about ERA5 precipitation quality. WeatherBench 2 provides a shared evaluation framework for such comparisons [Rasp et al., 2024].

**Adopted operationally.** ECMWF made its own machine-learning model, AIFS Single, operational on 25 February 2025, running alongside the physics-based IFS at 28 km grid spacing (the IFS uses 9 km), with initial conditions still supplied by the IFS data assimilation [ECMWF, February 2025]. The 51-member AIFS ENS followed on 1 July 2025 at 31 km [ECMWF, July 2025]. ECMWF reports gains of up to 20% on some measures and about 1,000 times less energy per forecast, and says the physics-based ensemble remains indispensable for high-resolution fields and coupled Earth-system processes. Its Spring 2026 newsletter describes AIFS v2, to be implemented alongside IFS cycle 50r1, and notes that some improvements in physical realism did not translate into better verification scores [ECMWF Newsletter 187, 2026].

**Contested or unsolved.**

- **Extremes beyond the training data.** Sun and colleagues retrained the FourCastNet model with all category 3 to 5 tropical cyclones removed from training. It then failed to forecast category 5 storms, although versions missing strong storms in only one basin showed some skill there, suggesting cross-basin transfer [Sun et al., 2025]. Rare events are exactly where forecasts matter most.
- **Climate.** NeuralGCM, a hybrid of a differentiable physics solver and learned components, tracks climate metrics over decades with prescribed sea-surface temperatures, but its authors state that it does not extrapolate to substantially different future climates [Kochkov et al., 2024]. Projecting a warmer world is extrapolation by definition.
- **Independence from physics.** Every operational system above still starts from physics-based data assimilation.

## 4. Materials and drug discovery

Discovery has a special evaluation problem: the goal is to find things *better than anything in the training set*, which is extrapolation, and the final test is a laboratory, not a held-out split.

**Materials: a large claim and its critics.** GNoME, a deep-learning approach scaled up with large-scale active learning, reported 2.2 million crystal structures below the current convex hull (predicted stable relative to known phases, by density functional theory), described as including 381,000 new stable materials, of which 736 had already been independently made [Merchant et al., 2023]. Cheetham and Seshadri, solid-state chemists, examined samples of the output: the first 250 of 2,047 entries in one archive, a space-group analysis of 384,870 entries, and 10 random compounds checked against the inorganic crystal database [Cheetham and Seshadri, 2024]. They report "scant evidence" of compounds that are novel, credible and useful at once: many predictions order metal ions that would be disordered at synthesis temperatures, all 10 random compounds had known analogues, and no functionality was demonstrated.

**Materials: an autonomous lab and a correction.** The A-Lab paper reported that a robotic lab, guided by computation and machine learning, made 41 novel compounds from 58 targets in 17 days [Szymanski et al., 2023]. Leeman and colleagues examined the reported products and concluded that "no new materials have been discovered", estimating that about two-thirds were likely known compositionally disordered versions of the predicted ordered compounds, and that automated analysis of the X-ray diffraction data was unreliable [Leeman et al., 2024]. In a January 2026 author correction, the A-Lab authors reported that a peer-reviewed re-analysis confirmed 36 of their 40 reported successes and found 4 inconclusive, removed one compound that had been in the training data, and clarified that "novel" meant new to the prediction platform, not necessarily new to science [Szymanski et al., correction, 2026]. The benchmark side has responded too: Matbench Discovery evaluates models as pre-filters in a prospective-style discovery task and finds that good regression metrics can still give many false positives near the stability boundary [Riebesell et al., 2025].

**Drugs: demonstrated early steps.** Stokes and colleagues trained a directed message-passing network on 2,335 molecules screened for growth inhibition of *E. coli*, and found halicin, a structurally unusual antibacterial, in a drug-repurposing library; it was active against several resistant pathogens in vitro and in mouse infection models [Stokes et al., 2020]. That is a strong preclinical result, validated in the lab, not a medicine.

**Drugs: the clinical pipeline.** An analysis of AI-native biotech pipelines found that AI-discovered molecules had an 80 to 90% success rate in phase I (mostly a safety test), higher than historical averages, and about 40% in phase II, similar to historical averages but on a limited sample [Jayatunga et al., 2024]. A prominent published example is rentosertib, whose target and molecule its developer attributes to generative AI. In a 12-week randomised phase 2a trial with 71 patients with idiopathic pulmonary fibrosis, the primary endpoint (patients with any treatment-emergent adverse event) was similar across arms, and lung capacity (forced vital capacity) changed by +98.4 mL (95% CI 10.9 to 185.9) on the highest dose against −20.3 mL on placebo [Xu et al., 2025]. The trial was sponsored by the developer, small, and exploratory. A phase 3 trial (NCT07687459) was listed as recruiting on ClinicalTrials.gov in October 2026. I found no authoritative record of a regulatory approval for a drug whose discovery is credited mainly to AI as of October 2026; note that regulators do not classify drugs by discovery method, so "first AI drug" claims depend on definitions.

Script 4 shows the evaluation problem in miniature: a random forest looks best on a random split but worst when whole families are held out, and it cannot predict any value above the best it was trained on.

## 5. Medical imaging and clinical prediction

**From retrospective accuracy to a pivotal trial.** A 2016 deep-learning system for referable diabetic retinopathy reached ROC AUC 0.991 and 0.990 on two retrospective validation sets [Gulshan et al., 2016]. A different system, IDx-DR, was tested prospectively in 900 people with diabetes at primary-care clinics, against a rigorous reading-centre reference, and reached sensitivity 87.2% and specificity 90.7%, above its pre-specified endpoints [Abràmoff et al., 2018]. FDA granted its De Novo request on 11 April 2018 [FDA, DEN180001]; the authors describe it as the first FDA-authorised autonomous AI diagnostic system. Two authors held equity in the company, which is common and is exactly why independent evaluation matters.

**Deployment is its own test.** Observing a deep-learning screening system in 11 clinics in Thailand, Beede and colleagues found that socio-environmental factors affected model performance, nursing workflows and patient experience, and that the model's image-quality thresholds were in tension with the images a resource-constrained clinic could produce [Beede et al., 2020].

**A randomised trial with a clinical endpoint.** MASAI randomised women in Swedish breast screening to AI-supported screening (an AI risk score triaged exams to one or two radiologists and flagged suspicious areas) or standard double reading. The 2023 safety analysis of 80,033 women found cancer detection of 6.1 against 5.1 per 1,000 (ratio 1.2, 95% CI 1.0 to 1.5) with 46,345 screen readings instead of 83,231, about 44% fewer [Lång et al., 2023]. The primary endpoint, published in 2026 for 105,934 women, was the **interval cancer** rate (cancers that surface between screening rounds): 1.55 against 1.76 per 1,000, a ratio of 0.88 (95% CI 0.65 to 1.18), which met the non-inferiority margin, with higher sensitivity (80.5% against 73.8%) and identical specificity (98.5%) [Gommers et al., 2026]. Read that carefully: press coverage described "12% fewer interval cancers", but the interval includes no difference. What was shown is "not worse, with much less radiologist work", for one AI product in one screening programme.

**Clinical prediction in the wild.** The Epic Sepsis Model was deployed at hundreds of US hospitals before Wong and colleagues validated it externally on 38,455 hospitalisations at Michigan Medicine: hospitalisation-level AUC 0.63, and it missed 67% of sepsis cases while generating alerts for 18% of all hospitalised patients [Wong et al., 2021]. A systematic review of COVID-19 imaging models screened 2,212 studies, included 62, and judged none of potential clinical use because of methodological flaws or biases [Roberts et al., 2021].

## 6. The specific risks

**Shortcut learning and site shift.** A **shortcut** is a decision rule that works on the benchmark but fails under more demanding conditions [Geirhos et al., 2020]. Medical data is full of them. Zech and colleagues trained pneumonia detectors on chest X-rays from three hospital systems: a CNN could identify the hospital system of a radiograph almost perfectly (99.95% and 99.98%), and because pneumonia prevalence differed so much between sites (34.2% against about 1%), sorting by hospital alone gave AUC 0.861; internal performance exceeded external in 3 of 5 comparisons [Zech et al., 2018]. DeGrave and colleagues showed COVID-19 X-ray models relying on confounding factors rather than pathology, and that external validation alone may not catch this, because a shortcut can also hold at the new hospital [DeGrave et al., 2021]. Script 1 reproduces both effects. Dataset shift in general is covered in L1-05.

**Evaluation pitfalls.** Leakage (L1-01) takes clinical forms: images of the same patient on both sides of a split, or features recorded after the decision point. AUC alone does not show deployment value: positive predictive value falls with prevalence, and calibration drifts (script 3). **Net benefit** from decision-curve analysis weighs true positives against false positives at the risk threshold a clinician would act on [Vickers and Elkin, 2006]. The TRIPOD+AI statement, a 27-item reporting checklist for prediction models built with regression or machine learning, supersedes TRIPOD 2015 [Collins et al., 2024].

**Equity.** Obermeyer and colleagues found that a widely used commercial algorithm, which predicted health-care cost as a proxy for need, assigned Black patients the same risk score as White patients who were considerably healthier, because less was spent on Black patients' care; fixing the disparity would raise the share of Black patients selected for extra help from 17.7% to 46.5% [Obermeyer et al., 2019]. Chest X-ray classifiers have been found to underdiagnose under-served groups, with higher rates in intersectional groups such as Hispanic female patients [Seyyed-Kalantari et al., 2021], and models can predict self-reported race from medical images (AUC 0.91 to 0.99 for X-ray images) even when images are heavily degraded and clinical experts cannot [Gichoya et al., 2022]. Script 2 shows the label-choice mechanism and why fixing the label alone may not be enough. Fairness definitions and their conflicts are in L4-05.

**Regulation (as of October 2026; not legal advice).**

- **United States.** FDA regulates some AI software as medical devices. Its public list of AI-enabled devices (content current as of 4 September 2026) is explicitly not comprehensive [FDA AI-enabled device list]. A June 2026 Congressional Research Service summary puts the number authorised at about 1,450, mostly in radiology, cardiology and neurology, mostly cleared through the 510(k) pathway (substantial equivalence to an existing device), and notes that FDA did not appear to have authorised any generative-AI device [CRS, 2026]. **Predetermined change control plans**, authorised by Congress in 2022, let a manufacturer pre-specify how a model will be updated; FDA's final guidance on them for AI-enabled devices is on its site in a version dated August 2025 [FDA PCCP guidance].
- **European Union.** Under the AI Act (Regulation (EU) 2024/1689), an AI system that is a safety component of, or is itself, a product covered by the legislation in its Annex I and requiring third-party conformity assessment is high-risk; Annex I includes the medical device and in vitro diagnostic regulations. The Digital Omnibus on AI (Regulation (EU) 2026/1744, published 24 July 2026, in force 27 July 2026) moved the application of the high-risk requirements to 2 December 2027 for Annex III systems and to 2 August 2028 for Annex I products such as medical devices [Regulation (EU) 2026/1744].

**Clinical validation.** A model is a means; the outcome is patient health. Retrospective AUC says nothing about whether clinicians use the output, whether it changes decisions, or whether those decisions help. MASAI shows what the stronger evidence looks like and how long it takes: randomisation began in April 2021 and the primary endpoint appeared in 2026.

## Hands-on

All four scripts use **synthetic data**: invented features, groups and effect sizes, built to make a methodological point. None of it is medical advice or describes a real population. Run from `code/L3-10-ai-for-science-and-healthcare/` with the course Python (scikit-learn 1.9.1). Each took under 10 seconds on the machine used. Trailing digits may differ on other platforms.

### Script 1: shortcuts, site shift and external validation

```python
"""SYNTHETIC DATA ONLY. NOT MEDICAL ADVICE. No real patients, features or
effect sizes are represented; the numbers are invented to show a method.

A clinical-prediction toy with four hospitals. At the two development sites,
sicker patients are more often imaged on a portable device (a shortcut), and
the sites differ in prevalence and in a scanner signature. Compare internal,
leave-one-site-out and external validation, with and without the shortcut."""
import os

# Many small fits: extra threads mostly add overhead, and on a shared machine
# they can slow things down a lot. Must be set before numpy is imported.
for var in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ.setdefault(var, "4")

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, roc_auc_score
from sklearn.model_selection import train_test_split

rng = np.random.default_rng(0)
PHYS = ["age", "lab_a", "lab_b", "lab_c"]
ALL = PHYS + ["portable", "scanner"]


def make_site(n, base_logit, portable_if_ill, portable_if_well, scanner_mu, age_shift=0.0):
    """Outcome depends only on physiology (plus a site-level baseline risk).
    'portable' depends on the outcome; 'scanner' only identifies the site."""
    age = rng.normal(age_shift, 1, n)
    lab_a, lab_b, lab_c = rng.normal(0, 1, (3, n))
    logit = base_logit + 0.8 * age + 1.0 * lab_a + 0.6 * lab_b + 0.5 * lab_a * lab_b
    y = rng.random(n) < 1 / (1 + np.exp(-logit))
    portable = np.where(y, rng.random(n) < portable_if_ill, rng.random(n) < portable_if_well)
    scanner = rng.normal(scanner_mu, 0.3, n)
    return np.column_stack([age, lab_a, lab_b, lab_c, portable, scanner]), y.astype(int)


sites = {  # development sites A, B; external C (shortcut absent), D (shortcut present)
    "A": make_site(5000, -2.5, 0.80, 0.10, 0.0),
    "B": make_site(5000, -0.5, 0.80, 0.10, 2.0),
    "C": make_site(5000, -1.5, 0.30, 0.30, 1.0, age_shift=0.5),
    "D": make_site(5000, -1.5, 0.80, 0.10, 1.0),
}
for s, (X, y) in sites.items():
    print(f"site {s}: n={len(y)}, outcome rate {y.mean():.1%}, "
          f"portable among ill {X[y == 1, 4].mean():.0%}, among well {X[y == 0, 4].mean():.0%}")


def metrics(model, X, y):
    """AUC (discrimination), calibration-in-the-large (observed minus mean
    predicted risk), calibration slope (1 is ideal) and Brier score."""
    p = np.clip(model.predict_proba(X)[:, 1], 1e-6, 1 - 1e-6)
    logit = np.log(p / (1 - p))[:, None]
    slope = LogisticRegression(C=1e6).fit(logit, y).coef_[0, 0]
    return roc_auc_score(y, p), y.mean() - p.mean(), slope, brier_score_loss(y, p)


def make_model():
    return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.05, random_state=0)


XA, yA = sites["A"]; XB, yB = sites["B"]
X_dev, y_dev = np.vstack([XA, XB]), np.r_[yA, yB]
site_dev = np.r_[np.zeros(len(yA)), np.ones(len(yB))]

# Can a model tell the development sites apart from the inputs alone?
Xs_tr, Xs_te, ss_tr, ss_te = train_test_split(X_dev, site_dev, test_size=0.3, random_state=0)
site_clf = make_model().fit(Xs_tr, ss_tr)
print(f"\nsite A vs B identifiable from the inputs: AUC {roc_auc_score(ss_te, site_clf.predict_proba(Xs_te)[:, 1]):.3f}")

print(f"\n{'features':<11} {'evaluation':<26} {'AUC':>5} {'obs-pred':>9} {'slope':>6} {'Brier':>6}")
for label, cols in [("all", list(range(6))), ("physiology", list(range(4)))]:
    Xtr, Xte, ytr, yte = train_test_split(X_dev[:, cols], y_dev, test_size=0.25,
                                          stratify=y_dev, random_state=0)
    model = make_model().fit(Xtr, ytr)
    rows = [("internal (random split)", metrics(model, Xte, yte))]
    # Leave one development site out: train on A, test on B, and vice versa.
    for tr, te in [("A", "B"), ("B", "A")]:
        m = make_model().fit(sites[tr][0][:, cols], sites[tr][1])
        rows.append((f"train {tr} -> test {te}", metrics(m, sites[te][0][:, cols], sites[te][1])))
    for ext in ["C", "D"]:
        rows.append((f"external site {ext}", metrics(model, sites[ext][0][:, cols], sites[ext][1])))
    for name, (auc, citl, slope, brier) in rows:
        print(f"{label:<11} {name:<26} {auc:5.3f} {citl:+9.3f} {slope:6.2f} {brier:6.3f}")
    if label == "all":
        imp = permutation_importance(model, Xte, yte, scoring="roc_auc", n_repeats=5, random_state=0)
        order = np.argsort(-imp.importances_mean)
        print("  permutation importance (drop in internal AUC): "
              + ", ".join(f"{ALL[i]} {imp.importances_mean[i]:.3f}" for i in order[:4]))
```

```text
site A: n=5000, outcome rate 13.9%, portable among ill 79%, among well 10%
site B: n=5000, outcome rate 40.6%, portable among ill 81%, among well 10%
site C: n=5000, outcome rate 30.1%, portable among ill 31%, among well 31%
site D: n=5000, outcome rate 24.8%, portable among ill 79%, among well 10%

site A vs B identifiable from the inputs: AUC 0.999

features    evaluation                   AUC  obs-pred  slope  Brier
all         internal (random split)    0.942    -0.001   0.90  0.077
all         train A -> test B          0.917    +0.180   0.73  0.158
all         train B -> test A          0.928    -0.180   0.85  0.124
all         external site C            0.676    +0.011   0.24  0.249
all         external site D            0.909    +0.007   0.75  0.092
  permutation importance (drop in internal AUC): portable 0.197, lab_a 0.047, scanner 0.041, lab_b 0.028
physiology  internal (random split)    0.776    +0.007   0.84  0.157
physiology  train A -> test B          0.778    +0.280   0.59  0.277
physiology  train B -> test A          0.828    -0.273   0.77  0.195
physiology  external site C            0.794    -0.019   0.97  0.160
physiology  external site D            0.802    -0.020   0.95  0.142
```

Read the table top to bottom. With all features, internal validation looks excellent (AUC 0.942) and permutation importance shows why: `portable` dominates. At site C, where portable imaging no longer tracks illness, AUC collapses to 0.676 with calibration slope 0.24, meaning wildly overconfident risks. At site D, where the same habit holds, external validation looks fine (0.909), even though the model relies on the shortcut exactly as much; this is DeGrave's warning in code. The physiology-only model looks *worse* internally (0.776) and is the one that transfers (0.794 and 0.802, slopes near 1). Leave-one-site-out shows a third problem: across sites with different prevalence, even the honest model's calibration-in-the-large is off by about 0.27, so a model moved to a new site needs recalibration even when its ranking holds. The site classifier's AUC of 0.999 mirrors Zech's finding that images reveal their hospital.

### Script 2: subgroup audit and label choice

```python
"""SYNTHETIC DATA ONLY. NOT MEDICAL ADVICE. Groups, features and rates are
invented to show a method; they do not describe any real population.

Two groups with the same distribution of true illness. G2 has less access to
care, so it has fewer prior visits at the same severity, and its illness is
recorded (diagnosed) less often. One measurement is also noisier for G2.
Audit the model per group, against the recorded label and against the true
condition, with bootstrap intervals. Then change the training label."""
import os

# Many small fits: extra threads mostly add overhead, and on a shared machine
# they can slow things down a lot. Must be set before numpy is imported.
for var in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ.setdefault(var, "4")

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

rng = np.random.default_rng(0)


def make_population(n, frac_g2=0.15):
    g2 = rng.random(n) < frac_g2
    severity = rng.normal(0, 1, n)                    # same distribution in both groups
    ill = severity > 0.85                             # about 20% truly ill
    x1 = severity + rng.normal(0, 0.6, n)
    x2 = severity + rng.normal(0, np.where(g2, 1.2, 0.6), n)   # noisier for G2
    access = np.where(g2, 0.5, 1.0)
    visits = rng.poisson(np.exp(0.5 + 0.7 * severity) * access)   # utilization
    recorded = ill & (rng.random(n) < np.where(g2, 0.55, 0.9))   # under-recorded in G2
    X = np.column_stack([x1, x2, np.log1p(visits)])
    return X, ill.astype(int), recorded.astype(int), g2


def boot_ci(fn, *arrays, reps=1000):
    """Percentile bootstrap 95% interval of fn over resampled rows."""
    n = len(arrays[0])
    vals = []
    for _ in range(reps):
        i = rng.integers(0, n, n)
        vals.append(fn(*(a[i] for a in arrays)))
    return np.percentile(vals, [2.5, 97.5])


def sens(y, flag):
    return flag[y == 1].mean()


X_tr, ill_tr, rec_tr, g_tr = make_population(20000)
X_te, ill_te, rec_te, g_te = make_population(4000)
print(f"test set: {np.sum(~g_te)} in G1, {np.sum(g_te)} in G2")
print(f"true illness rate   G1 {ill_te[~g_te].mean():.1%}   G2 {ill_te[g_te].mean():.1%}")
print(f"recorded label rate G1 {rec_te[~g_te].mean():.1%}   G2 {rec_te[g_te].mean():.1%}")

configs = [("recorded label", rec_tr, [0, 1, 2]),
           ("true condition", ill_tr, [0, 1, 2]),
           ("true condition, no visits", ill_tr, [0, 1])]
print(f"\n{'trained on':<26} {'group':<5} {'AUC':>5} {'95% CI':>14} {'sens.':>6} "
      f"{'95% CI':>14} {'flagged':>8} {'mean risk':>9}")
for train_label, y_train, cols in configs:
    model = LogisticRegression().fit(X_tr[:, cols], y_train)   # group is NOT an input
    p = model.predict_proba(X_te[:, cols])[:, 1]
    flag = p >= np.quantile(p, 0.80)                  # flag the top 20% for follow-up
    for gname, mask in [("G1", ~g_te), ("G2", g_te)]:
        y_g, p_g, f_g = ill_te[mask], p[mask], flag[mask]   # judged against the truth
        a_lo, a_hi = boot_ci(roc_auc_score, y_g, p_g)
        s_lo, s_hi = boot_ci(sens, y_g, f_g)
        print(f"{train_label:<26} {gname:<5} {roc_auc_score(y_g, p_g):5.3f} [{a_lo:.3f}, {a_hi:.3f}] "
              f"{sens(y_g, f_g):6.2f} [{s_lo:.2f}, {s_hi:.2f}] {f_g.mean():8.1%} {p_g.mean():9.3f}")
    if train_label == "recorded label":
        coef_note = ", ".join(f"{c:.2f}" for c in model.coef_[0])
        audit = [roc_auc_score(rec_te[m], p[m]) for m in (~g_te, g_te)]
        print(f"{'':<26} (coefficients x1, x2, log visits: {coef_note}; audited against the"
              f" recorded label instead, AUC G1 {audit[0]:.3f}, G2 {audit[1]:.3f})")
print(f"\ntrue illness rate: G1 {ill_te[~g_te].mean():.3f}, G2 {ill_te[g_te].mean():.3f}")
```

```text
test set: 3374 in G1, 626 in G2
true illness rate   G1 20.6%   G2 20.0%
recorded label rate G1 18.8%   G2 11.5%

trained on                 group   AUC         95% CI  sens.         95% CI  flagged mean risk
recorded label             G1    0.967 [0.961, 0.972]   0.82 [0.79, 0.85]    21.2%     0.175
recorded label             G2    0.935 [0.915, 0.954]   0.54 [0.46, 0.63]    13.7%     0.132
                           (coefficients x1, x2, log visits: 1.33, 0.96, 1.31; audited against the recorded label instead, AUC G1 0.956, G2 0.898)
true condition             G1    0.967 [0.962, 0.972]   0.81 [0.79, 0.84]    21.0%     0.206
true condition             G2    0.933 [0.912, 0.952]   0.57 [0.48, 0.65]    14.5%     0.163
true condition, no visits  G1    0.961 [0.955, 0.967]   0.78 [0.75, 0.81]    20.3%     0.201
true condition, no visits  G2    0.915 [0.891, 0.937]   0.63 [0.55, 0.72]    18.5%     0.199

true illness rate: G1 0.206, G2 0.200
```

True illness is equally common in both groups (20.6% and 20.0%), but G2's recorded rate is 11.5%. Trained on the recorded label, the model catches 82% of truly ill G1 patients but 54% of G2, and underestimates G2's risk (0.132 against 0.200). Group membership is not an input; the disparity enters through the label and through `visits`, which reflects access as well as illness. Training on the true condition fixes G1's calibration but barely helps G2's sensitivity (0.57), because `visits` still encodes access. Dropping it raises G2 to 0.63 and fixes calibration for both groups, at a small cost for G1 (0.78). The remaining gap comes from G2's noisier measurement, which no relabelling fixes. Note the interval widths: with 626 patients, G2's sensitivity is uncertain by about $\pm 0.08$. Audits need enough people in each group. Auditing against the recorded label instead gives a different picture again (AUC 0.956 against 0.898, a larger apparent gap than the 0.967 against 0.935 measured against the truth): the label used for evaluation is a choice too.

### Script 3: deployment metrics

```python
"""SYNTHETIC DATA ONLY. NOT MEDICAL ADVICE. All rates are invented.

What a fixed model means at the bedside: (1) the same alert threshold at
different prevalences, (2) net benefit (decision curves) for a calibrated
and a miscalibrated version, (3) calibration drift over time and the effect
of recalibrating on recent data."""
import os

# Many small fits: extra threads mostly add overhead, and on a shared machine
# they can slow things down a lot. Must be set before numpy is imported.
for var in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ.setdefault(var, "4")

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

rng = np.random.default_rng(0)
COEF = np.array([1.0, 0.7, 0.5])


def cohort(n, intercept, coef=COEF):
    X = rng.normal(0, 1, (n, 3))
    p = 1 / (1 + np.exp(-(intercept + X @ coef)))
    return X, (rng.random(n) < p).astype(int)


# ---- 1. one model, one threshold, three prevalences -------------------------
X_dev, y_dev = cohort(20000, -2.6)
model = LogisticRegression().fit(X_dev, y_dev)
thr = np.quantile(model.predict_proba(X_dev)[:, 1], 0.90)   # alert on top 10% at development
print(f"development prevalence {y_dev.mean():.1%}; alert threshold {thr:.3f}\n")
print(f"{'prevalence':>10} {'AUC':>6} {'alerts/100':>11} {'PPV':>6} {'sensitivity':>12} {'alerts per true case':>21}")
for intercept in [-4.6, -2.6, -1.2]:
    X, y = cohort(20000, intercept)
    p = model.predict_proba(X)[:, 1]
    alert = p >= thr
    tp = np.sum(alert & (y == 1))
    print(f"{y.mean():10.1%} {roc_auc_score(y, p):6.3f} {100 * alert.mean():11.1f} "
          f"{tp / alert.sum():6.2f} {tp / y.sum():12.2f} {alert.sum() / tp:21.1f}")

# ---- 2. decision curves -------------------------------------------------------
# Net benefit at threshold probability pt (Vickers and Elkin, 2006):
#   NB = TP/n - FP/n * pt / (1 - pt)
# pt encodes how many false alarms one true case is worth: at pt = 0.2, 4.
X, y = cohort(20000, -2.6)
p_cal = model.predict_proba(X)[:, 1]
logit = np.log(p_cal / (1 - p_cal))
p_over = 1 / (1 + np.exp(-(logit + 1.5)))             # same ranking, risks inflated
print(f"\nnet benefit per 100 patients (prevalence {y.mean():.1%}; AUC identical for both models)")
print(f"{'pt':>5} {'treat none':>11} {'treat all':>10} {'calibrated':>11} {'overestimating':>15}")
for pt in [0.05, 0.10, 0.20, 0.30, 0.50]:
    w = pt / (1 - pt)
    nb = lambda q: (np.sum((q >= pt) & (y == 1)) - w * np.sum((q >= pt) & (y == 0))) / len(y)
    nb_all = y.mean() - w * (1 - y.mean())
    print(f"{pt:5.2f} {0:11.2f} {100 * nb_all:10.2f} {100 * nb(p_cal):11.2f} {100 * nb(p_over):15.2f}")

# ---- 3. temporal drift --------------------------------------------------------
# Each quarter the baseline risk creeps up and one predictor matters less
# (think of a change in practice or in how a test is ordered).
print("\nquarterly monitoring of the frozen model, and of a copy recalibrated each quarter")
print("on the previous quarter's data (intercept and slope of the logit only):")
print(f"{'quarter':>7} {'rate':>6} {'AUC':>6} {'frozen: obs-pred':>17} {'recalibrated: obs-pred':>23}")
prev = None
for q in range(8):
    coef_q = COEF * np.array([1, 1 - 0.08 * q, 1])
    Xq, yq = cohort(5000, -2.6 + 0.15 * q, coef_q)
    pq = model.predict_proba(Xq)[:, 1]
    lq = np.log(pq / (1 - pq))[:, None]
    line = f"{q + 1:7d} {yq.mean():6.1%} {roc_auc_score(yq, pq):6.3f} {yq.mean() - pq.mean():+17.3f}"
    if prev is not None:
        recal = LogisticRegression(C=1e6).fit(*prev)       # fit on last quarter only
        line += f" {yq.mean() - recal.predict_proba(lq)[:, 1].mean():+23.3f}"
    print(line)
    prev = (lq, yq)
```

```text
development prevalence 11.5%; alert threshold 0.286

prevalence    AUC  alerts/100    PPV  sensitivity  alerts per true case
      2.0%  0.809        10.1   0.10         0.49                  10.3
     11.6%  0.803         9.9   0.45         0.38                   2.2
     28.4%  0.791         9.8   0.73         0.25                   1.4

net benefit per 100 patients (prevalence 11.4%; AUC identical for both models)
   pt  treat none  treat all  calibrated  overestimating
 0.05        0.00       6.77        7.80            7.17
 0.10        0.00       1.59        5.57            3.63
 0.20        0.00     -10.71        3.11           -1.50
 0.30        0.00     -26.52        1.66           -4.68
 0.50        0.00     -77.13        0.73           -6.88

quarterly monitoring of the frozen model, and of a copy recalibrated each quarter
on the previous quarter's data (intercept and slope of the logit only):
quarter   rate    AUC  frozen: obs-pred  recalibrated: obs-pred
      1  11.5%  0.811            -0.001
      2  13.6%  0.799            +0.020                  +0.021
      3  14.0%  0.779            +0.025                  +0.004
      4  15.4%  0.782            +0.037                  +0.012
      5  17.5%  0.801            +0.054                  +0.017
      6  18.7%  0.787            +0.071                  +0.018
      7  20.2%  0.767            +0.090                  +0.019
      8  23.1%  0.768            +0.114                  +0.024
```

At 2% prevalence the same model, with almost the same AUC, produces 10 alerts per true case found; at 28%, 1.4. Sensitivity also falls as prevalence rises, because the threshold fixed at development flags a roughly constant 10% of patients. In the decision curve, the overestimating copy has the same AUC, yet at a 20% action threshold its net benefit is negative, worse than treating nobody, while the calibrated copy is positive. In monitoring, the frozen model's observed-minus-predicted gap grows every quarter while AUC moves only a little, so tracking AUC alone would miss the drift. Recalibrating on the previous quarter keeps the gap near 0.02, a lag that remains because the drift continues.

### Script 4: random versus family splits, and finding the best

```python
"""SYNTHETIC DATA ONLY. 'Compounds' are random vectors grouped into families
(think scaffolds or chemical systems); the property is an invented function.

Two questions a discovery paper must answer:
1. Does the model work on new families, or only on new members of known ones?
2. Can it rank candidates better than anything it was trained on?"""
import os

# Many small fits: extra threads mostly add overhead, and on a shared machine
# they can slow things down a lot. Must be set before numpy is imported.
for var in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"]:
    os.environ.setdefault(var, "4")

import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from sklearn.model_selection import GroupKFold, KFold

rng = np.random.default_rng(0)
F, PER, D = 80, 30, 12                         # families, members per family, descriptors
centres = rng.normal(0, 1, (F, D))
fam = np.repeat(np.arange(F), PER)
X = centres[fam] + rng.normal(0, 0.25, (F * PER, D))
w = rng.normal(0, 1, D)
fam_offset = rng.normal(0, 0.8, F)             # family effect not explained by descriptors
y = X @ w * 0.6 + np.sin(2 * X[:, 0]) + fam_offset[fam] + rng.normal(0, 0.3, len(fam))
print(f"{len(y)} compounds in {F} families, {D} descriptors")


def cv_r2(model, splitter):
    scores = []
    groups = fam if isinstance(splitter, GroupKFold) else None
    for tr, te in splitter.split(X, y, groups):
        scores.append(r2_score(y[te], model.fit(X[tr], y[tr]).predict(X[te])))
    return np.mean(scores), np.std(scores)


models = {"ridge": Ridge(alpha=1.0),
          "random forest": RandomForestRegressor(300, min_samples_leaf=2, n_jobs=4, random_state=0)}
print(f"\n{'model':<14} {'random 5-fold R2':>17} {'family-grouped 5-fold R2':>25}")
for name, m in models.items():
    r, rs = cv_r2(m, KFold(5, shuffle=True, random_state=0))
    g, gs = cv_r2(m, GroupKFold(5))
    print(f"{name:<14} {r:10.2f} ({rs:.2f}) {g:17.2f} ({gs:.2f})")

# ---- Prospective-style test: hide the best 5% and see if they can be found.
top = y >= np.quantile(y, 0.95)
train = ~top & (rng.random(len(y)) < 0.7)              # known, ordinary compounds
pool = ~train                                          # unseen: all top + 30% of others
print(f"\ntraining set: {train.sum()} ordinary compounds, best property {y[train].max():.2f}")
print(f"candidate pool: {pool.sum()} unseen compounds, {top.sum()} of them better than every training compound")
base_rate = top[pool].mean()
print(f"\n{'model':<14} {'max predicted':>14} {'hits in top 50':>15} {'enrichment vs random':>21}")
for name, m in models.items():
    pred = m.fit(X[train], y[train]).predict(X[pool])
    chosen = np.argsort(-pred)[:50]
    hits = top[pool][chosen].sum()
    print(f"{name:<14} {pred.max():14.2f} {hits:15d} {hits / 50 / base_rate:21.1f}x")
print(f"{'random pick':<14} {'':>14} {50 * base_rate:15.1f} {1.0:21.1f}x")
```

```text
2400 compounds in 80 families, 12 descriptors

model           random 5-fold R2  family-grouped 5-fold R2
ridge                0.84 (0.01)              0.74 (0.06)
random forest        0.93 (0.01)              0.54 (0.05)

training set: 1587 ordinary compounds, best property 3.48
candidate pool: 813 unseen compounds, 120 of them better than every training compound

model           max predicted  hits in top 50  enrichment vs random
ridge                    6.74              47                   6.4x
random forest            3.10              27                   3.7x
random pick                               7.4                   1.0x
```

A random split rates the random forest above ridge (0.93 against 0.84), and a family-grouped split reverses the order (0.54 against 0.74), because the forest memorises family effects that do not carry to new families. In the discovery test, the forest's highest prediction (3.10) is below the best training value (3.48), since a tree predicts averages of training labels. Ridge extrapolates and finds 47 of the hidden best compounds in its top 50. The synthetic property is mostly linear, which favours ridge; with real data the lesson is to evaluate on the split that matches the claim, not that linear models are better.

## Common mistakes

- **Reporting internal validation as if it were external.** Name the rung: random split, grouped, temporal, external, prospective, randomised.
- **Treating external validation as proof against shortcuts.** If the shortcut also holds at the new site, it passes (script 1, site D).
- **AUC without prevalence, PPV, calibration and alert burden** at the deployment site.
- **Using a convenient proxy as the label** (cost, recorded diagnosis, prescriptions) without checking how it differs across groups.
- **Subgroup results without intervals**, or groups too small to estimate.
- **Random splits for molecules, materials or proteins**, when the claim is about new families.
- **Confusing "predicted stable" or "active in a screen" with "discovered".** Synthesis, novelty and function are separate claims.
- **Reading press releases instead of the primary endpoint and its confidence interval.**

## Exercises

1. [warm-up] Place each of these on the evidence ladder of section 1: CASP14, GraphCast's 2018 test, the Epic Sepsis Model study, MASAI.
2. [warm-up] A test has sensitivity 0.9 and specificity 0.9. Compute its PPV at prevalence 1% and 20%.
3. [core] In script 1, make site C's portable rates 0.10 among the ill and 0.80 among the well (a reversed shortcut). Predict the all-features AUC, then run it.
4. [core] In script 1, recalibrate the physiology model for site C using 500 labelled site-C patients (fit intercept and slope on the logit). How much of the remaining calibration error goes away?
5. [core] In script 2, shrink G2 to 5% of the population. How wide does the interval on G2's sensitivity become? How many G2 patients would you need for $\pm 0.05$?
6. [core] In script 2, use group-specific thresholds chosen so both groups have sensitivity 0.8 against the true condition. What happens to the fraction flagged in each group? Which fairness criterion did you just impose (see L4-05)?
7. [core] In script 3, add the decision curve for a model with lower AUC but perfect calibration. Can it have higher net benefit than a better-ranking, miscalibrated model?
8. [stretch] In script 4, add a $k$-nearest-neighbour regressor and a gradient-boosting model. Rank all four models under both splits and in the discovery test.
9. [stretch] Pick one recent AI-for-science paper. Write a one-page assessment: rung, evaluator, comparator, endpoint, what was shown, what is claimed beyond it.

### Hints and answers

1. CASP14: rung 3 (blind). GraphCast 2018 test: rung 2 to 3 (later period, scored after the fact against held-out data). Epic study: rung 4 (external, independent). MASAI: rung 5.
2. At 1%: TP = 0.009, FP = 0.099, PPV ≈ 0.08. At 20%: TP = 0.18, FP = 0.08, PPV ≈ 0.69.
3. Below 0.5: in our run it was 0.350, with a negative calibration slope, because the model's strongest cue now points the wrong way. The physiology-only model is unaffected (0.794).
4. Most of the calibration-in-the-large and slope error, since physiology transfers; AUC does not change, because recalibration is monotone.
5. Roughly $\sqrt{0.15/0.05} \approx 1.7$ times wider. For $\pm 0.05$ at sensitivity near 0.6 you need about $0.6 \cdot 0.4 / 0.025^2 \approx 380$ truly ill G2 patients, about 1,900 G2 patients at 20% prevalence.
6. G2 needs a lower threshold, so more G2 patients are flagged. Equal sensitivity across groups is "equal opportunity".
7. Yes. Net benefit depends on calibration at the chosen threshold, not only on ranking.
8. Expect neighbours and boosting to look good on random splits and drop under family splits, with neither exceeding the training maximum in the discovery test.
9. Use the questions at the end of section 1.

## Check your understanding

**1. Why is weather forecasting a stronger case for ML than most clinical prediction?** Its verification is automatically prospective and large-scale, the training data is consistent over decades, the metrics are agreed, and an operational centre has adopted ML models alongside physics. The remaining weaknesses (extremes, dependence on data assimilation) are known and measured.

**2. AlphaFold's high-confidence predictions are very accurate. Why do experts still call them hypotheses?** Even above pLDDT 90, about 10% of residues were more than 2 Å off in Terwilliger et al.'s comparison, side chains often differ, and the models are static. Detailed mechanistic or drug-design conclusions need experimental confirmation.

**3. An imaging model has AUC 0.95 internally and 0.93 at one external hospital. Is it free of shortcuts?** No. A shortcut that also holds at the external hospital (same device habits, similar prevalence) passes external validation, as DeGrave et al. warned and script 1's site D shows. Check what the model relies on, and test where the shortcut is broken.

**4. What did MASAI show about interval cancers?** Non-inferiority: the rate ratio was 0.88 with a 95% interval from 0.65 to 1.18, inside the pre-set margin. It did not show a reduction, because the interval includes 1. It also showed higher sensitivity at the same specificity, with far fewer radiologist readings.

**5. Why might a model trained on recorded diagnoses underserve a group even if group is not an input?** If that group is diagnosed less often at the same illness level, the labels understate its risk, and features tied to access to care (visits, cost) carry the same bias into the model. Script 2 shows both, and that changing the label alone may not remove the feature's effect.

## Further reading

- **John Jumper et al., "Highly accurate protein structure prediction with AlphaFold"** (2021, Nature 596:583-589; free via PubMed Central). https://doi.org/10.1038/s41586-021-03819-2
  *Read for:* the AlphaFold2 method, CASP14 results and stated limitations. *Level:* advanced.
- **Thomas C. Terwilliger et al., "AlphaFold predictions are valuable hypotheses and accelerate but do not replace experimental structure determination"** (2024, Nature Methods 21:110-116; free via PubMed Central). https://doi.org/10.1038/s41592-023-02087-4
  *Read for:* an independent, quantitative check of AlphaFold against experimental maps. *Level:* intermediate.
- **Remi Lam et al., "Learning skillful medium-range global weather forecasting"** (2023, Science 382:1416-1421; arXiv:2212.12794 is free). https://arxiv.org/abs/2212.12794
  *Read for:* GraphCast, a graph neural network evaluated against operational forecasts. *Level:* advanced.
- **Ilan Price et al., "Probabilistic weather forecasting with machine learning"** (2025, Nature 637:84-90; free via PubMed Central). https://doi.org/10.1038/s41586-024-08252-9
  *Read for:* GenCast, ensemble evaluation, and a frank limitations section. *Level:* advanced.
- **Anthony K. Cheetham, Ram Seshadri, "Artificial Intelligence Driving Materials Discovery? Perspective on the Article: Scaling Deep Learning for Materials Discovery"** (2024, Chemistry of Materials 36:3490-3495; free via PubMed Central). https://doi.org/10.1021/acs.chemmater.4c00643
  *Read for:* how domain experts test a large discovery claim. *Level:* intermediate.
- **Josh Leeman et al., "Challenges in High-Throughput Inorganic Materials Prediction and Autonomous Synthesis"** (2024, PRX Energy 3:011002; open access). https://doi.org/10.1103/PRXEnergy.3.011002
  *Read for:* what "a new material" requires, and the A-Lab critique. *Level:* intermediate.
- **John R. Zech et al., "Variable generalization performance of a deep learning model to detect pneumonia in chest radiographs: A cross-sectional study"** (2018, PLOS Medicine 15(11):e1002683; open access). https://doi.org/10.1371/journal.pmed.1002683
  *Read for:* the clearest worked example of site shortcuts in imaging. *Level:* intro.
- **Andrew Wong et al., "External Validation of a Widely Implemented Proprietary Sepsis Prediction Model in Hospitalized Patients"** (2021, JAMA Internal Medicine 181(8):1065-1070; open access via PMC). https://doi.org/10.1001/jamainternmed.2021.2626
  *Read for:* what independent external validation of a deployed model looks like. *Level:* intro.
- **Ziad Obermeyer, Brian Powers, Christine Vogeli, Sendhil Mullainathan, "Dissecting racial bias in an algorithm used to manage the health of populations"** (2019, Science 366:447-453; may be paywalled). https://doi.org/10.1126/science.aax2342
  *Read for:* how a proxy label creates bias, found by auditing outcomes by group. *Level:* intro.
- **Jessie Gommers et al., "Interval cancer, sensitivity, and specificity comparing AI-supported mammography screening with standard double reading without AI in the MASAI study"** (2026, The Lancet 407:505-514; may be paywalled). https://doi.org/10.1016/S0140-6736(25)02464-X
  *Read for:* a large randomised trial of clinical AI and how to read a non-inferiority result. *Level:* intermediate.
- **Robert Geirhos et al., "Shortcut learning in deep neural networks"** (2020, Nature Machine Intelligence 2:665; arXiv:2004.07780). https://doi.org/10.1038/s42256-020-00257-z
  *Read for:* a general account of shortcut learning across fields. *Level:* intro.
- **Gary S. Collins et al., "TRIPOD+AI statement: updated guidance for reporting clinical prediction models that use regression or machine learning methods"** (2024, BMJ 385:e078378; open access). https://doi.org/10.1136/bmj-2023-078378
  *Read for:* the checklist a clinical prediction paper should satisfy. *Level:* intermediate.

**Other sources cited in the text** (checked October 2026): Varadi et al., "AlphaFold Protein Structure Database in 2024" (Nucleic Acids Research 52:D368-D375), https://doi.org/10.1093/nar/gkad1011 · Royal Swedish Academy of Sciences, Nobel Prize in Chemistry 2024 press release, https://www.nobelprize.org/prizes/chemistry/2024/press-release/ · Abramson et al., "Accurate structure prediction of biomolecular interactions with AlphaFold 3" (Nature 630:493-500, 2024), https://doi.org/10.1038/s41586-024-07487-w, code at https://github.com/google-deepmind/alphafold3 · Rasp et al., "WeatherBench 2" (JAMES 2024; arXiv:2308.15560), https://arxiv.org/abs/2308.15560 · ECMWF, "ECMWF's AI forecasts become operational" (25 February 2025), https://www.ecmwf.int/node/29308 · ECMWF, "ECMWF's ensemble AI forecasts become operational" (1 July 2025), https://www.ecmwf.int/en/about/media-centre/news/2025/ecmwfs-ensemble-ai-forecasts-become-operational · ECMWF Newsletter 187, "Implementation of AIFS v2" (2026), https://www.ecmwf.int/en/newsletter/187/news/implementation-aifs-v2 · Sun, Hassanzadeh, Zand, Chattopadhyay, Weare, Abbot, "Can AI weather models predict out-of-distribution gray swan tropical cyclones?" (PNAS 2025), https://doi.org/10.1073/pnas.2420914122 · Kochkov et al., "Neural general circulation models for weather and climate" (Nature 632:1060-1066, 2024), https://doi.org/10.1038/s41586-024-07744-y · Merchant et al., "Scaling deep learning for materials discovery" (Nature 624:80-85, 2023), https://doi.org/10.1038/s41586-023-06735-9 · Szymanski et al., A-Lab paper (Nature 624:86-91, 2023), https://doi.org/10.1038/s41586-023-06734-w, and author correction (Nature, January 2026), https://doi.org/10.1038/s41586-025-09992-y · Riebesell et al., "A framework to evaluate machine learning crystal stability predictions" (Nature Machine Intelligence 7, 2025), https://doi.org/10.1038/s42256-025-01055-1 · Stokes et al., "A Deep Learning Approach to Antibiotic Discovery" (Cell 180:688-702, 2020), https://doi.org/10.1016/j.cell.2020.01.021 · Jayatunga et al., "How successful are AI-discovered drugs in clinical trials?" (Drug Discovery Today 29:104009, 2024), https://doi.org/10.1016/j.drudis.2024.104009 · Xu et al., "A generative AI-discovered TNIK inhibitor for idiopathic pulmonary fibrosis: a randomized phase 2a trial" (Nature Medicine 31:2602-2610, 2025), https://doi.org/10.1038/s41591-025-03743-2 · ClinicalTrials.gov NCT07687459, https://clinicaltrials.gov/study/NCT07687459 · Gulshan et al. (JAMA 316:2402-2410, 2016), https://doi.org/10.1001/jama.2016.17216 · Abràmoff et al., pivotal trial of an autonomous AI diagnostic system (npj Digital Medicine 1:39, 2018), https://doi.org/10.1038/s41746-018-0040-6 · FDA De Novo database, DEN180001, https://www.accessdata.fda.gov/scripts/cdrh/cfdocs/cfpmn/denovo.cfm?id=DEN180001 · Beede et al., "A Human-Centered Evaluation of a Deep Learning System Deployed in Clinics for the Detection of Diabetic Retinopathy" (CHI 2020), https://research.google/pubs/a-human-centered-evaluation-of-a-deep-learning-system-deployed-in-clinics-for-the-detection-of-diabetic-retinopathy/ · Lång et al., MASAI clinical safety analysis (Lancet Oncology 24:936-944, 2023), https://doi.org/10.1016/S1470-2045(23)00298-X · Roberts et al., "Common pitfalls and recommendations for using machine learning to detect and prognosticate for COVID-19 using chest radiographs and CT scans" (Nature Machine Intelligence 3:199, 2021), https://doi.org/10.1038/s42256-021-00307-0 · DeGrave, Janizek, Lee, "AI for radiographic COVID-19 detection selects shortcuts over signal" (Nature Machine Intelligence 3:610, 2021), https://doi.org/10.1038/s42256-021-00338-7 · Vickers, Elkin, "Decision curve analysis" (Medical Decision Making 26:565-574, 2006), https://doi.org/10.1177/0272989X06295361 · Seyyed-Kalantari et al., "Underdiagnosis bias of artificial intelligence algorithms applied to chest radiographs in under-served patient populations" (Nature Medicine 27:2176-2182, 2021), https://doi.org/10.1038/s41591-021-01595-0 · Gichoya et al., "AI recognition of patient race in medical imaging: a modelling study" (Lancet Digital Health 4:e406-e414, 2022), https://doi.org/10.1016/S2589-7500(22)00063-2 · FDA, "List of Artificial Intelligence-Enabled Medical Devices" (content current as of 4 September 2026), https://www.fda.gov/medical-devices/artificial-intelligence-enabled-medical-devices/list-artificial-intelligence-enabled-medical-devices · Congressional Research Service, "FDA Regulation of AI-Enabled Devices" (IF13245, 10 June 2026), https://www.congress.gov/crs-product/IF13245 · FDA, "Marketing Submission Recommendations for a Predetermined Change Control Plan for Artificial Intelligence-Enabled Device Software Functions" (final guidance, page dated August 2025), https://www.fda.gov/regulatory-information/search-fda-guidance-documents/marketing-submission-recommendations-predetermined-change-control-plan-artificial-intelligence · Regulation (EU) 2026/1744 (Digital Omnibus on AI), https://eur-lex.europa.eu/eli/reg/2026/1744/oj
