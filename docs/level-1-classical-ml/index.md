---
title: "Level 1: Classical machine learning"
---

Learning from data without deep networks: evaluation done honestly, linear and logistic models, trees and boosting, unsupervised learning, and the data and feature work that decides most real projects. These habits carry into every later level.

| ID | Module | Prerequisites | Time | Scripts |
|---|---|---|---|---|
| L1-01 | The ML workflow and evaluation | L0-01, L0-04 | 5 to 7 hours | 5 |
| L1-02 | Linear and Logistic Models | L0-02, L0-03, L0-04, L0-05, L1-01 | 6 to 8 hours | 4 |
| L1-03 | Trees and Ensembles | L1-01, L1-02 | 6 to 8 hours | 5 |
| L1-04 | Unsupervised Learning | L0-02, L0-04, L1-01 | 5 to 6 hours | 5 |
| L1-05 | Data and Features | L1-01, L1-02, L1-03 | 5 to 6 hours | 5 |

## What each module teaches

### L1-01: The ML workflow and evaluation

- Frame a prediction problem as a decision with a baseline and a metric that reflects the cost of errors.
- Split data so the evaluation answers the deployment question, including grouped and time-ordered data.
- Detect and prevent the common forms of data leakage.
- Choose and interpret classification metrics (precision, recall, ROC and PR curves, calibration) and regression metrics.

### L1-02: Linear and Logistic Models

- Derive the least-squares solution, fit it in closed form and by gradient descent, and explain why feature scaling changes how fast gradient descent converges.
- Implement ridge (closed form) and lasso (coordinate descent) from scratch, match scikit-learn, and choose the penalty by cross-validation.
- Explain least squares, ridge, lasso, and logistic regression as maximum-likelihood or MAP estimates under explicit assumptions.
- Implement binary logistic and multiclass softmax regression from scratch, and explain what goes wrong on linearly separable data.

### L1-03: Trees and Ensembles

- Build a classification tree from scratch, compute Gini and entropy splits by hand, and explain why deep trees overfit.
- Derive why averaging trees reduces variance, and use it to explain bagging, random forests, and `max_features`.
- Implement gradient boosting from scratch for squared loss and for log-loss with Newton leaf values, as in XGBoost.
- Choose and tune a tree ensemble for a tabular problem with a sensible default recipe.

### L1-04: Unsupervised Learning

- Derive and implement k-means and EM for Gaussian mixtures, and explain why each improves monotonically yet gets stuck in local optima.
- Derive PCA from the SVD, compute the variance a projection keeps, and predict when unscaled features will dominate it.
- Choose a hierarchical linkage for the cluster shapes you expect, and recognize chaining.
- Read a t-SNE or UMAP plot without over-reading it.

### L1-05: Data and Features

- Audit a dataset for how it was collected and labeled, and estimate how label errors affect training and evaluation.
- Encode categorical variables and scale numeric ones without leaking the target, and say which models need which treatment.
- Classify a missing-data problem as MCAR, MAR or MNAR, and choose an imputation strategy whose bias you can predict.
- Handle class imbalance by fixing the decision threshold and the metric before reaching for resampling, and explain what resampling does to probabilities.
