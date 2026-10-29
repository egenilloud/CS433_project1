# CS-433 Project 1 — Predicting coronary heart disease (MICHD) from the BRFSS survey

Binary classification of MICHD from the BRFSS 2015 health survey, using **NumPy only**.
Final model: L2-regularized logistic regression trained with Newton's method, with
codebook-based preprocessing, pairwise interactions between risk factors, and a
decision threshold tuned for F1 by 5-fold stratified cross-validation.

**Cross-validated F1 (5 folds): 0.4315 ± 0.0048**

Team: Erine Genilloud, Lou Colombies, Chloé Achtari

## Reproducing our submission

1. Requirements: Python 3 and NumPy (matplotlib only for the notebooks in `notebooks/`).
2. Download the data from the AIcrowd challenge and place the three files, with
   unchanged names, in `data/dataset/` (the data is not included in the repository):
   ```
   data/dataset/x_train.csv
   data/dataset/y_train.csv
   data/dataset/x_test.csv
   ```
3. From the repository root, run:
   ```
   python run.py
   ```
   or run all the cells of `run.ipynb` (same pipeline, with explanations).
   Both write `submission.csv`, which is exactly the file of our final AIcrowd
   submission. Everything is deterministic (fixed seed = 1). Runtime: about 4 min,
   most of which is loading the CSV files.

## Repository structure

| File | Content |
|---|---|
| `run.py` | Final pipeline: load → preprocess → 5-fold CV (threshold choice) → retrain on all data → `submission.csv` |
| `run.ipynb` | Same pipeline as `run.py`, as a commented notebook |
| `implementations.py` | The 6 required methods (GD, SGD, least squares, ridge, logistic and regularized logistic regression) |
| `preprocessing.py` | Data cleaning and feature engineering (see below) |
| `newton.py` | Regularized logistic regression trained with Newton's method |
| `cross_validation.py` | Stratified K-fold CV, F1 score, F1-optimal threshold, final training |
| `svm.py` | Linear SVM (subgradient descent and dual coordinate ascent), explored model |
| `helpers.py` | Data loading and submission writing (provided by the course) |
| `experiments_features.py` | Ablation study of the feature engineering and choice of λ → `results/feature_ablation.md` |
| `experiments_baselines.py` | Comparison with baseline models → `results/model_comparison.md` |
| `results/` | Result tables used in the report |
| `notebooks/data_visualization.ipynb` | Exploratory data analysis (missing values) |
| `notebooks/newton_run.ipynb` | Newton's method on a 80/20 train/validation split |
| `notebooks/svm_run.ipynb` | Linear SVM on the same split (comparison with Newton) |
| `report/` | Report (LaTeX source and PDF) |

## Method

**Preprocessing** (`preprocessing.py`). All statistics (medians, clipping bounds,
means/stds, one-hot categories) are computed on the training folds only.
1. Codebook cleaning: survey codes for "don't know / refused" (7, 9, 77, 99, …) → NaN,
   "none" (88) → 0; yes/no answers recoded to 1/0.
2. Only 65 relevant variables are kept (dictionary `RULES`). Administrative columns,
   sampling weights and `HAREHAB1` (only asked to people who already had a heart
   attack: target leakage) are dropped.
3. Skip patterns: when a question was skipped because of a previous answer, the NaN is
   replaced by its logical value (e.g. no blood-pressure medication if not hypertensive).
4. Columns with > 40 % missing values dropped; outliers clipped at the 1st/99th
   percentiles; median (or mode) imputation, with a "was missing" indicator;
   one-hot encoding of nominal variables; standardization.
5. Feature engineering: one-hot of age group, BMI category and general health,
   age² and BMI², four optional survey modules, and all pairwise products of
   15 cardiovascular risk factors.

**Model.** Logistic regression with L2 penalty λ‖w‖² (λ = 1e-4, bias not penalized),
optimized with Newton's method (converges in ~10 iterations).

**Class imbalance.** Only ~9 % of the respondents are positive, so the default
threshold of 0.5 gives a poor F1. The threshold maximizing F1 is chosen on the
out-of-fold scores of a 5-fold stratified cross-validation, then the model is
retrained on the whole training set.

## Tests

The 6 functions of `implementations.py` pass the course's public tests
(`pytest --github_link <repo-url> .` from `projects/project1/grading_tests`).

## Use of AI tools

We used an AI assistant (Claude, Anthropic) to help write parts of the code
(preprocessing, cross-validation, Newton's method, SVM, experiment scripts),
the comments and docstrings, and this README. All modelling choices, results and
the report were checked and are understood by the team.
