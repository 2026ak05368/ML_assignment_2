# Income Classification — Model Comparison (ML Assignment 2)

**Student:** Mitz
**Course:** M.Tech AIML, BITS Pilani WILP
**Submission date:** 26-Oct-2026

## 1. Problem Statement

This project predicts whether an individual's annual income exceeds $50,000 based
on demographic and employment attributes recorded in a census survey. This is a
binary classification problem with practical relevance to socioeconomic research,
targeted public policy, and resource allocation — organizations and governments
routinely need to estimate income brackets from easily observable attributes
(age, occupation, education, marital status, hours worked, etc.) when direct
income data isn't available or is self-reported unreliably. The goal of this
assignment is to build, compare, and evaluate five different classification
algorithms on this task, identify which model generalizes best under the
dataset's class imbalance (roughly 75% low-income / 25% high-income), and deploy
the best-performing pipeline as an interactive web app that scores new data
without retraining.

## 2. Dataset Description

- **Name:** UCI Adult (Census Income) Dataset
- **Source:** UCI Machine Learning Repository, retrieved via a public GitHub
  mirror of the UCI dataset (`jbrownlee/Datasets/adult-all.csv`), which hosts the
  dataset's canonical CSV form unmodified from the original UCI release.
- **Original size:** 48,842 instances, 14 raw attributes + 1 target
- **Used in this project:** a stratified subsample of 6,000 instances (to keep
  SVM training tractable and the repo lightweight for Streamlit Community
  Cloud's free tier), split 80/20 into train (4,800) and test (1,200), both
  stratified on the target to preserve class balance.
- **Target:** `income` — binarized to 1 if `>50K`, 0 if `<=50K`
- **Features used (12, after cleaning):** `age`, `education_num`,
  `capital_gain`, `capital_loss`, `hours_per_week` (numeric), and `workclass`,
  `marital_status`, `occupation`, `relationship`, `race`, `sex`,
  `native_country` (categorical).
- **Dropped from the original 14 raw columns:** `fnlwgt` (a census sampling
  weight, not a predictive feature) and `education` (a redundant string
  duplicate of the already-retained ordinal `education_num`).
- **Preprocessing:** rows with missing `workclass`/`occupation` dropped;
  `native_country` collapsed to `United-States` vs `Other` (90%+ of rows are
  `United-States`, and the long tail of ~40 other countries has too few rows
  each to be informative); numeric features standardized; categorical features
  one-hot encoded (`drop='first'`, unknown categories handled gracefully).

This satisfies the assignment's requirements of ≥500 instances (6,000 used),
≥12 features (exactly 12 after deliberate, justified feature engineering), and
a binary target sourced from UCI.

## 3. GitHub Repository

> **GitHub repo link:** _add your repo URL here after you push these files,
> e.g. `https://github.com/<your-username>/ml-assignment-2`_

## 4. Model Comparison

All 6 metrics computed on the same 1,200-row held-out test split, using
`sklearn.metrics` (no retraining — each model is a fitted
`Pipeline(preprocessing + classifier)` saved as a single `.joblib` file).

| Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8492 | **0.8995** | 0.7468 | 0.5940 | **0.6617** | **0.5724** |
| Decision Tree | 0.8442 | 0.8697 | 0.7362 | 0.5805 | 0.6492 | 0.5572 |
| KNN | 0.8258 | 0.8819 | 0.6802 | 0.5638 | 0.6165 | 0.5088 |
| Naive Bayes | 0.6775 | 0.8676 | 0.4297 | **0.9128** | 0.5843 | 0.4435 |
| Support Vector Machine | **0.8475** | 0.8948 | **0.7674** | 0.5537 | 0.6433 | 0.5613 |

*(Bold = best value per metric.)*

### Per-Model Observations

| Model | Observation |
|---|---|
| **Logistic Regression** | Best AUC (0.900) and F1 (0.662), second-best accuracy, trains almost instantly. A linear decision boundary is evidently sufficient to separate income classes on these features — the dataset doesn't reward extra model complexity. |
| **Decision Tree** | (depth-capped at 8 to control overfitting) lands just behind Logistic Regression across the board — a reasonable, interpretable second choice, and the fastest to train. |
| **KNN** | (k=15) is the weakest on accuracy and F1. Distance-based methods are sensitive to the mixed numeric/one-hot feature space — the one-hot dimensions dilute the numeric features' contribution to distance. |
| **Naive Bayes** | The standout outlier: by far the lowest accuracy (0.68) and precision (0.43), but the highest recall (0.91). Its independence assumption is clearly violated here (age, education, hours-worked are correlated), hurting calibration — but it compensates by flagging almost every true high-earner, at the cost of many false positives. Best choice if the goal is "never miss a high earner." |
| **Support Vector Machine** | (RBF kernel) achieves the best precision (0.77) and a strong AUC (0.895), but trades off recall — the most conservative model about predicting the positive class. Also by far the slowest to train (~3.4s vs <0.1s for the others). |

### Overall Winner: **Logistic Regression**

It leads on AUC, F1, and MCC — the three metrics that matter most on an
imbalanced binary target — while also being the cheapest model to train and the
easiest to explain to a non-technical stakeholder.

## 5. Repository Structure

```
.
├── app.py                      # Streamlit app (loads saved models, scores uploaded CSV)
├── requirements.txt
├── README.md
├── test_data.csv                # Held-out test split (raw features + true labels)
├── ml2_assignment.ipynb         # Source notebook: EDA, training, evaluation, saving
├── model/
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── knn.joblib
│   ├── naive_bayes.joblib
│   ├── svm.joblib
│   └── comparison_table.csv
└── data/
    └── adult_raw.csv             # Original UCI Adult dataset
```

## 6. Running the App Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then upload `test_data.csv` (or any CSV with the same 12 feature columns plus
an `income` column) and select a model from the sidebar dropdown.

## 7. Deployment

Deployed on Streamlit Community Cloud at: `https://<your-app-name>.streamlit.app`
_(fill in after deploying — see deployment instructions provided separately)._

## 8. BITS Virtual Lab Execution

_Screenshot proving notebook execution on the BITS Virtual Lab to be added here
per submission requirements._
