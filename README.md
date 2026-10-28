# Waveform Type Classification: Multi-Model Comparison

ML Assignment 2 · M.Tech (AIML) · BITS ID 2026ak05368 · Rakshantha M

- **Live Streamlit app:** `https://2026AK05368_mlassignment-2.streamlit.app/`
- **GitHub repository:** `https://github.com/2026ak05368/ML_assignment_2`

## a. Problem statement

Given a 21-point noisy signal, decide which of three wave classes it came from. Each sample is a random mix of two of three base triangular waveforms plus Gaussian noise, so the classes overlap heavily and the task is a genuine test of how well different model families cope with noisy, correlated numeric inputs.

Five classifiers (Logistic Regression, Decision Tree, KNN, Gaussian Naive Bayes, SVM) are trained on the same data and compared on six metrics. The models are served in a Streamlit app that scores an uploaded labelled test file **without retraining**.

## b. Dataset description

| Item | Value |
|---|---|
| Name | Waveform Database Generator (Version 1) |
| Source | UCI Machine Learning Repository, dataset id 107 (https://archive.ics.uci.edu/dataset/107). Raw file obtained through the PMLB mirror (`waveform_21`) |
| Type | Tabular, numeric (not an image dataset) |
| Instances | 5,000 (requirement: ≥ 500) |
| Features | 21 continuous attributes, `attr_01` … `attr_21` (requirement: ≥ 12) |
| Target | `wave_class` with 3 classes: 0 = 1,657 (33.1 %), 1 = 1,647 (32.9 %), 2 = 1,696 (33.9 %). **Balanced.** |
| Missing values | None |

Preprocessing is inside each saved scikit-learn `Pipeline`, so the app takes the raw columns: `StandardScaler` for Logistic Regression, KNN and SVM; none for Decision Tree and Naive Bayes. Split: stratified 80/20 (`random_state=5368`): 4,000 training rows and 1,000 test rows. The 1,000 test rows are saved as `test_data.csv` and are never used for training or tuning. Hyper-parameters were chosen with `GridSearchCV` (5-fold stratified CV on the training split, macro-F1).

## c. GitHub repository contents

```
2026ak05368_ML_Assignment-2/
├── app.py                  # Streamlit app (loads saved models, never retrains)
├── train_models.py         # split, tuning, training, evaluation, model export
├── requirements.txt
├── README.md
├── test_data.csv           # 1,000-row hold-out set (21 features + wave_class)
├── .streamlit/config.toml  # app theme
├── data/waveform_raw.csv   # full dataset (5,000 rows)
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    ├── svm.joblib
    ├── test_metrics.csv    # comparison table below
    └── metadata.json       # feature list, classes, best hyper-parameters
```

## d. Models used

| Model | Best hyper-parameters (5-fold CV) |
|---|---|
| Logistic Regression | C = 0.001 |
| Decision Tree | criterion = entropy, max_depth = 8, min_samples_leaf = 3 |
| KNN | k = 9, Euclidean distance, distance-weighted |
| Naive Bayes (Gaussian) | var_smoothing = 1e-9 |
| Support Vector Machine | RBF kernel, C = 100, gamma = 0.0003; probabilities via sigmoid calibration |

### Comparison table (1,000-row hold-out test set)

Precision, Recall and F1 are macro-averaged over the 3 classes. AUC is the one-vs-rest macro ROC-AUC.

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8590 | 0.9704 | 0.8602 | 0.8590 | 0.8580 | 0.7898 |
| Decision Tree | 0.7510 | 0.8980 | 0.7529 | 0.7513 | 0.7500 | 0.6284 |
| KNN | 0.8390 | 0.9601 | 0.8393 | 0.8390 | 0.8384 | 0.7591 |
| Naive Bayes | 0.8160 | 0.9601 | 0.8435 | 0.8156 | 0.8064 | 0.7402 |
| Support Vector Machine | 0.8660 | 0.9731 | 0.8660 | 0.8661 | 0.8658 | 0.7992 |

Per-class recall on the test set (class 0 / 1 / 2): LR 0.78 / 0.92 / 0.89, DT 0.68 / 0.83 / 0.74, KNN 0.78 / 0.88 / 0.86, NB 0.57 / 0.95 / 0.93, SVM 0.83 / 0.90 / 0.86.

### Observations on model performance

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Within about 1 point of the best model on accuracy and F1, and a very close second on AUC (0.970). CV picked a very small C (0.001), i.e. heavy shrinkage: the 21 attributes are noisy and correlated (mean absolute pairwise correlation about 0.30), so a strongly regularised linear boundary generalises well. The largest coefficients sit on attributes 5, 6 and 10 to 13, where the class means differ most. It makes 141 errors out of 1,000, trains in seconds, and is the easiest model to interpret. |
| Decision Tree | Clearly the weakest (accuracy 0.751, MCC 0.628). Even after tuning (depth 8, 150 leaves) it overfits the noise: every split looks at one attribute, but the class information is spread across many noisy attributes at once, so single-attribute thresholds are a poor fit. Recall is below 0.75 on classes 0 and 2, and its piece-wise constant probabilities give the lowest AUC (0.898). |
| KNN | Mid-table (accuracy 0.839, F1 0.838). Distance-weighted voting with k = 9 helps average out noise, and it ranks examples well (AUC 0.960), but with 21 noisy dimensions neighbourhoods are less pure than a global linear boundary, so it trails LR and SVM by about 2 to 3 points of accuracy. |
| Naive Bayes | Accuracy 0.816, but its recall is very uneven: 0.57 on class 0 against 0.95 and 0.93 on classes 1 and 2, so the macro recall and MCC (0.740) are lower than the AUC (0.960) suggests. The independence assumption does not hold (neighbouring attributes are correlated because they sample the same smooth wave), so class 0 is often absorbed into the other two. Ranking quality is still good, which means a threshold or prior adjustment could recover much of the gap. Very fast to train. |
| Support Vector Machine | Best on Accuracy (0.866), AUC (0.973), Precision, Recall, F1 (0.866) and MCC (0.799). The RBF kernel with a very small gamma gives a smooth, low-complexity boundary that can still bend where the classes overlap, which suits this noisy data. It has the best recall on the hardest class 0 (0.83), and the fewest errors (134 of 1,000). It is the slowest model to tune because of the C/gamma grid and probability calibration. |
| **Overall, winner for this dataset?** | **Support Vector Machine (RBF)** has the top value on every metric, but its lead over Logistic Regression is small (0.7 points of accuracy, about 0.01 of MCC), which is within noise for a 1,000-row test set. If speed and interpretability matter, Logistic Regression is the practical choice. The Decision Tree is the weakest model by a wide margin, and class 0 is the hardest class for every model. |

## Streamlit app features

- **CSV upload** of the labelled test data (sample `test_data.csv` downloadable from the sidebar).
- **Model dropdown** with a one-line description and the tuned hyper-parameters of each model.
- **Evaluation metrics:** Accuracy, AUC, Precision, Recall, F1 and MCC, computed live on the uploaded file.
- **Confusion matrix and classification report**, plus the most frequent mistake.
- **Compare all models** on the uploaded file, with the best value per metric highlighted and a bar chart ranked by any metric.
- **Predictions** table (predicted class and confidence per row, wrong-only filter, CSV download) and a plot of the first misclassified waveforms.
- **Input validation:** missing columns, unreadable files, blank values and unknown class labels are reported instead of crashing.

The app only calls `predict` / `predict_proba` on the saved pipelines. Nothing is retrained on the uploaded data.

## How to run locally

```bash
pip install -r requirements.txt
python train_models.py        # optional: re-creates model/*.joblib, test_data.csv, test_metrics.csv
streamlit run app.py
```

## Deployment (Streamlit Community Cloud)

New app → select this repository → branch `main` → main file `app.py` → Deploy.
