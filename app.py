"""
Streamlit app — Assignment 2 (ML Model Comparison)
Loads pre-trained model pipelines and scores an uploaded test CSV.
No retraining happens here.
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

st.set_page_config(page_title="Income Classification - Model Comparison", layout="wide")

MODEL_FILES = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "KNN": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "Support Vector Machine": "model/svm.joblib",
}

FEATURE_COLS = ['age', 'education_num', 'capital_gain', 'capital_loss', 'hours_per_week',
                'workclass', 'marital_status', 'occupation', 'relationship', 'race', 'sex',
                'native_country']
TARGET_COL = "income"


@st.cache_resource
def load_model(path):
    return joblib.load(path)


def compute_metrics(y_true, y_pred, y_proba):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_proba),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


st.title("Income Classification — Model Comparison")
st.caption(
    "UCI Adult (Census Income) dataset · 5 classifiers trained offline · "
    "upload a test CSV below to see how each model performs."
)

# ---------------------------------------------------------------------------
# Sidebar: upload + model selection
# ---------------------------------------------------------------------------
st.sidebar.header("Controls")

uploaded_file = st.sidebar.file_uploader(
    "Upload test CSV", type="csv",
    help="Must contain the 12 feature columns plus the 'income' target column (0/1)."
)

selected_model_name = st.sidebar.selectbox("Select model", list(MODEL_FILES.keys()))

st.sidebar.markdown("---")
st.sidebar.markdown(
    "**Expected columns:**\n\n" + ", ".join(FEATURE_COLS) + f", **{TARGET_COL}**"
)

if uploaded_file is None:
    st.info("Upload a test CSV from the sidebar to get started. "
            "(Use the `test_data.csv` included in this repo to try it out.)")
    st.stop()

# ---------------------------------------------------------------------------
# Load + validate uploaded data
# ---------------------------------------------------------------------------
try:
    data = pd.read_csv(uploaded_file)
except Exception as e:
    st.error(f"Could not read the uploaded file: {e}")
    st.stop()

missing_cols = [c for c in FEATURE_COLS + [TARGET_COL] if c not in data.columns]
if missing_cols:
    st.error(f"Uploaded CSV is missing required column(s): {', '.join(missing_cols)}")
    st.stop()

X = data[FEATURE_COLS]
y_true = data[TARGET_COL]

st.subheader("Uploaded Data Preview")
st.dataframe(data.head(10), use_container_width=True)
st.caption(f"{data.shape[0]} rows loaded.")

# ---------------------------------------------------------------------------
# Selected-model evaluation
# ---------------------------------------------------------------------------
st.header(f"Results — {selected_model_name}")

model = load_model(MODEL_FILES[selected_model_name])
y_pred = model.predict(X)
y_proba = model.predict_proba(X)[:, 1]
metrics = compute_metrics(y_true, y_pred, y_proba)

metric_cols = st.columns(6)
for col, (name, value) in zip(metric_cols, metrics.items()):
    col.metric(name, f"{value:.4f}")

col_cm, col_report = st.columns(2)

with col_cm:
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="viridis", cbar=False,
                xticklabels=["≤50K", ">50K"], yticklabels=["≤50K", ">50K"], ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)

with col_report:
    st.subheader("Classification Report")
    report = classification_report(y_true, y_pred, target_names=["≤50K", ">50K"], output_dict=True)
    report_df = pd.DataFrame(report).T.round(3)
    st.dataframe(report_df, use_container_width=True)

# ---------------------------------------------------------------------------
# All-models comparison on the same uploaded data
# ---------------------------------------------------------------------------
st.header("All Models on This Test Data")

all_results = {}
for name, path in MODEL_FILES.items():
    m = load_model(path)
    pred = m.predict(X)
    proba = m.predict_proba(X)[:, 1]
    all_results[name] = compute_metrics(y_true, pred, proba)

comparison_df = pd.DataFrame(all_results).T.round(4)
st.dataframe(
    comparison_df.style.highlight_max(axis=0, color="#2a9d8f44"),
    use_container_width=True,
)

fig2, ax2 = plt.subplots(figsize=(11, 4.5))
x = np.arange(len(comparison_df.columns))
width = 0.15
palette = sns.color_palette("viridis", len(comparison_df))
for i, (name, row) in enumerate(comparison_df.iterrows()):
    ax2.bar(x + i * width, row.values, width, label=name, color=palette[i])
ax2.set_xticks(x + width * (len(comparison_df) - 1) / 2)
ax2.set_xticklabels(comparison_df.columns)
ax2.set_ylabel("Score")
ax2.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
ax2.grid(axis="y", alpha=0.3)
st.pyplot(fig2)

st.caption("All models are pre-trained and loaded from disk — nothing is retrained in this app.")
