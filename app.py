import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    confusion_matrix, ConfusionMatrixDisplay,
    accuracy_score, precision_score, recall_score,
    f1_score, classification_report,
    roc_curve, roc_auc_score,
    precision_recall_curve, auc
)

st.set_page_config(page_title="Employee Attrition Prediction", layout="wide")
st.title("Employee Attrition Prediction using Decision Tree")
st.markdown("Dataset: **IBM HR Analytics Employee Attrition**")

# Upload
uploaded_file = st.file_uploader("Upload your CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    # --- Data Understanding ---
    st.header("1. Data Understanding")
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Duplicates", df.duplicated().sum())

    st.subheader("First 5 Rows")
    st.dataframe(df.head())

    st.subheader("Info")
    st.text(df.info().__repr__())

    st.subheader("Describe")
    st.dataframe(df.describe(include="all"))

    st.subheader("Missing Values")
    st.dataframe(df.isnull().sum().to_frame("Missing"))

    # --- EDA ---
    st.header("2. Exploratory Data Analysis")
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.countplot(data=df, x="Attrition", ax=ax)
    ax.set_title("Attrition Distribution")
    st.pyplot(fig)

    # --- Preprocessing ---
    st.header("3. Preprocessing")
    drop_cols = ["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"]
    drop_cols = [c for c in drop_cols if c in df.columns]
    df = df.drop(columns=drop_cols)
    st.info(f"Dropped columns: {drop_cols}")

    encoder = LabelEncoder()
    for col in df.select_dtypes(include="object").columns:
        df[col] = encoder.fit_transform(df[col])

    st.subheader("Encoded Data (First 5 Rows)")
    st.dataframe(df.head())

    # --- Split ---
    X = df.drop("Attrition", axis=1)
    y = df["Attrition"]

    test_size = st.slider("Test Size", 0.1, 0.5, 0.2, 0.05)
    max_depth = st.slider("Max Depth", 1, 20, 5)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    # --- Model ---
    st.header("4. Decision Tree Classifier")
    model = DecisionTreeClassifier(random_state=42, max_depth=max_depth)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # --- Metrics ---
    st.header("5. Model Evaluation")

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Accuracy", f"{acc:.4f}")
    m2.metric("Precision", f"{prec:.4f}")
    m3.metric("Recall", f"{rec:.4f}")
    m4.metric("F1 Score", f"{f1:.4f}")
    m5.metric("ROC-AUC", f"{roc_auc:.4f}")

    # Confusion Matrix
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax_cm = plt.subplots()
    ConfusionMatrixDisplay(cm, display_labels=["No", "Yes"]).plot(ax=ax_cm)
    st.pyplot(fig_cm)

    # Classification Report
    st.subheader("Classification Report")
    st.code(classification_report(y_test, y_pred))

    # ROC Curve
    st.subheader("ROC Curve")
    fpr, tpr, thr = roc_curve(y_test, y_prob)
    fig_roc, ax_roc = plt.subplots(figsize=(6, 5))
    ax_roc.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    ax_roc.plot([0, 1], [0, 1], "--")
    ax_roc.set_xlabel("False Positive Rate")
    ax_roc.set_ylabel("True Positive Rate")
    ax_roc.set_title("ROC Curve")
    ax_roc.legend()
    st.pyplot(fig_roc)

    # Precision-Recall Curve
    st.subheader("Precision-Recall Curve")
    precision_arr, recall_arr, thr_pr = precision_recall_curve(y_test, y_prob)
    pr_auc_val = auc(recall_arr, precision_arr)
    fig_pr, ax_pr = plt.subplots(figsize=(6, 5))
    ax_pr.plot(recall_arr, precision_arr, label=f"PR AUC = {pr_auc_val:.3f}")
    ax_pr.set_xlabel("Recall")
    ax_pr.set_ylabel("Precision")
    ax_pr.set_title("Precision-Recall Curve")
    ax_pr.legend()
    st.pyplot(fig_pr)

    # Feature Importance
    st.header("6. Feature Importance")
    fi = pd.Series(model.feature_importances_, index=X.columns)
    fi = fi.sort_values(ascending=False)

    fig_fi, ax_fi = plt.subplots(figsize=(8, 6))
    fi.head(15).plot(kind="barh", ax=ax_fi)
    ax_fi.set_title("Top 15 Feature Importance")
    st.pyplot(fig_fi)

    st.dataframe(fi.head(15).to_frame("Importance"))

    # Decision Tree
    st.header("7. Decision Tree Visualization")
    fig_dt, ax_dt = plt.subplots(figsize=(18, 10))
    plot_tree(
        model,
        feature_names=X.columns,
        class_names=["No", "Yes"],
        filled=True,
        fontsize=5,
        ax=ax_dt,
    )
    st.pyplot(fig_dt)

    # Interview Questions
    st.header("8. Interview Questions")
    st.markdown("""
    1. Difference between Accuracy and F1 Score?
    2. When should Recall be preferred over Precision?
    3. What does ROC-AUC represent?
    4. Why use a Precision-Recall Curve?
    5. How does a Decision Tree decide its splits?
    6. How can overfitting in Decision Trees be reduced?
    """)

else:
    st.info("Upload the IBM HR Analytics CSV file to begin.")
