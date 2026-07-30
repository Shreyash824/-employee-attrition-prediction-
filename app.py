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

st.set_page_config(
    page_title="Employee Attrition Prediction",
    page_icon="🏢",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .stApp { background: #f8f9fa; }
    .main-header {
        text-align: center; padding: 1.5rem 0; margin-bottom: 1rem;
        background: linear-gradient(135deg, #1e3a5f 0%, #2d6a9f 100%);
        border-radius: 16px; color: white !important;
    }
    .main-header h1 { color: white !important; font-size: 2.2rem; margin: 0; }
    .main-header p { color: #d4e4f7 !important; margin: 0.3rem 0 0 0; font-size: 1rem; }
    .section-card {
        background: white; padding: 1.5rem; border-radius: 14px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.06); margin-bottom: 1.5rem;
        border-left: 4px solid #2d6a9f;
        transition: box-shadow 0.2s;
    }
    .section-card:hover { box-shadow: 0 4px 20px rgba(0,0,0,0.1); }
    .section-card h2 { color: #1e3a5f; font-size: 1.4rem; margin-top: 0; margin-bottom: 0.8rem; }
    .section-card h3 { color: #2d6a9f; font-size: 1.1rem; }
    .metric-card {
        background: linear-gradient(135deg, #e8f0fe, #f0f4ff); padding: 1rem 0.5rem;
        border-radius: 12px; text-align: center; border: 1px solid #d0dff5;
    }
    .metric-card .label { font-size: 0.8rem; color: #555; font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }
    .metric-card .value { font-size: 1.6rem; font-weight: 700; color: #1e3a5f; margin: 4px 0; }
    .metric-card .caption { font-size: 0.7rem; color: #888; }
    .interview-card {
        background: #fffbe6; padding: 1.2rem; border-radius: 12px;
        border: 1px solid #ffe58f; margin-bottom: 0.6rem;
    }
    .interview-card p { margin: 0; font-weight: 600; color: #8d6e00; }
    .stButton button {
        background: #2d6a9f; color: white; border: none;
        border-radius: 8px; font-weight: 600; padding: 0.4rem 1.5rem;
    }
    .stButton button:hover { background: #1e3a5f; }
    div[data-testid="stStatusWidget"] { display: none; }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="main-header">
    <h1>🏢 Employee Attrition Prediction</h1>
    <p>Decision Tree Classifier &bull; IBM HR Analytics Dataset</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("## ⚙️ Controls")
st.sidebar.markdown("---")

uploaded_file = st.sidebar.file_uploader("📂 Upload CSV", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("## 📊 Data Overview")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Duplicates", df.duplicated().sum())
    col4.metric("Missing", df.isnull().sum().sum())
    st.markdown('</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["👀 Preview", "📋 Summary", "📉 Statistics"])
    with tab1:
        st.dataframe(df.head(), use_container_width=True)
    with tab2:
        buf = pd.DataFrame({
            "Dtype": df.dtypes,
            "Non-Null": df.count(),
            "Null": df.isnull().sum(),
            "Unique": df.nunique()
        })
        st.dataframe(buf, use_container_width=True)
    with tab3:
        st.dataframe(df.describe(include="all"), use_container_width=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("## 🔍 Exploratory Data Analysis")
    col_e1, col_e2 = st.columns(2)
    with col_e1:
        fig, ax = plt.subplots(figsize=(5, 3.5))
        sns.set_style("whitegrid")
        colors = ["#2d6a9f", "#e74c3c"]
        sns.countplot(data=df, x="Attrition", ax=ax, palette=colors)
        ax.set_title("Attrition Distribution", fontsize=13, fontweight="bold")
        ax.set_xlabel("")
        for p in ax.patches:
            ax.annotate(f"{p.get_height()}", (p.get_x() + p.get_width()/2, p.get_height()),
                        ha="center", va="bottom", fontweight="bold")
        st.pyplot(fig)
    with col_e2:
        num_cols = df.select_dtypes(include=np.number).columns[:4]
        if len(num_cols) >= 2:
            fig2, axes2 = plt.subplots(1, min(2, len(num_cols)), figsize=(7, 3.5))
            if len(num_cols) == 1:
                axes2 = [axes2]
            for i, c in enumerate(num_cols[:2]):
                sns.histplot(df[c], bins=30, ax=axes2[i], color=colors[i])
                axes2[i].set_title(c, fontsize=11)
            plt.tight_layout()
            st.pyplot(fig2)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("## 🧹 Preprocessing")
    drop_cols = ["EmployeeCount", "EmployeeNumber", "Over18", "StandardHours"]
    drop_cols = [c for c in drop_cols if c in df.columns]
    cols_before = df.shape[1]
    df = df.drop(columns=drop_cols)
    st.success(f"✅ Dropped {len(drop_cols)} columns: `{', '.join(drop_cols)}` — shape: `{df.shape}`")

    encoder = LabelEncoder()
    obj_cols = df.select_dtypes(include="object").columns.tolist()
    if obj_cols:
        st.info(f"🔤 Label encoding {len(obj_cols)} categorical columns: `{', '.join(obj_cols)}`")
        for col in obj_cols:
            df[col] = encoder.fit_transform(df[col])

    with st.expander("👁️ Preview Encoded Data"):
        st.dataframe(df.head(), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("## 🤖 Model Configuration")

    X = df.drop("Attrition", axis=1)
    y = df["Attrition"]

    col_c1, col_c2, col_c3 = st.columns(3)
    with col_c1:
        test_size = st.slider("Test Size", 0.1, 0.5, 0.2, 0.05, help="Proportion of data used for testing")
    with col_c2:
        max_depth = st.slider("Max Depth", 1, 20, 5, help="Maximum depth of the decision tree")
    with col_c3:
        min_samples_split = st.slider("Min Samples Split", 2, 20, 2, help="Minimum samples required to split a node")

    st.markdown("### 📈 Performance Metrics", unsafe_allow_html=True)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42, stratify=y
    )

    model = DecisionTreeClassifier(
        random_state=42,
        max_depth=max_depth,
        min_samples_split=min_samples_split
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_prob)

    metrics_html = "<div style='display:flex;gap:1rem;flex-wrap:wrap;margin:1rem 0;'>"
    metric_data = [
        ("🎯", "Accuracy", f"{acc:.4f}"),
        ("📌", "Precision", f"{prec:.4f}"),
        ("🔎", "Recall", f"{rec:.4f}"),
        ("⚖️", "F1 Score", f"{f1:.4f}"),
        ("📈", "ROC-AUC", f"{roc_auc:.4f}"),
    ]
    for icon, label, value in metric_data:
        delta = "✅" if float(value) > 0.7 else "⚠️"
        metrics_html += f"""
        <div class="metric-card" style="flex:1;min-width:120px;">
            <div class="label">{icon} {label}</div>
            <div class="value">{value}</div>
            <div class="caption">{delta}</div>
        </div>"""
    metrics_html += "</div>"
    st.markdown(metrics_html, unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    viz_tab1, viz_tab2, viz_tab3, viz_tab4 = st.tabs([
        "🟦 Confusion Matrix", "📈 ROC Curve", "📉 PR Curve", "📋 Report"
    ])

    with viz_tab1:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax_cm,
                    xticklabels=["No Attrition", "Attrition"],
                    yticklabels=["No Attrition", "Attrition"],
                    cbar=False, linewidths=1, linecolor="white")
        ax_cm.set_ylabel("Actual", fontweight="bold")
        ax_cm.set_xlabel("Predicted", fontweight="bold")
        ax_cm.set_title("Confusion Matrix", fontsize=13, fontweight="bold")
        st.pyplot(fig_cm)
        st.markdown('</div>', unsafe_allow_html=True)

    with viz_tab2:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        fpr, tpr, thr = roc_curve(y_test, y_prob)
        fig_roc, ax_roc = plt.subplots(figsize=(6, 4))
        ax_roc.plot(fpr, tpr, color="#2d6a9f", lw=2.5, label=f"AUC = {roc_auc:.3f}")
        ax_roc.fill_between(fpr, tpr, alpha=0.15, color="#2d6a9f")
        ax_roc.plot([0, 1], [0, 1], "--", color="gray", lw=1.5)
        ax_roc.set_xlabel("False Positive Rate", fontweight="bold")
        ax_roc.set_ylabel("True Positive Rate", fontweight="bold")
        ax_roc.set_title("ROC Curve", fontsize=13, fontweight="bold")
        ax_roc.legend(loc="lower right", fontsize=11)
        ax_roc.set_xlim(0, 1)
        ax_roc.set_ylim(0, 1)
        st.pyplot(fig_roc)
        st.markdown('</div>', unsafe_allow_html=True)

    with viz_tab3:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        precision_arr, recall_arr, thr_pr = precision_recall_curve(y_test, y_prob)
        pr_auc_val = auc(recall_arr, precision_arr)
        fig_pr, ax_pr = plt.subplots(figsize=(6, 4))
        ax_pr.plot(recall_arr, precision_arr, color="#e74c3c", lw=2.5, label=f"PR AUC = {pr_auc_val:.3f}")
        ax_pr.fill_between(recall_arr, precision_arr, alpha=0.15, color="#e74c3c")
        ax_pr.set_xlabel("Recall", fontweight="bold")
        ax_pr.set_ylabel("Precision", fontweight="bold")
        ax_pr.set_title("Precision-Recall Curve", fontsize=13, fontweight="bold")
        ax_pr.legend(loc="lower left", fontsize=11)
        st.pyplot(fig_pr)
        st.markdown('</div>', unsafe_allow_html=True)

    with viz_tab4:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        report = classification_report(y_test, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose().round(4)
        st.dataframe(report_df.style.highlight_max(axis=0, color="#d4e4f7"), use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("## 🌟 Feature Importance")
    fi = pd.Series(model.feature_importances_, index=X.columns)
    fi_sorted = fi.sort_values(ascending=True).tail(15)

    col_f1, col_f2 = st.columns([3, 2])
    with col_f1:
        fig_fi, ax_fi = plt.subplots(figsize=(8, 5))
        colors_fi = plt.cm.Blues(np.linspace(0.4, 0.9, len(fi_sorted)))
        fi_sorted.plot(kind="barh", ax=ax_fi, color=colors_fi[::-1])
        ax_fi.set_title("Top 15 Feature Importance", fontsize=13, fontweight="bold")
        ax_fi.set_xlabel("Importance", fontweight="bold")
        ax_fi.spines["top"].set_visible(False)
        ax_fi.spines["right"].set_visible(False)
        st.pyplot(fig_fi)
    with col_f2:
        imp_df = fi_sorted.reset_index()
        imp_df.columns = ["Feature", "Importance"]
        imp_df["Importance"] = imp_df["Importance"].round(4)
        imp_df.index = range(1, len(imp_df) + 1)
        st.dataframe(imp_df.style.background_gradient(cmap="Blues"), use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("## 🌲 Decision Tree Visualization")
    fig_dt, ax_dt = plt.subplots(figsize=(20, 12))
    plot_tree(
        model,
        feature_names=X.columns,
        class_names=["No", "Yes"],
        filled=True,
        fontsize=6,
        ax=ax_dt,
        rounded=True,
        proportion=True,
    )
    st.pyplot(fig_dt)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown("## 💡 Interview Questions")
    questions = [
        "⚡ Difference between Accuracy and F1 Score?",
        "🎯 When should Recall be preferred over Precision?",
        "📊 What does ROC-AUC represent?",
        "📉 Why use a Precision-Recall Curve?",
        "🌲 How does a Decision Tree decide its splits?",
        "🛡️ How can overfitting in Decision Trees be reduced?"
    ]
    for q in questions:
        st.markdown(f'<div class="interview-card"><p>{q}</p></div>', unsafe_allow_html=True)
    st.markdown('</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown(
        "<div style='text-align:center;color:#888;padding:1rem;'>"
        "Built with ❤️ using Streamlit &bull; scikit-learn &bull; pandas"
        "</div>",
        unsafe_allow_html=True
    )

else:
    col_w1, col_w2, col_w3 = st.columns([1, 2, 1])
    with col_w2:
        st.markdown("""
        <div style='text-align:center;padding:4rem 1rem;'>
            <div style='font-size:4rem;'>📂</div>
            <h3 style='color:#1e3a5f;'>Upload your dataset</h3>
            <p style='color:#666;'>Upload the <strong>IBM HR Analytics Employee Attrition</strong> CSV file<br>
            from the sidebar to get started.</p>
            <p style='color:#999;font-size:0.85rem;'>Required column: <code>Attrition</code></p>
        </div>
        """, unsafe_allow_html=True)
