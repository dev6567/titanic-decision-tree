import io
import os

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.compose import ColumnTransformer
from sklearn.datasets import load_breast_cancer, load_iris
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.tree import DecisionTreeClassifier, export_text, plot_tree

st.set_page_config(page_title="Decision Tree Classifier", page_icon="🌳", layout="wide")

TITANIC_PATH = "data/train.csv"
TARGET_HINTS = ["survived", "target", "label", "class", "species", "diagnosis", "y", "outcome"]


# ---------------------------------------------------------------- helpers
@st.cache_data
def load_sample(name):
    if name == "Iris flowers":
        d = load_iris(as_frame=True)
        df = d.frame.copy()
        df["species"] = df["target"].map(dict(enumerate(d.target_names)))
        return df.drop(columns="target")
    if name == "Breast cancer":
        d = load_breast_cancer(as_frame=True)
        df = d.frame.copy()
        df["diagnosis"] = df["target"].map(dict(enumerate(d.target_names)))
        return df.drop(columns="target")
    return pd.read_csv(TITANIC_PATH)


def prepare_X(df, features):
    X = df[list(features)].copy()
    for c in X.select_dtypes(include="bool").columns:
        X[c] = X[c].astype(str)
    return X


def suggest_features(df, target):
    """Pick sensible default features: skip IDs, free-text and mostly-empty columns."""
    keep = []
    for c in df.columns:
        if c == target:
            continue
        s = df[c].dropna()
        if len(s) == 0 or df[c].isna().mean() > 0.6:
            continue
        ratio = s.nunique() / len(s)
        if pd.api.types.is_integer_dtype(s) and s.nunique() == len(df):
            continue  # looks like an ID
        if not pd.api.types.is_numeric_dtype(s) and ratio > 0.5:
            continue  # looks like free text (names, tickets...)
        keep.append(c)
    return keep


@st.cache_resource(show_spinner=False)
def train_model(df, target, features, criterion, max_depth, min_leaf, test_size):
    data = df.dropna(subset=[target])
    X = prepare_X(data, features)
    y = data[target]

    num_cols = list(X.select_dtypes(include="number").columns)
    cat_cols = [c for c in X.columns if c not in num_cols]

    pre = ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median"), num_cols),
            (
                "cat",
                Pipeline(
                    [
                        ("imp", SimpleImputer(strategy="most_frequent")),
                        ("enc", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
                    ]
                ),
                cat_cols,
            ),
        ]
    )
    model = Pipeline(
        [
            ("pre", pre),
            (
                "tree",
                DecisionTreeClassifier(
                    criterion=criterion,
                    max_depth=max_depth,
                    min_samples_leaf=min_leaf,
                    random_state=42,
                ),
            ),
        ]
    )

    try:
        X_tr, X_te, y_tr, y_te = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )
    except ValueError:  # a class has too few rows to stratify
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=42)

    model.fit(X_tr, y_tr)
    return {
        "model": model,
        "X_tr": X_tr, "X_te": X_te, "y_tr": y_tr, "y_te": y_te,
        "feature_names": num_cols + cat_cols,
        "num_cols": num_cols, "cat_cols": cat_cols,
    }


# ---------------------------------------------------------------- sidebar
st.sidebar.title("🌳 Decision Tree Lab")
st.sidebar.caption("Upload any CSV, pick what to predict, and explore the model.")

st.sidebar.header("1 · Data")
sources = ["Upload CSV", "Iris flowers", "Breast cancer"]
if os.path.exists(TITANIC_PATH):
    sources.append("Titanic")
source = st.sidebar.radio("Data source", sources)

if source == "Upload CSV":
    up = st.sidebar.file_uploader("Choose a CSV file", type="csv")
    df = pd.read_csv(up) if up else None
else:
    df = load_sample(source)

st.title("🌳 Decision Tree Classifier")
st.caption("A beginner-friendly, general-purpose machine learning app built with scikit-learn and Streamlit.")

if df is None:
    st.info("👈 Upload a CSV file in the sidebar, or choose a sample dataset to get started.")
    st.stop()

st.sidebar.header("2 · What to predict")
default_t = next((i for i, c in enumerate(df.columns) if c.lower() in TARGET_HINTS), len(df.columns) - 1)
target = st.sidebar.selectbox("Target column", df.columns, index=default_t)

features = st.sidebar.multiselect(
    "Feature columns", [c for c in df.columns if c != target],
    default=suggest_features(df, target),
    help="IDs, names and mostly-empty columns are left out by default.",
)

st.sidebar.header("3 · Model settings")
criterion = st.sidebar.selectbox("Split criterion", ["gini", "entropy"])
max_depth = st.sidebar.slider("Max depth", 1, 15, 4, help="Deeper trees fit the training data better but can overfit.")
min_leaf = st.sidebar.slider("Min samples per leaf", 1, 30, 1)
test_size = st.sidebar.slider("Test set size (%)", 10, 40, 20) / 100

# ---------------------------------------------------------------- validation
clean = df.dropna(subset=[target])
n_classes = clean[target].nunique()
if not features:
    st.warning("Select at least one feature column in the sidebar.")
    st.stop()
if n_classes < 2:
    st.error(f"'{target}' has only one class, so there is nothing to classify.")
    st.stop()
if pd.api.types.is_float_dtype(clean[target]) or n_classes > 30:
    st.error(
        f"'{target}' has {n_classes} different values, which looks like a number to estimate "
        "(regression), not a category. Pick a column with a small set of classes."
    )
    st.stop()

with st.spinner("Training..."):
    r = train_model(df, target, tuple(features), criterion, max_depth, min_leaf, test_size)

model, tree = r["model"], r["model"].named_steps["tree"]
classes = [str(c) for c in tree.classes_]
pred_te = model.predict(r["X_te"])
test_acc = accuracy_score(r["y_te"], pred_te)
train_acc = accuracy_score(r["y_tr"], model.predict(r["X_tr"]))

tab_data, tab_res, tab_tree, tab_pred = st.tabs(["📊 Data", "🎯 Results", "🌳 Tree", "🔮 Predict"])

# ---------------------------------------------------------------- data tab
with tab_data:
    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", f"{len(df):,}")
    c2.metric("Columns", len(df.columns))
    c3.metric("Missing values", f"{int(df.isna().sum().sum()):,}")
    st.subheader("Preview")
    st.dataframe(df.head(50), width="stretch")
    left, right = st.columns(2)
    with left:
        st.subheader(f"Distribution of '{target}'")
        st.bar_chart(clean[target].astype(str).value_counts())
    with right:
        st.subheader("Missing values per column")
        miss = df.isna().sum()
        if miss.sum() == 0:
            st.success("No missing values 🎉")
        else:
            st.bar_chart(miss[miss > 0])
    with st.expander("Summary statistics"):
        st.dataframe(df.describe(include="all").T, width="stretch")

# ---------------------------------------------------------------- results tab
with tab_res:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Test accuracy", f"{test_acc:.1%}")
    c2.metric("Train accuracy", f"{train_acc:.1%}")
    c3.metric("Tree depth", tree.get_depth())
    c4.metric("Leaves", tree.get_n_leaves())
    if train_acc - test_acc > 0.10:
        st.warning("Train accuracy is much higher than test accuracy: the tree is likely **overfitting**. Try a smaller max depth.")
    st.caption(f"Trained on {len(r['X_tr'])} rows, tested on {len(r['X_te'])} rows. Missing values are filled automatically.")

    left, right = st.columns(2)
    with left:
        st.subheader("Confusion matrix")
        fig, ax = plt.subplots(figsize=(5, 4))
        ConfusionMatrixDisplay.from_predictions(
            r["y_te"].astype(str), pd.Series(pred_te).astype(str), labels=classes,
            ax=ax, cmap="Blues", colorbar=False,
        )
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")
        fig.tight_layout()
        st.pyplot(fig)
    with right:
        st.subheader("Feature importance")
        imp = pd.Series(tree.feature_importances_, index=r["feature_names"]).sort_values(ascending=False)
        st.bar_chart(imp)
        st.caption("How much each column helped the tree make its splits.")

    with st.expander("Detailed report (precision / recall / F1)"):
        rep = classification_report(r["y_te"].astype(str), pd.Series(pred_te).astype(str), output_dict=True, zero_division=0)
        st.dataframe(pd.DataFrame(rep).T.round(3), width="stretch")

    buf = io.BytesIO()
    joblib.dump({"model": model, "features": list(features), "target": target}, buf)
    st.download_button("⬇️ Download trained model (.pkl)", buf.getvalue(), "decision_tree_model.pkl")

# ---------------------------------------------------------------- tree tab
with tab_tree:
    depth_shown = st.slider("Levels to draw", 1, 5, min(3, max(tree.get_depth(), 1)))
    width = {1: 8, 2: 12, 3: 18, 4: 24, 5: 30}[depth_shown]
    fig, ax = plt.subplots(figsize=(width, 3 + 2 * depth_shown))
    plot_tree(
        tree, feature_names=r["feature_names"], class_names=classes, filled=True,
        rounded=True, max_depth=depth_shown, fontsize=9, ax=ax,
    )
    st.pyplot(fig)
    st.caption("Text columns are turned into numbers (e.g. 'Sex <= 0.5'). The mapping is below.")
    enc_cols = r["cat_cols"]
    if enc_cols:
        with st.expander("How text values were encoded"):
            enc = model.named_steps["pre"].named_transformers_["cat"].named_steps["enc"]
            for col, cats in zip(enc_cols, enc.categories_):
                st.write(f"**{col}**: " + ", ".join(f"{v} → {i}" for i, v in enumerate(cats)))
    with st.expander("Tree as plain-text rules"):
        st.code(export_text(tree, feature_names=r["feature_names"], max_depth=depth_shown))

# ---------------------------------------------------------------- predict tab
with tab_pred:
    st.subheader("Try your own input")
    X_all = prepare_X(clean, features)
    with st.form("predict_form"):
        cols = st.columns(3)
        values = {}
        for i, c in enumerate(r["feature_names"]):
            with cols[i % 3]:
                s = X_all[c]
                if c in r["cat_cols"]:
                    values[c] = st.selectbox(c, list(s.dropna().unique()))
                elif pd.api.types.is_integer_dtype(s):
                    values[c] = st.number_input(c, value=int(s.median()), step=1)
                else:
                    values[c] = st.number_input(c, value=float(s.median()))
        go = st.form_submit_button("Predict", type="primary")

    if go:
        row = pd.DataFrame([values], columns=list(features))
        label = model.predict(row)[0]
        probs = pd.Series(model.predict_proba(row)[0], index=classes)
        st.success(f"Predicted **{target}**: **{label}**  (confidence {probs.max():.0%})")
        st.bar_chart(probs)