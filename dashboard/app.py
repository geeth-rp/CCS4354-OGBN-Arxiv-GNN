"""
CCS4354 Task 08 – Graph Intelligence Dashboard
Run with: streamlit run app.py

Expects these files (copy them from the notebook's working directory into this
same folder before running):
    - gcn_best.pt, gat_best.pt
    - degree_distribution.png, feature_distribution.png, subgraph_sample.png
    - training_curves.png, embedding_tsne.png
    - results_df.csv   (save with: results_df.to_csv("results_df.csv", index=False) in the notebook)
    - predictions.npy  (save with: np.save("predictions.npy", preds) for the model you want to browse)
    - true_labels.npy  (save with: np.save("true_labels.npy", data.y.cpu().numpy()))
"""

import streamlit as st
import pandas as pd
import numpy as np
import os

st.set_page_config(page_title="OGBN-Arxiv Graph Intelligence Dashboard", layout="wide")

st.title("OGBN-Arxiv Graph Intelligence Dashboard")
st.caption("CCS4354 — Tensors and Graphs Coursework | Task 08")

DATA_DIR = os.path.join(os.path.dirname(__file__), "outputs")


def safe_image(filename, caption):
    path = os.path.join(DATA_DIR, filename)
    if os.path.exists(path):
        st.image(path, caption=caption, use_container_width=True)
    else:
        st.info(f"'{filename}' not found — copy it here from the notebook output first.")


# ---------------- Sidebar ----------------
st.sidebar.header("Navigation")
section = st.sidebar.radio(
    "Go to",
    ["Graph Statistics", "Model Performance", "Node Classification", "Embedding Visualization"],
)

# ---------------- Graph Statistics ----------------
if section == "Graph Statistics":
    st.header("Graph Statistics")
    col1, col2, col3 = st.columns(3)
    col1.metric("Nodes", "~169,343")
    col2.metric("Edges", "~1,166,243")
    col3.metric("Feature Dimension", "128")

    c1, c2 = st.columns(2)
    with c1:
        safe_image("degree_distribution.png", "Degree distribution (log scale)")
    with c2:
        safe_image("subgraph_sample.png", "Sample citation subgraph")
    safe_image("feature_distribution.png", "Node feature value distribution")

# ---------------- Model Performance ----------------
elif section == "Model Performance":
    st.header("Model Performance")

    results_path = os.path.join(DATA_DIR, "results_df.csv")
    if os.path.exists(results_path):
        results_df = pd.read_csv(results_path)
        st.dataframe(results_df, use_container_width=True)

        st.subheader("Test-set comparison")
        test_results = results_df[results_df["split"] == "test"]
        st.bar_chart(test_results.set_index("model")[["accuracy", "f1_macro"]])
    else:
        st.info("'results_df.csv' not found — save it from the notebook: "
                "`results_df.to_csv('results_df.csv', index=False)`")

    safe_image("training_curves.png", "Training / validation loss & accuracy curves")

# ---------------- Node Classification ----------------
elif section == "Node Classification":
    st.header("Node Classification Lookup")

    preds_path = os.path.join(DATA_DIR, "predictions.npy")
    labels_path = os.path.join(DATA_DIR, "true_labels.npy")

    if os.path.exists(preds_path) and os.path.exists(labels_path):
        preds = np.load(preds_path)
        true_labels = np.load(labels_path)

        node_id = st.number_input(
            "Enter Node ID", min_value=0, max_value=len(preds) - 1, value=0, step=1
        )
        col1, col2 = st.columns(2)
        col1.metric("Predicted class", int(preds[node_id]))
        col2.metric("True class", int(true_labels[node_id]))

        if preds[node_id] == true_labels[node_id]:
            st.success("✅ Correct prediction")
        else:
            st.error("❌ Incorrect prediction")
    else:
        st.info("'predictions.npy' and 'true_labels.npy' not found — save them from the "
                "notebook after running inference on the full node set.")

# ---------------- Embedding Visualization ----------------
elif section == "Embedding Visualization":
    st.header("Embedding Visualization")
    safe_image("embedding_tsne.png", "t-SNE projection of learned node embeddings, colored by class")
    st.caption(
        "Tight, separated clusters indicate the model has learned embeddings that "
        "group papers of the same subject category together."
    )
