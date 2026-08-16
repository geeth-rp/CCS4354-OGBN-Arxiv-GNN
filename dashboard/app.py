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

# Custom CSS injection for UI restyling
st.markdown("""
<style>
    /* Import Google Font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    /* Global Typography & Background */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif !important;
    }
    .stApp {
        background-color: #F8F9FA;
    }

    /* Headings */
    h1, h2, h3, h4, h5, h6 {
        color: #1B4332 !important;
        font-weight: 700 !important;
    }
    p, span, label {
        color: #333333;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF !important;
        box-shadow: 2px 0 12px rgba(0, 0, 0, 0.05);
    }

    /* Cards for Metrics */
    [data-testid="metric-container"] {
        background-color: #FFFFFF;
        border-radius: 20px;
        padding: 24px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
        border: 1px solid rgba(27, 67, 50, 0.05);
        margin-bottom: 1rem;
    }
    [data-testid="stMetricLabel"] p {
        color: #6C757D !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
    }
    [data-testid="stMetricValue"] {
        color: #1B4332 !important;
        font-weight: 700 !important;
        font-size: 2.2rem !important;
    }
    [data-testid="stMetricDelta"] svg {
        color: #74C69D !important;
    }
    
    /* Images / Charts wrappers */
    [data-testid="stImage"] {
        background-color: #FFFFFF;
        border-radius: 20px;
        padding: 16px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
        border: 1px solid rgba(27, 67, 50, 0.05);
    }
    [data-testid="stImage"] img {
        border-radius: 12px;
    }
    [data-testid="stImage"] caption {
        color: #6C757D !important;
        font-weight: 500;
        margin-top: 10px;
    }

    /* Buttons */
    .stButton > button {
        background-color: #1B4332 !important;
        color: #FFFFFF !important;
        border-radius: 50px !important;
        border: none !important;
        padding: 0.6rem 1.5rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease;
        box-shadow: 0 4px 10px rgba(27, 67, 50, 0.2);
    }
    .stButton > button:hover {
        background-color: #2D6A4F !important;
        box-shadow: 0 6px 14px rgba(45, 106, 79, 0.3);
        transform: translateY(-2px);
    }
    
    /* Number Input */
    .stNumberInput input {
        border-radius: 12px !important;
        border: 1px solid #E0E0E0 !important;
        padding: 0.5rem !important;
    }
    
    /* Alerts */
    .stAlert {
        border-radius: 16px !important;
        border: none !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.03);
    }

    /* Dataframe wrapper styling */
    [data-testid="stDataFrame"] {
        border-radius: 20px;
        overflow: hidden;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.05);
        border: 1px solid rgba(27, 67, 50, 0.05);
    }

    /* White spacing between sections */
    .stHorizontalBlock {
        gap: 2rem !important;
    }
</style>
""", unsafe_allow_html=True)

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
