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
    
    /* Mesh Gradient Background */
    .stApp {
        background-color: #f0f4f8;
        background-image: 
            radial-gradient(at 10% 20%, rgba(205, 240, 225, 0.7) 0px, transparent 50%),
            radial-gradient(at 80% 10%, rgba(215, 230, 250, 0.7) 0px, transparent 50%),
            radial-gradient(at 40% 60%, rgba(175, 195, 245, 0.6) 0px, transparent 50%),
            radial-gradient(at 90% 80%, rgba(245, 230, 215, 0.7) 0px, transparent 50%),
            radial-gradient(at 10% 90%, rgba(195, 205, 245, 0.7) 0px, transparent 50%);
        background-attachment: fixed;
    }

    /* Headings */
    h1, h2, h3, h4, h5, h6 {
        color: #1a202c !important;
        font-weight: 700 !important;
    }
    p, span, label {
        color: #2d3748;
    }
    
    /* Sidebar - Glassmorphism */
    [data-testid="stSidebar"] {
        background: rgba(255, 255, 255, 0.6) !important;
        backdrop-filter: blur(16px) !important;
        -webkit-backdrop-filter: blur(16px) !important;
        border-right: 1px solid rgba(255, 255, 255, 0.8) !important;
        box-shadow: 2px 0 24px rgba(0, 0, 0, 0.02);
    }

    /* Cards for Metrics - Glassmorphism */
    [data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.65) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        border-radius: 20px;
        padding: 24px;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.8) !important;
        margin-bottom: 1rem;
    }
    [data-testid="stMetricLabel"] p {
        color: #4a5568 !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
    }
    [data-testid="stMetricValue"] {
        color: #1a202c !important;
        font-weight: 700 !important;
        font-size: 2.2rem !important;
    }
    [data-testid="stMetricDelta"] svg {
        color: #48bb78 !important;
    }
    
    /* Images / Charts wrappers - Glassmorphism */
    [data-testid="stImage"] {
        background: rgba(255, 255, 255, 0.65) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        border-radius: 20px;
        padding: 16px;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.8) !important;
    }
    [data-testid="stImage"] img {
        border-radius: 12px;
    }
    [data-testid="stImage"] caption {
        color: #4a5568 !important;
        font-weight: 500;
        margin-top: 10px;
    }

    /* Buttons */
    .stButton > button {
        background-color: #1a202c !important;
        color: #FFFFFF !important;
        border-radius: 50px !important;
        border: none !important;
        padding: 0.6rem 1.5rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease;
        box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
    }
    .stButton > button:hover {
        background-color: #2d3748 !important;
        box-shadow: 0 6px 14px rgba(0, 0, 0, 0.15);
        transform: translateY(-2px);
    }
    
    /* Number Input */
    .stNumberInput input {
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.8) !important;
        background: rgba(255, 255, 255, 0.5) !important;
        padding: 0.5rem !important;
    }
    
    /* Alerts */
    .stAlert {
        border-radius: 16px !important;
        border: 1px solid rgba(255, 255, 255, 0.8) !important;
        background: rgba(255, 255, 255, 0.65) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        box-shadow: 0 4px 12px rgba(0,0,0,0.03);
    }

    /* Dataframe wrapper styling - Glassmorphism */
    [data-testid="stDataFrame"] {
        background: rgba(255, 255, 255, 0.65) !important;
        backdrop-filter: blur(12px) !important;
        -webkit-backdrop-filter: blur(12px) !important;
        border-radius: 20px;
        padding: 16px;
        box-shadow: 0 8px 32px rgba(31, 38, 135, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.8) !important;
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
    ["Graph Statistics", "Model Performance", "Node Classification", "Embedding Visualization", "Attention Analysis"],
)

st.sidebar.markdown("---")
st.sidebar.subheader("Dataset Splits")
st.sidebar.caption("Train: 90,941 nodes\\nValid: 29,799 nodes\\nTest: 48,603 nodes\\nTotal: 169,343 nodes")

st.sidebar.subheader("Model Configurations")
st.sidebar.caption("**GCN:** 3 layers, hidden=256, dropout=0.5, lr=0.005\\n**GAT:** 2 layers, hidden=32, heads=8, dropout=0.5, lr=0.005")

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
        try:
            preds = np.load(preds_path)
            true_labels = np.load(labels_path)
        except Exception as e:
            st.error(f"Error loading prediction files. They may be corrupted or from an incompatible version: {e}")
            st.stop()

        splits_path = os.path.join(DATA_DIR, "node_splits.json")
        node_splits = None
        if os.path.exists(splits_path):
            import json
            with open(splits_path, 'r') as f:
                node_splits = json.load(f)

        st.info("ℹ️ **Note:** The predictions displayed are generated by the **GCN** model. The true label is shown purely for evaluation purposes. To test another model, save its predictions to `predictions.npy`.")
        
        node_id = st.number_input(
            "Enter Node ID", min_value=0, max_value=len(preds) - 1, value=0, step=1
        )
        
        try:
            pred_val = int(preds[node_id])
            true_val = int(true_labels[node_id])
            
            if node_splits:
                split_name = node_splits[node_id]
                st.caption(f"Selected Node {node_id} belongs to the **{split_name.upper()}** set.")

            col1, col2 = st.columns(2)
            col1.metric("Predicted class", pred_val)
            col2.metric("True class", true_val)

            if pred_val == true_val:
                st.success("✅ Correct prediction")
            else:
                st.error("❌ Incorrect prediction")
        except IndexError:
            st.error(f"Invalid Node ID: {node_id}. Please enter a value between 0 and {len(preds)-1}.")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
            
    else:
        st.info("'predictions.npy' and 'true_labels.npy' not found — save them from the "
                "notebook after running inference on the full node set.")

# ---------------- Embedding Visualization ----------------
elif section == "Embedding Visualization":
    st.header("Embedding Visualization")
    safe_image("embedding_tsne.png", "t-SNE projection of learned node embeddings, colored by class")
    st.caption(
        "t-SNE projection of 2,000 test-set node embeddings (perplexity=30, random_state=42). "
        "Visually separated colour clusters suggest the model has learned representations "
        "that group similar subject categories together — this is qualitative, suggestive "
        "evidence of structure, not proof of generalisation."
    )

# ---------------- Attention Analysis ----------------
elif section == "Attention Analysis":
    st.header("Attention Analysis")
    
    st.caption(
        "GAT layer-1 attention weights (averaged across 8 heads, self-loops included) "
        "for 5 sample test-set nodes, showing each node's top attended neighbours "
        "and the neighbour's true class."
    )
    
    safe_image("attention_analysis_chart.png", "Top attended neighbours per sample node")
    
    st.subheader("Attention Weights Table")
    table_path = os.path.join(DATA_DIR, "attention_analysis_table.csv")
    if os.path.exists(table_path):
        attn_df = pd.read_csv(table_path)
        st.dataframe(attn_df, use_container_width=True)
    else:
        st.info("'attention_analysis_table.csv' not found — run the attention analysis in the notebook first.")
