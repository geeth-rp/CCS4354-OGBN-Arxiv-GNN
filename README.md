# OGBN-Arxiv Graph Neural Networks

GNN coursework on OGBN-Arxiv dataset featuring GCN and GAT models, along with an interactive Streamlit results dashboard.

## Repository Structure

- `notebooks/`: Contains the main Jupyter Notebook for data processing, model training, and evaluation.
- `dashboard/`: Contains the Streamlit dashboard application and its corresponding output artifacts.
  - `app.py`: Streamlit application script.
  - `outputs/`: Stored outputs including trained model checkpoints (`.pt`), metrics (`.csv`), predictions (`.npy`), and visualization plots (`.png`).
- `docs/`: Placeholder directory for future reports and write-ups.
- `requirements.txt`: Project dependencies for both the notebook and dashboard.

## Setup & Installation

Install the required dependencies using pip:
```bash
pip install -r requirements.txt
```

## Running the Notebook

To run the notebook:
1. Start Jupyter Notebook from the root of this project:
   ```bash
   jupyter notebook notebooks/CCS4354_OGBN_Arxiv_Coursework.ipynb
   ```
2. Run through the cells to load data, train the GCN and GAT models, and export results to the `dashboard/outputs/` directory.

## Running the Dashboard

Ensure the dependencies are installed and the model outputs are present in `dashboard/outputs/`. Then launch the dashboard:
```bash
python -m streamlit run dashboard/app.py
```

## Models and Results Summary

This repository trains two types of Graph Neural Networks on the OGBN-Arxiv citation dataset:
- **GCN (Graph Convolutional Network)**
- **GAT (Graph Attention Network)**

Trained checkpoints for both models are saved as `gcn_best.pt` and `gat_best.pt`.
- `results_df.csv` contains test set metrics (e.g., accuracy and F1 score) to compare models.
- `predictions.npy` stores the inference results.
- `true_labels.npy` holds the actual classes for evaluation.

## Individual Contribution

| Name | Student ID | Contribution |
| :--- | :--- | :--- |
| | | |
