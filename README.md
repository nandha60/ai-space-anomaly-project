# Advanced NASA Turbofan Anomaly Detection (Production-Ready)

## Overview
This project implements an advanced Anomaly Detection and Remaining Useful Life (RUL) prediction system for the NASA CMAPSS FD001 dataset. It features a BiLSTM + Attention Autoencoder with anomaly scoring based on Reconstruction MSE and KL Divergence.

## Features
- **Data Pipeline:** Piecewise linear RUL calculation, MinMax & Z-score normalization, 30-cycle windowing.
- **Model:** Bidirectional LSTM (128 -> 64) with Multi-head self-attention.
- **Training:** Early stopping, LR scheduling, validation RMSE checking, TensorBoard logging.
- **Dashboard:** Production Streamlit dashboard with interactive sensor plots, anomaly heatmaps, and RUL predictions.

## Setup
1. Create a virtual environment and install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
2. Place the NASA dataset files (`train_FD001.txt`, `test_FD001.txt`, `RUL_FD001.txt`) inside the `data/` folder.
3. Run tests:
   ```bash
   pytest tests/
   ```
4. Train the model:
   ```bash
   python src/train.py
   ```
5. Run the Streamlit Dashboard:
   ```bash
   streamlit run dashboard/app.py
   ```
