import os
import sys
import yaml
import json
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

# Add src to path to import model and data
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from data import CMAPSSDataHandler

st.set_page_config(page_title="NASA Turbofan Anomaly Detection", layout="wide", page_icon="🚀")

@st.cache_data
def load_config():
    config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
    with open(config_path, 'r') as file:
        return yaml.safe_load(file)

@st.cache_data
def load_data(config, dataset_id):
    # Change paths to relative from project root based on selected dataset
    root = os.path.join(os.path.dirname(__file__), '..')
    
    # Override config paths based on selection (e.g., FD001)
    config['data']['train_path'] = os.path.join(root, f"data/CMAPSSData/train_{dataset_id}.txt")
    config['data']['test_path'] = os.path.join(root, f"data/CMAPSSData/test_{dataset_id}.txt")
    config['data']['rul_path'] = os.path.join(root, f"data/CMAPSSData/RUL_{dataset_id}.txt")
    
    handler = CMAPSSDataHandler(config)
    try:
        train_df, test_df, truth_df = handler.load_data()
        test_df = handler.prepare_test_validation(test_df, truth_df)
        return test_df, handler.features
    except Exception as e:
        return None, None

def main():
    st.title("🚀 NASA Turbofan Anomaly Detection Dashboard")
    st.markdown("Monitor engine health, predict Remaining Useful Life (RUL), and detect anomalies in real-time across NASA CMAPSS constraints.")
    
    st.sidebar.header("Data Selection")
    # Added dynamic selection to rotate between the differing constraint files natively
    dataset_id = st.sidebar.selectbox("Select NASA Dataset Component", ["FD001", "FD002", "FD003", "FD004"])
    
    config = load_config()
    test_df, features = load_data(config, dataset_id)
    
    if test_df is None:
        st.error(f"Data not found for {dataset_id}. Please ensure the underlying data files are correctly placed inside the `data/CMAPSSData/` directory!")
        return
        
    st.sidebar.markdown("---")
    
    engine_id = st.sidebar.selectbox("Select Engine Unit ID", test_df['unit_nr'].unique())
    
    engine_data = test_df[test_df['unit_nr'] == engine_id].copy()
    
    st.sidebar.markdown("---")
    st.sidebar.header("Model Metrics")
    
    metrics_file = os.path.join(os.path.dirname(__file__), '..', 'results', 'metrics.json')
    if os.path.exists(metrics_file):
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
            st.sidebar.metric(label="Validation RMSE", value=f"{metrics.get('Test_RMSE', 0):.2f}")
            st.sidebar.metric(label="Anomaly F1 Score", value=f"{metrics.get('Anomaly_F1', 0):.2f}")
            st.sidebar.metric(label="Anomaly Score Threshold", value=f"{metrics.get('Anomaly_Threshold', 0):.2f}")
    else:
        st.sidebar.info("Model metrics not found. Run `src/predict.py` to natively generate them.")
        
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("RUL Degradation Over Time")
        fig_rul = px.line(engine_data, x="time_cycles", y="RUL", title=f"Dataset {dataset_id} - Engine {engine_id} Remaining Useful Life")
        fig_rul.add_hline(y=30, line_dash="dash", line_color="red", annotation_text="Critical Failure Zone")
        fig_rul.update_layout(xaxis_title="Operational Cycles", yaxis_title="Remaining Useful Life (Cycles)")
        st.plotly_chart(fig_rul, use_container_width=True)

    with col2:
        st.subheader("Sensor Interactive View")
        selected_sensor = st.selectbox("Select Sensor", features)
        fig_sensor = px.line(engine_data, x="time_cycles", y=selected_sensor, title=f"Sensor {selected_sensor} Dynamic Readings")
        st.plotly_chart(fig_sensor, use_container_width=True)
        
    st.subheader(f"Anomaly Heatmap (Multivariate Sensor View - {dataset_id} Unit {engine_id})")
    # Normalize features for heatmap visualization iteratively
    heatmap_data = engine_data[features].apply(lambda x: (x - x.mean()) / (x.std() + 1e-8))
    fig_heatmap = go.Figure(data=go.Heatmap(
                    z=heatmap_data.T.values,
                    x=engine_data['time_cycles'],
                    y=features,
                    colorscale='Magma'))
    fig_heatmap.update_layout(title="Standardized Cross-Sensor Readings Topography", xaxis_title="Cycles", yaxis_title="Dimensional Sensors")
    st.plotly_chart(fig_heatmap, use_container_width=True)

if __name__ == '__main__':
    main()
