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
def load_data(config):
    # Change paths to relative from project root
    root = os.path.join(os.path.dirname(__file__), '..')
    config['data']['train_path'] = os.path.join(root, config['data']['train_path'])
    config['data']['test_path'] = os.path.join(root, config['data']['test_path'])
    config['data']['rul_path'] = os.path.join(root, config['data']['rul_path'])
    
    handler = CMAPSSDataHandler(config)
    try:
        train_df, test_df, truth_df = handler.load_data()
        test_df = handler.prepare_test_validation(test_df, truth_df)
        return test_df, handler.features
    except Exception as e:
        return None, None

def main():
    st.title("🚀 NASA Turbofan Anomaly Detection Dashboard")
    st.markdown("Monitor engine health, predict Remaining Useful Life (RUL), and detect anomalies in real-time.")
    
    config = load_config()
    test_df, features = load_data(config)
    
    if test_df is None:
        st.error("Data not found. Please ensure the NASA CMAPSS data files are placed in the `data/` folder inside the project root.")
        return
        
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
        st.sidebar.info("Model metrics not found. Run `src/predict.py` to generate them.")
        
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("RUL Degradation Over Time")
        fig_rul = px.line(engine_data, x="time_cycles", y="RUL", title=f"Engine {engine_id} Remaining Useful Life")
        fig_rul.add_hline(y=30, line_dash="dash", line_color="red", annotation_text="Critical Failure Zone")
        fig_rul.update_layout(xaxis_title="Operational Cycles", yaxis_title="Remaining Useful Life (Cycles)")
        st.plotly_chart(fig_rul, use_container_width=True)

    with col2:
        st.subheader("Sensor Interactive View")
        selected_sensor = st.selectbox("Select Sensor", features)
        fig_sensor = px.line(engine_data, x="time_cycles", y=selected_sensor, title=f"Sensor {selected_sensor} Readings")
        st.plotly_chart(fig_sensor, use_container_width=True)
        
    st.subheader("Anomaly Heatmap (Multivariate Sensor View)")
    # Normalize features for heatmap visualization
    heatmap_data = engine_data[features].apply(lambda x: (x - x.mean()) / (x.std() + 1e-8))
    fig_heatmap = go.Figure(data=go.Heatmap(
                    z=heatmap_data.T.values,
                    x=engine_data['time_cycles'],
                    y=features,
                    colorscale='Magma'))
    fig_heatmap.update_layout(title="Standardized Sensor Readings Heatmap", xaxis_title="Cycles", yaxis_title="Sensors")
    st.plotly_chart(fig_heatmap, use_container_width=True)

if __name__ == '__main__':
    main()
