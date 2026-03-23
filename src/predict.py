import os
import yaml
import json
import torch
import numpy as np
from tqdm import tqdm
from data import get_dataloaders
from model import BiLSTMAttentionAE
from sklearn.metrics import mean_squared_error, f1_score

def compute_anomaly_score(x_recon, x, mu, logvar):
    # x_recon, x: (batch, seq, features)
    # mu, logvar: (batch, latent_dim)
    
    # MSE per sample: mean across seq length and features
    mse_per_sample = torch.mean((x_recon - x)**2, dim=[1, 2])
    
    # KL per sample: sum across latent dim
    kl_per_sample = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1)
    
    # Anomaly score
    anomaly_score = mse_per_sample + 0.001 * kl_per_sample
    return anomaly_score

def predict():
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Dataloader
    _, _, test_loader, _ = get_dataloaders(config)
    
    # Model Setup
    model = BiLSTMAttentionAE(
        input_dim=config['model']['input_dim'],
        hidden_dim=config['model']['hidden_dim'],
        lstm_layers=config['model']['lstm_layers'],
        attention_heads=config['model']['attention_heads'],
        dropout=config['model']['dropout']
    ).to(device)
    
    model_path = os.path.join(config['train']['save_dir'], 'best_model.pth')
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Loaded best_model.pth successfully.")
    else:
        print("No pre-trained model found. Initialized with random weights.")
        
    model.eval()
    all_preds = []
    all_trues = []
    all_anomaly_scores = []
    
    with torch.no_grad():
        for batch_x, batch_y in tqdm(test_loader, desc="Predicting"):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            x_recon, rul_pred, mu, logvar, _ = model(batch_x)
            
            anomaly_score = compute_anomaly_score(x_recon, batch_x, mu, logvar)
            
            all_preds.append(rul_pred.cpu().numpy())
            all_trues.append(batch_y.cpu().numpy())
            all_anomaly_scores.append(anomaly_score.cpu().numpy())
            
    all_preds = np.concatenate(all_preds).flatten()
    all_trues = np.concatenate(all_trues).flatten()
    all_anomaly_scores = np.concatenate(all_anomaly_scores).flatten()
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(all_trues, all_preds))
    print(f"Test RMSE: {rmse:.4f}")
    
    # F1 Score calculation
    # For CMAPSS, we assume engine failure zone (RUL < 30) correlates with anomalous behavior
    true_anomaly = (all_trues < 30).astype(int)
    
    # Set threshold dynamically or heuristically
    threshold = np.percentile(all_anomaly_scores, 90)
    pred_anomaly = (all_anomaly_scores > threshold).astype(int)
    
    f1 = f1_score(true_anomaly, pred_anomaly, zero_division=0)
    print(f"Anomaly F1: {f1:.4f}")
    
    # Save results
    os.makedirs('results', exist_ok=True)
    results = {
        'Test_RMSE': float(rmse),
        'Anomaly_F1': float(f1),
        'Anomaly_Threshold': float(threshold)
    }
    with open('results/metrics.json', 'w') as f:
        json.dump(results, f, indent=4)
        
    print("Metrics saved to results/metrics.json")

if __name__ == '__main__':
    predict()
