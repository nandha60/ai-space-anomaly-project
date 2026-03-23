import os
import yaml
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import numpy as np
from tqdm import tqdm
from data import get_dataloaders
from model import BiLSTMAttentionAE, loss_function
from sklearn.metrics import mean_squared_error, f1_score

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_loss = None
        self.early_stop = False

    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif self.mode == 'min' and val_loss > self.best_loss - self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

def train_model():
    with open('config.yaml', 'r') as file:
        config = yaml.safe_load(file)
        
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Dataloaders
    train_loader, val_loader, test_loader, handler = get_dataloaders(config)
    
    # Model
    model = BiLSTMAttentionAE(
        input_dim=config['model']['input_dim'],
        hidden_dim=config['model']['hidden_dim'],
        lstm_layers=config['model']['lstm_layers'],
        attention_heads=config['model']['attention_heads'],
        dropout=config['model']['dropout']
    ).to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=config['train']['learning_rate'], weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    early_stopping = EarlyStopping(patience=config['train']['patience'])
    
    # Tensorboard
    os.makedirs(config['train']['log_dir'], exist_ok=True)
    os.makedirs(config['train']['save_dir'], exist_ok=True)
    writer = SummaryWriter(log_dir=config['train']['log_dir'])
    
    best_val_rmse = float('inf')
    
    epochs = config['train']['epochs']
    
    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        train_recon = 0.0
        train_rul = 0.0
        
        loop = tqdm(train_loader, leave=False, desc=f"Epoch {epoch+1}/{epochs}")
        for batch_x, batch_y in loop:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            
            optimizer.zero_grad()
            x_recon, rul_pred, mu, logvar, _ = model(batch_x)
            
            loss, recon_loss, rul_loss, kl_loss = loss_function(
                x_recon, batch_x, rul_pred, batch_y, mu, logvar, alpha=1.0, beta=0.001
            )
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            
            train_loss += loss.item()
            train_recon += recon_loss.item()
            train_rul += rul_loss.item()
            
            loop.set_postfix(loss=loss.item())

        train_loss /= len(train_loader)
        train_recon /= len(train_loader)
        train_rul /= len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        preds, trues = [], []
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                
                x_recon, rul_pred, mu, logvar, _ = model(batch_x)
                
                loss, recon_loss, rul_loss, kl_loss = loss_function(
                    x_recon, batch_x, rul_pred, batch_y, mu, logvar, alpha=1.0, beta=0.001
                )
                
                val_loss += loss.item()
                preds.append(rul_pred.cpu().numpy())
                trues.append(batch_y.cpu().numpy())
                
        val_loss /= len(val_loader)
        
        preds = np.concatenate(preds).flatten()
        trues = np.concatenate(trues).flatten()
        val_rmse = np.sqrt(mean_squared_error(trues, preds))
        
        # We can simulate an Anomaly F1 by assuming RUL < 30 is Anomaly
        true_anomaly = (trues < 30).astype(int)
        pred_anomaly = (preds < 30).astype(int)
        val_f1 = f1_score(true_anomaly, pred_anomaly, zero_division=0)
        
        print(f"Epoch [{epoch+1}/{epochs}] | Train Loss: {train_loss:.4f} | Val RMSE: {val_rmse:.4f} | Val F1: {val_f1:.4f}")
        
        writer.add_scalar('Loss/Train', train_loss, epoch)
        writer.add_scalar('Loss/Val', val_loss, epoch)
        writer.add_scalar('Metrics/Val_RMSE', val_rmse, epoch)
        writer.add_scalar('Metrics/Val_F1', val_f1, epoch)
        
        scheduler.step(val_rmse)
        early_stopping(val_rmse)
        
        if val_rmse < best_val_rmse:
            best_val_rmse = val_rmse
            torch.save(model.state_dict(), os.path.join(config['train']['save_dir'], 'best_model.pth'))
            
        if early_stopping.early_stop:
            print("Early Stopping triggered.")
            break

if __name__ == '__main__':
    train_model()
