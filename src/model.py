import torch
import torch.nn as nn
import torch.nn.functional as F

class AttentionBlock(nn.Module):
    def __init__(self, hidden_dim, num_heads, dropout=0.2):
        super(AttentionBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(hidden_dim)
        
    def forward(self, x):
        # x is (batch_size, seq_len, hidden_dim)
        attn_out, weights = self.attention(x, x, x)
        out = self.norm(x + attn_out)
        return out, weights

class BiLSTMAttentionAE(nn.Module):
    def __init__(self, input_dim=14, hidden_dim=128, lstm_layers=2, attention_heads=4, dropout=0.2):
        super(BiLSTMAttentionAE, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        # Encoder
        # 1st Layer: input_dim -> hidden_dim (128 per dir, 256 total)
        self.encoder_lstm1 = nn.LSTM(input_dim, hidden_dim, batch_first=True, bidirectional=True)
        # 2nd Layer: 256 -> hidden_dim // 2 (64 per dir, 128 total)
        self.encoder_lstm2 = nn.LSTM(hidden_dim * 2, hidden_dim // 2, batch_first=True, bidirectional=True)
        
        # Attention
        self.attention = AttentionBlock(hidden_dim, attention_heads, dropout)
        
        # Variational (latent space)
        self.fc_mu = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc_logvar = nn.Linear(hidden_dim, hidden_dim // 2)
        
        # RUL Prediction
        self.rul_regressor = nn.Sequential(
            nn.Linear(hidden_dim // 2, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1)
        )
        
        # Decoder
        self.decoder_input = nn.Linear(hidden_dim // 2, hidden_dim)
        # Decoder receives context so we use single-direction LSTM to reconstruct sequence
        self.decoder_lstm1 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, bidirectional=False)
        self.decoder_lstm2 = nn.LSTM(hidden_dim, hidden_dim, batch_first=True, bidirectional=False)
        
        self.reconstructor = nn.Linear(hidden_dim, input_dim)

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        # x: (batch_size, seq_len, input_dim)
        seq_len = x.size(1)
        batch_size = x.size(0)
        
        # Encoder
        out, _ = self.encoder_lstm1(x)
        out, _ = self.encoder_lstm2(out) # out: (batch_size, seq_len, 128)
        
        # Attention
        attn_out, attn_weights = self.attention(out)
        
        # Context vector (Global Average Pooling over the sequence)
        context = torch.mean(attn_out, dim=1) # (batch_size, 128)
        
        # Latent Space (Variational)
        mu = self.fc_mu(context)        # (batch_size, 64)
        logvar = self.fc_logvar(context) # (batch_size, 64)
        z = self.reparameterize(mu, logvar) # (batch_size, 64)
        
        # RUL Prediction
        rul_pred = self.rul_regressor(z) # (batch_size, 1)
        
        # Decoder
        dec_in = self.decoder_input(z) # (batch_size, 128)
        dec_in = dec_in.unsqueeze(1).repeat(1, seq_len, 1) # (batch_size, seq_len, 128)
        
        dec_out, _ = self.decoder_lstm1(dec_in)
        dec_out, _ = self.decoder_lstm2(dec_out)
        
        # Reconstruction
        x_recon = self.reconstructor(dec_out) # (batch_size, seq_len, 14)
        
        return x_recon, rul_pred, mu, logvar, attn_weights

def loss_function(x_recon, x, rul_pred, rul_true, mu, logvar, alpha=1.0, beta=1e-3):
    # Reconstruction Loss
    recon_loss = F.mse_loss(x_recon, x, reduction='mean')
    
    # RUL Prediction Loss (Forecasting)
    rul_true = rul_true.view(-1, 1)
    rul_loss = F.mse_loss(rul_pred, rul_true, reduction='mean')
    
    # KL Divergence
    kl_loss = -0.5 * torch.mean(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Total loss
    total_loss = alpha * rul_loss + recon_loss + beta * kl_loss
    
    return total_loss, recon_loss, rul_loss, kl_loss
