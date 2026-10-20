# -*- coding: utf-8 -*-
"""
VAE-based Climate Scenario Generator for H2RES.

High-fidelity Conditional Variational Autoencoder for generating synthetic
annual climate profiles (solar irradiance, wind speed, temperature) with
accurate diurnal patterns and extreme value capture.

Features:
    - 1D ResNet-based encoder/decoder architecture
    - Physics-informed loss functions (spectral, gradient, correlation)
    - Asymmetric peak penalty for extreme value fidelity
    - Google Colab compatibility with automatic dependency installation
    - Optuna hyperparameter optimization
    - Comprehensive scientific validation metrics
"""

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm
from scipy.stats import wasserstein_distance

# =============================================================================
# ENVIRONMENT DETECTION & DEPENDENCIES
# =============================================================================
try:
    import google.colab
    IS_COLAB = True
except ImportError:
    IS_COLAB = False

if IS_COLAB:
    import subprocess
    import sys
    missing_pkgs = []
    try: import optuna
    except ImportError: missing_pkgs.append("optuna")
    try: import pyarrow
    except ImportError: missing_pkgs.append("pyarrow")
    
    if missing_pkgs:
        print(f"Installing missing packages: {missing_pkgs}")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", *missing_pkgs])

import optuna

# =============================================================================
# GLOBAL CONFIGURATION
# =============================================================================
HOURS_PER_YEAR = 8760
DEFAULT_LATENT_DIM = 128
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True

# =============================================================================
# PATH MANAGEMENT
# =============================================================================
def get_path(relative_path: str) -> str:
    """Resolve paths for both local and Colab environments."""
    script_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in globals() else os.getcwd()
    possible_paths = [
        os.path.abspath(os.path.join(script_dir, relative_path)), 
        os.path.abspath(os.path.join(os.getcwd(), os.path.basename(relative_path))), 
        os.path.abspath(os.path.join("/content", os.path.basename(relative_path))), 
    ]
    if "data" in relative_path or "weights" in relative_path or "norm_params" in relative_path:
        for p in possible_paths:
            if os.path.exists(p): return p
    if IS_COLAB: return os.path.basename(relative_path)
    full_path = os.path.abspath(os.path.join(script_dir, relative_path))
    os.makedirs(os.path.dirname(full_path), exist_ok=True)
    return full_path

# =============================================================================
# DATA LOADING & PREPROCESSING
# =============================================================================
def load_data(parquet_path: str = None, stride_hours: int = 168) -> tuple:
    """
    Load and preprocess climate data for training.
    
    Args:
        parquet_path: Path to the input data file
        stride_hours: Window stride for training samples (default: 168 = 1 week)
    
    Returns:
        Tuple of (train_data, train_cond, val_data, val_cond, test_data, test_cond, data_min, data_max)
    """
    if parquet_path is None:
        parquet_path = get_path(os.path.join("..", "data", "SC", "HR_data.parquet"))
    
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Data file not found: {parquet_path}")
        
    df = pd.read_parquet(parquet_path)
    
    # Remove leap days for consistent 8760-hour years
    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'])
        is_leap_day = (df['time'].dt.month == 2) & (df['time'].dt.day == 29)
        if is_leap_day.any():
            df = df[~is_leap_day].copy()
            
    columns = ["solar", "wind", "temperature"]
    raw_data = df[columns].values.copy()
    
    # Log-transform wind for better distribution
    raw_data[:, 1] = np.log1p(raw_data[:, 1])
    
    # Split by years (80% train, 10% val, 10% test)
    num_years = len(raw_data) // HOURS_PER_YEAR
    years_data = raw_data[:num_years*HOURS_PER_YEAR].reshape(num_years, HOURS_PER_YEAR, 3)
    
    n_train = int(num_years * 0.80)
    n_val = int(num_years * 0.10)
    
    # Compute normalization from training data only
    train_years = years_data[:n_train].reshape(-1, 3)
    data_min = train_years.min(axis=0).reshape(1, 3)
    data_max = train_years.max(axis=0).reshape(1, 3)
    data_range = np.clip(data_max - data_min, 1e-9, None)
    
    def process_split(data_split, split_name):
        """Normalize data and create temporal conditioning features."""
        norm_data = (data_split - data_min) / data_range
        norm_data = np.clip(norm_data, 0.0, 1.0)
        
        T = len(norm_data)
        hours = np.arange(T)
        
        # Temporal conditioning: diurnal and annual cycles
        sin_day = np.sin(2 * np.pi * hours / 24)
        cos_day = np.cos(2 * np.pi * hours / 24)
        sin_year = np.sin(2 * np.pi * hours / HOURS_PER_YEAR)
        cos_year = np.cos(2 * np.pi * hours / HOURS_PER_YEAR)
        seasonal_data = np.stack([sin_day, cos_day, sin_year, cos_year], axis=1)
        
        # Create sliding windows
        stride = stride_hours if split_name == 'train' else HOURS_PER_YEAR
        samples_data, samples_cond = [], []
        num_windows = (T - HOURS_PER_YEAR) // stride + 1
        
        for i in range(num_windows):
            start = i * stride
            samples_data.append(norm_data[start:start+HOURS_PER_YEAR])
            samples_cond.append(seasonal_data[start:start+HOURS_PER_YEAR])
            
        return torch.FloatTensor(np.stack(samples_data)).transpose(1, 2), \
               torch.FloatTensor(np.stack(samples_cond)).transpose(1, 2)

    train_data, train_cond = process_split(train_years, 'train')
    val_data, val_cond = process_split(years_data[n_train:n_train+n_val].reshape(-1, 3), 'val')
    test_data, test_cond = process_split(years_data[n_train+n_val:].reshape(-1, 3), 'test')
    
    return train_data, train_cond, val_data, val_cond, test_data, test_cond, \
           data_min.reshape(1, 3, 1), data_max.reshape(1, 3, 1)

# =============================================================================
# MODEL ARCHITECTURE
# =============================================================================
class ResBlock1D(nn.Module):
    """1D Residual Block with GroupNorm and Dropout."""
    
    def __init__(self, channels, kernel_size=5):
        super().__init__()
        self.conv1 = nn.Conv1d(channels, channels, kernel_size, padding='same')
        self.gn1 = nn.GroupNorm(4, channels)
        self.conv2 = nn.Conv1d(channels, channels, kernel_size, padding='same')
        self.gn2 = nn.GroupNorm(4, channels)
        self.relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(0.1)

    def forward(self, x):
        residual = x
        out = self.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        return self.relu(self.dropout(out) + residual)


class CVAE_1D(nn.Module):
    """
    Conditional Variational Autoencoder for 1D time-series generation.
    
    Architecture uses 1D convolutions with residual blocks for both encoder
    and decoder, with skip connections for temporal conditioning.
    """
    
    def __init__(self, latent_dim=128):
        super().__init__()
        self.latent_dim = latent_dim
        
        # Encoder: progressively downsample temporal dimension
        self.enc_net = nn.Sequential(
            nn.Conv1d(7, 48, kernel_size=7, stride=2, padding=3),
            nn.LeakyReLU(0.2),
            ResBlock1D(48),
            nn.Conv1d(48, 96, kernel_size=7, stride=2, padding=3),
            nn.LeakyReLU(0.2),
            ResBlock1D(96),
            nn.Conv1d(96, 160, kernel_size=7, stride=2, padding=3),
            nn.LeakyReLU(0.2),
            ResBlock1D(160),
            nn.Flatten(),
            nn.Linear(160 * 1095, 512),
            nn.LeakyReLU(0.2)
        )
        self.fc_mu = nn.Linear(512, latent_dim)
        self.fc_logvar = nn.Linear(512, latent_dim)
        
        # Condition embedding
        self.cond_emb = nn.Sequential(
            nn.Conv1d(4, 16, kernel_size=5, stride=4, padding=2),
            nn.LeakyReLU(0.2),
            nn.Flatten(),
            nn.Linear(16 * 2190, 32)
        )
        
        # Decoder: progressively upsample with condition injection
        self.dec_input = nn.Linear(latent_dim + 32, 160 * 1095)
        
        self.up1 = nn.Upsample(scale_factor=2, mode='linear', align_corners=False)
        self.conv1 = nn.Conv1d(160 + 4, 96, kernel_size=7, padding='same')
        self.gn1 = nn.GroupNorm(8, 96)
        self.res1 = ResBlock1D(96)
        
        self.up2 = nn.Upsample(scale_factor=2, mode='linear', align_corners=False)
        self.conv2 = nn.Conv1d(96 + 4, 48, kernel_size=7, padding='same')
        self.gn2 = nn.GroupNorm(4, 48)
        self.res2 = ResBlock1D(48)
        
        self.up3 = nn.Upsample(scale_factor=2, mode='linear', align_corners=False)
        self.conv3 = nn.Conv1d(48 + 4, 32, kernel_size=7, padding='same')
        self.gn3 = nn.GroupNorm(4, 32)
        self.res3 = ResBlock1D(32)
        
        self.final_conv = nn.Sequential(
            nn.Conv1d(32, 3, kernel_size=7, padding='same'),
            nn.Sigmoid()
        )
        self.relu = nn.LeakyReLU(0.2)

    def reparameterize(self, mu, logvar):
        """Reparameterization trick with clamped logvar for stability."""
        std = torch.exp(0.5 * torch.clamp(logvar, -10.0, 5.0))
        return mu + torch.randn_like(std) * std

    def decode(self, z, cond):
        """Decode latent vector to output, with condition injection at each scale."""
        batch_size = z.size(0)
        z_cond = torch.cat([z, self.cond_emb(cond)], dim=1)
        h = self.dec_input(z_cond).view(batch_size, 160, 1095)
        
        for up, conv, gn, res, size in [
            (self.up1, self.conv1, self.gn1, self.res1, 2190),
            (self.up2, self.conv2, self.gn2, self.res2, 4380),
            (self.up3, self.conv3, self.gn3, self.res3, 8760)
        ]:
            h = up(h)
            c = F.interpolate(cond, size=size, mode='linear', align_corners=False)
            h = res(self.relu(gn(conv(torch.cat([h, c], dim=1)))))
        return self.final_conv(h)

    def forward(self, x, cond):
        """Full forward pass through encoder and decoder."""
        h = self.enc_net(torch.cat([x, cond], dim=1))
        mu_z, logvar_z = self.fc_mu(h), self.fc_logvar(h)
        z = self.reparameterize(mu_z, logvar_z)
        mu_x = self.decode(z, cond)
        return mu_x, None, mu_z, logvar_z
    
    def sample(self, cond, temperature=1.0):
        """Generate samples from the learned distribution."""
        self.eval()
        with torch.no_grad():
            z = torch.randn(cond.size(0), self.latent_dim).to(DEVICE) * temperature
            return self.decode(z, cond)

# =============================================================================
# LOSS FUNCTION
# =============================================================================
def vae_loss_1d(mu_x, x, mu_z, logvar_z, beta=0.0001):
    """
    Physics-informed VAE loss with asymmetric peak penalty.
    
    Components:
        - Channel-wise MSE with peak-aware asymmetric penalty
        - Wind-specific spectral and variance constraints
        - Temperature bias/variance matching
        - KL divergence regularization
        - Temporal gradient loss for volatility capture
        - Spectral loss for diurnal pattern matching
        - Solar-Temperature correlation preservation
    """
    
    def peak_aware_loss(pred, target, peak_factor=10.0):
        """Penalize underestimation of peaks more heavily than overestimation."""
        error = (target - pred)**2
        asymm_penalty = torch.where(pred < target, peak_factor * error, error)
        return torch.mean(asymm_penalty)
    
    # Channel-specific reconstruction losses
    mse_solar = peak_aware_loss(mu_x[:, 0], x[:, 0], peak_factor=15.0) * 50000.0
    mse_wind = peak_aware_loss(mu_x[:, 1], x[:, 1], peak_factor=8.0) * 15000.0
    mse_temp = peak_aware_loss(mu_x[:, 2], x[:, 2], peak_factor=10.0) * 200000.0
    
    # Wind-specific spectral constraint for volatility
    fft_wind_real = torch.fft.rfft(x[:, 1], dim=1)
    fft_wind_pred = torch.fft.rfft(mu_x[:, 1], dim=1)
    wind_spectral = torch.mean(torch.abs(torch.abs(fft_wind_real) - torch.abs(fft_wind_pred))) * 20000.0
    
    # Wind variance floor (prevent collapse to flat line)
    wind_var_real = torch.var(x[:, 1])
    wind_var_pred = torch.var(mu_x[:, 1])
    var_ratio = wind_var_pred / (wind_var_real + 1e-6)
    wind_var_floor = torch.relu(0.7 - var_ratio) * 100000.0
    
    # Temperature bias and variance matching
    bias_var_temp = (torch.mean(mu_x[:, 2]) - torch.mean(x[:, 2]))**2 * 2000000.0 + \
                    (torch.var(mu_x[:, 2]) - torch.var(x[:, 2]))**2 * 500000.0
    
    recon_loss = mse_solar + mse_wind + mse_temp + bias_var_temp + wind_spectral + wind_var_floor
    
    # KL divergence with safety clamping
    kl_loss = -0.5 * torch.mean(torch.sum(1 + logvar_z - mu_z.pow(2) - logvar_z.exp(), dim=1))
    
    # Temporal gradient loss (captures volatility patterns)
    real_diff = x[:, :, 1:] - x[:, :, :-1]
    pred_diff = mu_x[:, :, 1:] - mu_x[:, :, :-1]
    diff_loss = torch.mean(torch.abs(real_diff - pred_diff)) * 3000.0
    
    # Full spectral loss for diurnal patterns
    fft_real = torch.fft.rfft(x, dim=2) / (x.size(2)**0.5)
    fft_pred = torch.fft.rfft(mu_x, dim=2) / (x.size(2)**0.5)
    spectral_loss = torch.mean(torch.abs(torch.abs(fft_real) - torch.abs(fft_pred))) * 10000.0
    
    # Solar-Temperature correlation preservation
    def cross_corr(a, b):
        a_v = a - a.mean(dim=2, keepdim=True)
        b_v = b - b.mean(dim=2, keepdim=True)
        return torch.mean(a_v * b_v, dim=2) / (a.std(dim=2).clamp(1e-4) * b.std(dim=2).clamp(1e-4))

    corr_loss = torch.mean((cross_corr(x[:, 0:1], x[:, 2:3]) - cross_corr(mu_x[:, 0:1], mu_x[:, 2:3]))**2) * 100.0

    total_loss = recon_loss + beta * kl_loss + diff_loss + spectral_loss + corr_loss
    return total_loss, recon_loss, kl_loss

# =============================================================================
# TRAINING
# =============================================================================
def train_model(train_data, train_cond, val_data, val_cond, epochs=150, batch_size=4, 
                learning_rate=5e-5, latent_dim=128, max_beta=0.0001, save_path=None, 
                silent=False, patience=50):
    """
    Train the CVAE model with early stopping and physical metric monitoring.
    
    Args:
        train_data, train_cond: Training data and conditions
        val_data, val_cond: Validation data and conditions
        epochs: Maximum training epochs
        batch_size: Batch size for training
        learning_rate: Initial learning rate
        latent_dim: Latent space dimension
        max_beta: Maximum KL weight (annealed during training)
        save_path: Path to save best model weights
        silent: Suppress progress output
        patience: Early stopping patience
    
    Returns:
        Tuple of (trained model, final train loss, best validation loss)
    """
    if save_path is None:
        save_path = get_path(os.path.join("..", "models", "vae_weights.pth"))
    
    model = CVAE_1D(latent_dim=latent_dim).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=2e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6)
    
    train_loader = DataLoader(TensorDataset(train_data, train_cond), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(val_data, val_cond), batch_size=batch_size)
    
    best_val, no_improve, best_peak_ratio = float('inf'), 0, 0.0
    pbar = tqdm(range(epochs), desc="Training VAE", disable=silent)
    
    for epoch in pbar:
        model.train()
        beta = min(max_beta, max_beta * (epoch / (0.5 * epochs + 1e-9)))
        t_loss = 0
        
        for b_data, b_cond in train_loader:
            b_data, b_cond = b_data.to(DEVICE), b_cond.to(DEVICE)
            optimizer.zero_grad()
            # Add small noise for regularization
            mu_x, _, mu_z, logvar_z = model(b_data + torch.randn_like(b_data)*0.005, b_cond)
            loss, _, _ = vae_loss_1d(mu_x, b_data, mu_z, logvar_z, beta=beta)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            t_loss += loss.item()
            
        # Validation with physical metrics
        model.eval()
        v_loss, solar_mae, temp_mae, wind_mae = 0.0, 0.0, 0.0, 0.0
        peak_ratio, var_ratio = 0.0, 0.0
        
        with torch.no_grad():
            for b_data, b_cond in val_loader:
                b_data, b_cond = b_data.to(DEVICE), b_cond.to(DEVICE)
                mu_x, _, mu_z, logvar_z = model(b_data, b_cond)
                l, _, _ = vae_loss_1d(mu_x, b_data, mu_z, logvar_z, beta=beta)
                v_loss += l.item()
                
                # Physical metrics
                solar_mae += torch.mean(torch.abs(b_data[:, 0] - mu_x[:, 0])).item()
                wind_mae += torch.mean(torch.abs(b_data[:, 1] - mu_x[:, 1])).item()
                temp_mae += torch.mean(torch.abs(b_data[:, 2] - mu_x[:, 2])).item()
                
                # Peak and variance ratios for solar
                real_peak = b_data[:, 0].max().item()
                synth_peak = mu_x[:, 0].max().item()
                peak_ratio += synth_peak / (real_peak + 1e-9)
                var_ratio += mu_x[:, 0].var().item() / (b_data[:, 0].var().item() + 1e-9)
        
        n_batches = len(val_loader)
        solar_mae /= n_batches
        wind_mae /= n_batches
        temp_mae /= n_batches
        peak_ratio /= n_batches
        var_ratio /= n_batches
        
        avg_v = v_loss / n_batches
        avg_t = t_loss / len(train_loader)
        scheduler.step()
        gap = ((avg_v - avg_t) / (avg_t + 1e-9)) * 100
        
        # Early stopping checks
        if epoch > 15 and gap > 30.0:
            if not silent: print(f"\n[STOP] Overfitting detected: gap {gap:.1f}%")
            break
            
        if epoch > 10 and var_ratio < 0.3:
            if not silent: print(f"\n[STOP] Variance collapse: ratio {var_ratio:.2f}")
            break
            
        if epoch > 20 and peak_ratio < 0.5:
            if not silent: print(f"\n[STOP] Peak loss: ratio {peak_ratio:.2f}")
            break
        
        # Save best model
        if avg_v < best_val:
            best_val = avg_v
            best_peak_ratio = peak_ratio
            torch.save(model.state_dict(), save_path)
            no_improve = 0
        else: 
            no_improve += 1
            
        # Progress display
        if not silent: 
            pbar.set_postfix({
                "val": f"{avg_v:.0f}", 
                "gap": f"{gap:.1f}%",
                "peak": f"{peak_ratio:.2f}",
                "wMAE": f"{wind_mae:.3f}",
                "sMAE": f"{solar_mae:.3f}"
            })
            if (epoch + 1) % 10 == 0:
                print(f"\nEpoch {epoch+1}: Solar={solar_mae:.4f}, Wind={wind_mae:.4f}, "
                      f"Temp={temp_mae:.4f}, Peak={peak_ratio:.2f}, Gap={gap:.1f}%")
                
        if no_improve >= patience: 
            if not silent: print(f"\n[STOP] No improvement for {patience} epochs")
            break
            
    if os.path.exists(save_path):
        model.load_state_dict(torch.load(save_path, map_location=DEVICE))
    return model, avg_t, best_val

# =============================================================================
# HYPERPARAMETER OPTIMIZATION
# =============================================================================
def tune_hyperparameters(train_data, train_cond, val_data, val_cond, n_trials=10):
    """Use Optuna to find optimal hyperparameters."""
    def objective(trial):
        lr = trial.suggest_float("learning_rate", 1e-5, 1e-4, log=True)
        max_beta = trial.suggest_float("max_beta", 1e-5, 1e-3, log=True)
        _, _, v = train_model(
            train_data, train_cond, val_data, val_cond,
            epochs=30, learning_rate=lr, max_beta=max_beta, silent=True
        )
        return v
    
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
    print(f"Best hyperparameters: {study.best_params}")
    return study.best_params

# =============================================================================
# SCENARIO GENERATION
# =============================================================================
def generate_scenarios(num_samples=10, model_path=None, norm_path=None, latent_dim=128, cond_tensor=None):
    """
    Generate synthetic climate scenarios using the trained model.
    
    Args:
        num_samples: Number of scenarios to generate
        model_path: Path to saved model weights
        norm_path: Path to normalization parameters
        latent_dim: Latent dimension (must match trained model)
        cond_tensor: Conditioning tensor for generation
    
    Returns:
        Array of shape (num_samples, 8760, 3) with denormalized climate data
    """
    if model_path is None:
        model_path = get_path(os.path.join("..", "models", "vae_weights.pth"))
    if norm_path is None:
        norm_path = get_path(os.path.join("..", "models", "norm_params.npz"))
    
    model = CVAE_1D(latent_dim=latent_dim).to(DEVICE)
    if not os.path.exists(model_path):
        print(f"Warning: Model not found at {model_path}")
        return np.zeros((num_samples, 8760, 3))
    
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    
    cond_batch = cond_tensor[:num_samples].to(DEVICE)
    if cond_batch.size(0) < num_samples:
        cond_batch = cond_batch.repeat((num_samples // cond_batch.size(0)) + 1, 1, 1)[:num_samples]
    
    gen = model.sample(cond_batch, temperature=0.5).cpu().numpy()
    
    # Denormalize
    if os.path.exists(norm_path):
        m = np.load(norm_path)
        d_min, d_max = m["data_min"].reshape(1,3,1), m["data_max"].reshape(1,3,1)
        gen = gen * (d_max - d_min) + d_min
        gen[:, 1] = np.expm1(gen[:, 1])  # Inverse log-transform wind
        gen[:, 0] = np.where(gen[:, 0] < 0.05, 0.0, gen[:, 0])  # Zero out night solar
    
    return np.transpose(gen, (0, 2, 1))

# =============================================================================
# VALIDATION & VISUALIZATION
# =============================================================================
def plot_comparison(real_year, synth_year, output_path):
    """Generate comparison plots between real and synthetic data."""
    fig, axes = plt.subplots(3, 2, figsize=(24, 15), sharex='col')
    labels = ["Solar PV", "Wind", "Temperature"]
    colors = ["#f39c12", "#3498db", "#e74c3c"]
    
    for i in range(3):
        # Weekly trend (left column)
        r_smooth = pd.Series(real_year[:, i]).rolling(window=168, center=True).mean()
        s_smooth = pd.Series(synth_year[:, i]).rolling(window=168, center=True).mean()
        axes[i, 0].plot(real_year[:, i], color=colors[i], alpha=0.1)
        axes[i, 0].plot(r_smooth, color=colors[i], lw=2.5, label="Real")
        axes[i, 0].plot(s_smooth, color="black", lw=2, ls="--", label="Synthetic")
        axes[i, 0].set_title(f"{labels[i]} - Annual Trend")
        axes[i, 0].legend()
        
        # Week zoom (right column)
        axes[i, 1].plot(real_year[:168, i], color=colors[i], lw=1.5, label="Real")
        axes[i, 1].plot(synth_year[:168, i], color="black", lw=1.5, label="Synthetic")
        axes[i, 1].set_title(f"{labels[i]} - First Week Detail")
        axes[i, 1].legend()
        
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Comparison plot saved to: {output_path}")


def validate_scientific_metrics(model, test_data, test_cond, train_data=None):
    """Compute and display scientific validation metrics."""
    model.eval()
    with torch.no_grad():
        test_data, test_cond = test_data.to(DEVICE), test_cond.to(DEVICE)
        gen = model.sample(test_cond, temperature=0.5)
        
    # Wind autocorrelation (lag-1)
    def lag1_autocorr(x):
        return torch.mean((x[:, 1:] - x[:, 1:].mean()) * (x[:, :-1] - x[:, :-1].mean())) / (x.var() + 1e-7)
    
    ac_real = lag1_autocorr(test_data[:, 1])
    ac_synth = lag1_autocorr(gen[:, 1])
    
    # Solar-Temperature correlation
    def pearson(x, y):
        vx, vy = x - x.mean(), y - y.mean()
        return torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx**2)) * torch.sqrt(torch.sum(vy**2)) + 1e-7)
    
    corr_real = pearson(test_data[:, 0].flatten(), test_data[:, 2].flatten()).item()
    corr_synth = pearson(gen[:, 0].flatten(), gen[:, 2].flatten()).item()

    # Wasserstein distances
    wd_solar = wasserstein_distance(
        test_data[:, 0].flatten().cpu().numpy(),
        gen[:, 0].flatten().cpu().numpy()
    )
    wd_wind = wasserstein_distance(
        test_data[:, 1].flatten().cpu().numpy(),
        gen[:, 1].flatten().cpu().numpy()
    )
    
    # Nearest neighbor novelty check
    synth_sample = gen[0].flatten()
    min_dist_test = float('inf')
    for i in range(min(len(test_data), 100)):
        dist = torch.norm(synth_sample - test_data[i].flatten()).item()
        min_dist_test = min(min_dist_test, dist)
            
    min_dist_train = float('inf')
    if train_data is not None:
        n_samples = min(len(train_data), 200)
        indices = torch.randperm(len(train_data))[:n_samples]
        for i in indices:
            dist = torch.norm(synth_sample - train_data[i].to(DEVICE).flatten()).item()
            min_dist_train = min(min_dist_train, dist)

    print("\n" + "="*60)
    print("SCIENTIFIC VALIDATION REPORT")
    print("="*60)
    print(f"Wind Autocorrelation (lag-1):  Real={ac_real:.3f}, Synth={ac_synth:.3f}")
    print(f"Solar-Temp Correlation:        Real={corr_real:.3f}, Synth={corr_synth:.3f}")
    print(f"Wasserstein Distance:          Solar={wd_solar:.4f}, Wind={wd_wind:.4f}")
    print(f"Novelty Check (NN Distance):   Test={min_dist_test:.2f}, Train={min_dist_train:.2f}")
    print("="*60 + "\n")


def validate_scientific_metrics_full(model, test_data, test_cond, data_min, data_max):
    """Compute MAE in real physical units."""
    model.eval()
    with torch.no_grad():
        test_data, test_cond = test_data.to(DEVICE), test_cond.to(DEVICE)
        gen = model.sample(test_cond, temperature=0.0)
        
        d_min = torch.tensor(data_min).to(DEVICE)
        d_max = torch.tensor(data_max).to(DEVICE)
        d_range = (d_max - d_min).clamp(min=1e-9)
        
        real_units = test_data * d_range + d_min
        pred_units = gen * d_range + d_min
        
        # Inverse log-transform wind
        real_wind = torch.expm1(real_units[:, 1])
        pred_wind = torch.expm1(pred_units[:, 1])
        
        mae_solar = torch.mean(torch.abs(real_units[:, 0] - pred_units[:, 0])).item()
        mae_wind = torch.mean(torch.abs(real_wind - pred_wind)).item()
        mae_temp = torch.mean(torch.abs(real_units[:, 2] - pred_units[:, 2])).item()

    print("\n" + "="*50)
    print("ACCURACY REPORT (Physical Units)")
    print(f"Solar MAE: {mae_solar:.4f}")
    print(f"Wind MAE:  {mae_wind:.4f}")
    print(f"Temp MAE:  {mae_temp:.4f} °C")
    print("="*50 + "\n")

# =============================================================================
# MAIN ENTRY POINT
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Train VAE for climate scenario generation")
    parser.add_argument("--tune", action="store_true", help="Run hyperparameter optimization")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs")
    args, _ = parser.parse_known_args()

    # Load data
    train_data, train_cond, val_data, val_cond, test_data, test_cond, d_min, d_max = load_data()
    np.savez(get_path("../models/norm_params.npz"), data_min=d_min, data_max=d_max)
    print(f"Data loaded: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test samples")

    # Hyperparameters
    params = {
        "learning_rate": 3e-5,
        "latent_dim": 128,
        "max_beta": 0.0001,
        "batch_size": 4
    }
    
    if args.tune:
        print("\nStarting hyperparameter optimization...")
        params.update(tune_hyperparameters(train_data, train_cond, val_data, val_cond))
            
    # Train model
    print("\nTraining model...")
    model, _, _ = train_model(train_data, train_cond, val_data, val_cond, epochs=args.epochs, **params)
    
    # Validation
    print("\n--- FINAL VALIDATION ---")
    validate_scientific_metrics_full(model, test_data, test_cond, d_min, d_max)
    validate_scientific_metrics(model, test_data, test_cond, train_data=train_data)
    
    # Generate scenarios
    scenarios = generate_scenarios(num_samples=100, cond_tensor=test_cond, latent_dim=128)
    output_path = get_path("../data/SC/synthetic_scenarios.npy")
    np.save(output_path, scenarios)
    print(f"Generated {len(scenarios)} scenarios, saved to: {output_path}")
    
    # Visualization
    r_units = (test_data[0].cpu().numpy() * (d_max[0] - d_min[0]) + d_min[0])
    r_units[1] = np.expm1(r_units[1])
    plot_comparison(r_units.T, scenarios[0], get_path("../results/vae_comparison.png"))
    
    print("\n[OK] Climate scenario generation complete!")

if __name__ == "__main__":
    main()
