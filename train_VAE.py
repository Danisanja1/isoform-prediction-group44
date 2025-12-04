#!/usr/bin/env python3
# train_VAE.py
# VAE for gene expression -> predict isoform expression

import os
import math
import time
import json
import csv
import datetime
import pathlib
import numpy as np

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

# =========================
# Config
# =========================
# Must match the cache built by train_baseline.py and train_pca.py
N_SAMPLES    = 5000        # same as baseline
TOP_GENES    = 10000       # same as baseline
MAX_ISOFORMS = 10000       # same as baseline

VAL_FRACTION = 0.1

# VAE config
LATENT_DIM   = 512         # Latent dimensionality for VAE (increased from 256)

# Training
SEED = int(os.getenv("SEED", "42"))
HIDDEN_DIM   = 1024
DROPOUT_P    = 0.10
EPOCHS       = 30          # increased epochs to 30
BATCH_SIZE   = 128
LR           = 1e-4        # lowered learning rate
WEIGHT_DECAY = 1e-5
PRINT_EVERY  = 1

MLP_HIDDEN = 1024  # Number of hidden units for the isoformmlp

CACHE_DIR    = "cache"
RUNS_DIR     = "runs_vae_repr"  # separate folder for vae-representation runs

# =========================
# Reproducibility & device
# =========================
def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_seed(SEED)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[INFO] Device: {DEVICE}")

# =========================
# Helpers
# =========================
def torch_pearsonr(pred: torch.Tensor, target: torch.Tensor) -> float:
    x = pred.view(-1).float()
    y = target.view(-1).float()
    vx = x - x.mean()
    vy = y - y.mean()
    denom = (vx.norm() * vy.norm())
    if denom.item() == 0:
        return 0.0
    r = (vx @ vy) / denom
    return float(r.detach().cpu().item())

def cache_paths(n_samples: int, top_genes: int, max_iso: int):
    x_path = os.path.join(CACHE_DIR, f"X_n{n_samples}_g{top_genes}.npy")
    y_path = os.path.join(CACHE_DIR, f"Y_n{n_samples}_t{max_iso}.npy")
    return x_path, y_path

# =========================
# Isoform MLP (same style as train_pca)
# =========================
class IsoformMLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 1024, p_drop: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Dropout(p_drop),
            nn.Linear(hidden, out_dim),
        )
    def forward(self, x):
        return self.net(x)

def run_epoch(model, loader, optimizer, loss_fn, train: bool):
    model.train(train)
    total = 0.0
    with torch.set_grad_enabled(train):
        for xb, yb in loader:
            xb = xb.to(DEVICE, non_blocking=True)
            yb = yb.to(DEVICE, non_blocking=True)
            pred = model(xb)
            loss = loss_fn(pred, yb)
            if train:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total += loss.item() * xb.size(0)
    return total / len(loader.dataset)

@torch.no_grad()
def eval_pearson(model, loader):
    model.eval()
    preds, reals = [], []
    for xb, yb in loader:
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)
        pred = model(xb)
        preds.append(pred.detach().cpu())
        reals.append(yb.detach().cpu())
    pred_all = torch.cat(preds, 0)
    real_all = torch.cat(reals, 0)
    return torch_pearsonr(pred_all, real_all)

# =========================
# VAE definition
# =========================
class VAE(nn.Module):
    def __init__(self, input_dim: int, latent_dim: int = 256, hidden_dim: int = 1024):
        super().__init__()
        self.fc_enc1 = nn.Linear(input_dim, hidden_dim)
        self.fc_enc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_mu   = nn.Linear(hidden_dim, latent_dim)
        self.fc_logv = nn.Linear(hidden_dim, latent_dim)

        self.fc_dec1 = nn.Linear(latent_dim, hidden_dim)
        self.fc_dec2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc_out  = nn.Linear(hidden_dim, input_dim)

    def encode(self, x):
        h = F.relu(self.fc_enc1(x))
        h = F.relu(self.fc_enc2(h))
        mu    = self.fc_mu(h)
        logv  = self.fc_logv(h)
        return mu, logv

    def reparameterize(self, mu, logv):
        std = torch.exp(0.5 * logv)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        h = F.relu(self.fc_dec1(z))
        h = F.relu(self.fc_dec2(h))
        x_recon = self.fc_out(h)
        return x_recon

    def forward(self, x):
        mu, logv = self.encode(x)
        z = self.reparameterize(mu, logv)
        x_recon = self.decode(z)
        return x_recon, mu, logv

def vae_loss(x_recon, x, mu, logv, kl_weight=1e-3):
    # Reconstruction mse over batch
    recon = F.mse_loss(x_recon, x, reduction="mean")
    # KL divergence per sample, then mean
    kl_per_sample = -0.5 * torch.sum(1 + logv - mu.pow(2) - logv.exp(), dim=1)
    kl = kl_per_sample.mean()
    return recon + kl_weight * kl, recon, kl

def train_vae(vae, loader, optimizer, epochs: int, kl_weight: float):
    vae.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        total_recon = 0.0
        total_kl = 0.0
        n_samples = 0
        t0 = time.time()
        for (xb,) in loader:
            xb = xb.to(DEVICE, non_blocking=True)
            optimizer.zero_grad()
            x_recon, mu, logv = vae(xb)
            loss, recon, kl = vae_loss(x_recon, xb, mu, logv, kl_weight=kl_weight)
            loss.backward()
            optimizer.step()

            bs = xb.size(0)
            n_samples += bs
            total_loss += loss.item() * bs
            total_recon += recon.item() * bs
            total_kl += kl.item() * bs

        dt = time.time() - t0
        print(f"[VAE E{epoch:02d}] loss={total_loss/n_samples:.4f} "
              f"recon={total_recon/n_samples:.4f} kl={total_kl/n_samples:.4f} ({dt:.1f}s)")

@torch.no_grad()
def encode_dataset(vae, X: torch.Tensor, batch_size: int = 256) -> np.ndarray:
    vae.eval()
    zs = []
    for i in range(0, X.size(0), batch_size):
        xb = X[i:i+batch_size].to(DEVICE)
        mu, logv = vae.encode(xb)
        zs.append(mu.detach().cpu())
    Z = torch.cat(zs, dim=0).numpy()
    return Z

# =========================
# Main
# =========================
def main():
    x_cache, y_cache = cache_paths(N_SAMPLES, TOP_GENES, MAX_ISOFORMS)

    if not (os.path.exists(x_cache) and os.path.exists(y_cache)):
        raise RuntimeError(
            f"Cached X/Y not found. Run train_baseline.py once first "
            f"to build {x_cache} and {y_cache}."
        )

    # ----- Load X, Y from baseline cache -----
    print("[INFO] Loading cached X/Y from baseline...")
    X = np.load(x_cache, mmap_mode="r")
    Y = np.load(y_cache, mmap_mode="r")
    print(f"[INFO] X shape: {X.shape}, Y shape: {Y.shape}")

    # ----- Train/val split (on samples) -----
    n = X.shape[0]
    n_val = max(1, int(math.ceil(VAL_FRACTION * n)))
    n_train = n - n_val
    idx = np.arange(n)
    np.random.shuffle(idx)
    train_idx, val_idx = idx[:n_train], idx[n_train:]

    X_train_full = X[train_idx]
    Y_train = Y[train_idx]
    X_val_full   = X[val_idx]
    Y_val        = Y[val_idx]

    # ----- Phase 1: train VAE on X -----
    X_train_t = torch.tensor(X_train_full, dtype=torch.float32)
    X_val_t   = torch.tensor(X_val_full,   dtype=torch.float32)

    vae = VAE(input_dim=X_train_full.shape[1],
              latent_dim=LATENT_DIM,
              hidden_dim=HIDDEN_DIM).to(DEVICE)

    vae_opt = torch.optim.Adam(vae.parameters(), lr=LR)
    vae_train_loader = DataLoader(
        TensorDataset(X_train_t),
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
    )

    print(f"[INFO] Training VAE representation (LATENT_DIM={LATENT_DIM})...")
    t0 = time.time()
    train_vae(vae, vae_train_loader, vae_opt, epochs=EPOCHS, kl_weight=1e-3)
    print(f"[INFO] VAE training done in {time.time()-t0:.1f}s")

    # ----- Encode X to latent space -----
    print("[INFO] Encoding train/val X to latent space...")
    Z_train = encode_dataset(vae, X_train_t, batch_size=256)
    Z_val   = encode_dataset(vae, X_val_t,   batch_size=256)
    print(f"[INFO] Z_train: {Z_train.shape}, Z_val: {Z_val.shape}")

    # ----- Phase 2: Isoform MLP on latent Z -> Y -----
    Z_train_t = torch.tensor(Z_train, dtype=torch.float32)
    Z_val_t   = torch.tensor(Z_val,   dtype=torch.float32)
    Y_train_t = torch.tensor(Y_train, dtype=torch.float32)
    Y_val_t   = torch.tensor(Y_val,   dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(Z_train_t, Y_train_t),
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        TensorDataset(Z_val_t, Y_val_t),
        batch_size=BATCH_SIZE,
        shuffle=False,
        drop_last=False,
    )

    in_dim  = Z_train.shape[1]     # LATENT_DIM
    out_dim = Y_train.shape[1]     # isoform dim

    model = IsoformMLP(in_dim, out_dim, hidden=MLP_HIDDEN, p_drop=DROPOUT_P).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] IsoformMLP on VAE repr params: {n_params/1e6:.2f}M  (IN={in_dim}, OUT={out_dim})")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=True
    )

    # ----- Logging setup -----
    RUN_ID = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    RUN_DIR = pathlib.Path(RUNS_DIR) / f"run_{RUN_ID}_seed{SEED}_vae{LATENT_DIM}"
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    CSV_PATH    = RUN_DIR / "metrics.csv"
    BEST_PATH   = RUN_DIR / "best_model.pt"
    CONFIG_PATH = RUN_DIR / "config.json"

    config_dict = {
        "SEED": SEED,
        "N_SAMPLES": N_SAMPLES,
        "TOP_GENES": TOP_GENES,
        "MAX_ISOFORMS": MAX_ISOFORMS,
        "VAL_FRACTION": VAL_FRACTION,
        "MLP_HIDDEN": MLP_HIDDEN,
        "DROPOUT_P": DROPOUT_P,
        "EPOCHS": EPOCHS,
        "BATCH_SIZE": BATCH_SIZE,
        "LR": LR,
        "WEIGHT_DECAY": WEIGHT_DECAY,
        "DEVICE": str(DEVICE),
        "LATENT_DIM": LATENT_DIM,
        "VAE_HIDDEN_DIM": HIDDEN_DIM,
        "VAE_EPOCHS": EPOCHS,
        "VAE_LR": LR,
        "KL_WEIGHT": 1e-3,
        "REPRESENTATION": "VAE",
    }
    with open(CONFIG_PATH, "w") as f:
        json.dump(config_dict, f, indent=2)

    with open(CSV_PATH, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["epoch", "train_mse", "val_mse", "val_pearson", "seconds"])

    best_val_pear = -1.0
    t0 = time.time()
    for epoch in range(1, EPOCHS + 1):
        t1 = time.time()
        train_mse = run_epoch(model, train_loader, optimizer, loss_fn, train=True)
        val_mse   = run_epoch(model, val_loader,   optimizer, loss_fn, train=False)
        val_r     = eval_pearson(model, val_loader)
        dt        = time.time() - t1

        scheduler.step(val_mse)

        if epoch % PRINT_EVERY == 0:
            print(f"[E{epoch:02d}] train_MSE={train_mse:.6f}  val_MSE={val_mse:.6f}  "
                  f"val_Pearson={val_r:.4f}  ({dt:.1f}s)")

        with open(CSV_PATH, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([epoch, f"{train_mse:.6f}", f"{val_mse:.6f}", f"{val_r:.6f}", f"{dt:.2f}"])

        if val_r > best_val_pear:
            best_val_pear = val_r
            torch.save({
                "model_state": model.state_dict(),
                "in_dim": in_dim,
                "out_dim": out_dim,
                "config": config_dict,
                "epoch": epoch,
                "val_pearson": float(val_r),
                "val_mse": float(val_mse),
            }, BEST_PATH)

    print(f"\n[INFO] Done in {time.time()-t0:.1f}s")
    print(f"[INFO] Logs (CSV):   {CSV_PATH}")
    print(f"[INFO] Best model:   {BEST_PATH}")
    print(f"[INFO] Run config:   {CONFIG_PATH}")

if __name__ == "__main__":
    main()
