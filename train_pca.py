#!/usr/bin/env python3
# train_pca.py
# MLP on PCA representation of gene expression -> predict isoform expression

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
from sklearn.decomposition import PCA

# =========================
# Config
# =========================
# Must match the cache built by train_baseline.py
N_SAMPLES    = 5000        # same as baseline for the three
TOP_GENES    = 10000       
MAX_ISOFORMS = 10000      

# PCA config
N_COMPONENTS = 1000        # number of principal components (input dim to MLP)

VAL_FRACTION = 0.1

# Training
SEED = int(os.getenv("SEED", "42"))
HIDDEN_DIM   = 1024
DROPOUT_P    = 0.10
EPOCHS       = 10
BATCH_SIZE   = 128
LR           = 1e-3
WEIGHT_DECAY = 1e-5
PRINT_EVERY  = 1

CACHE_DIR = "cache"
RUNS_DIR  = "runs_pca"     # separate folder for PCA runs

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
# helpers
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
    pca_path = os.path.join(CACHE_DIR, f"PCA_n{n_samples}_g{top_genes}_pc{N_COMPONENTS}.npz")
    return x_path, y_path, pca_path

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
# main
# =========================
def main():
    x_cache, y_cache, pca_cache = cache_paths(N_SAMPLES, TOP_GENES, MAX_ISOFORMS)

    if not (os.path.exists(x_cache) and os.path.exists(y_cache)):
        raise RuntimeError(
            f"Cached X/Y not found. Run train_baseline.py once first to build {x_cache} and {y_cache}."
        )

    # ----- Load X, Y from baseline cache ----
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

    # ----- PCA: fit on train, transform both -----
    t0 = time.time()
    if os.path.exists(pca_cache):
        print("[INFO] Loading PCA projection from cache...")
        data_pca = np.load(pca_cache)
        X_train_pca = data_pca["X_train_pca"]
        X_val_pca   = data_pca["X_val_pca"]
    else:
        print(f"[INFO] Fitting PCA with n_components={N_COMPONENTS} on train...")
        # Optional: center/scale inputs before PCA (assumes X already log1p)
        # Here we just center (PCA does centering internally)
        pca = PCA(n_components=N_COMPONENTS, svd_solver="randomized", random_state=SEED)
        X_train_pca = pca.fit_transform(X_train_full)
        X_val_pca   = pca.transform(X_val_full)
        print("[INFO] Saving PCA projections to cache...")
        np.savez_compressed(pca_cache, X_train_pca=X_train_pca, X_val_pca=X_val_pca)
    print(f"[INFO] PCA done in {time.time()-t0:.1f}s")
    print(f"[INFO] X_train_pca: {X_train_pca.shape}, X_val_pca: {X_val_pca.shape}")

    # ----- Tensors & loaders -----
    X_train_t = torch.tensor(X_train_pca, dtype=torch.float32)
    Y_train_t = torch.tensor(Y_train,      dtype=torch.float32)
    X_val_t   = torch.tensor(X_val_pca,    dtype=torch.float32)
    Y_val_t   = torch.tensor(Y_val,        dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_train_t, Y_train_t),
                              batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader   = DataLoader(TensorDataset(X_val_t,   Y_val_t),
                              batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    in_dim  = X_train_pca.shape[1]   # N_COMPONENTS
    out_dim = Y_train.shape[1]       # same isoform dim as baseline

    model = IsoformMLP(in_dim, out_dim, hidden=HIDDEN_DIM, p_drop=DROPOUT_P).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Model params: {n_params/1e6:.2f}M  (IN={in_dim}, OUT={out_dim})")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=True
    )

    # ----- Per-run logging setup -----
    RUN_ID = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    RUN_DIR = pathlib.Path(RUNS_DIR) / f"run_{RUN_ID}_seed{SEED}_pca{N_COMPONENTS}"
    RUN_DIR.mkdir(parents=True, exist_ok=True)
    CSV_PATH = RUN_DIR / "metrics.csv"
    BEST_PATH = RUN_DIR / "best_model.pt"
    CONFIG_PATH = RUN_DIR / "config.json"

    config_dict = {
        "SEED": SEED,
        "N_SAMPLES": N_SAMPLES,
        "TOP_GENES": TOP_GENES,
        "MAX_ISOFORMS": MAX_ISOFORMS,
        "VAL_FRACTION": VAL_FRACTION,
        "HIDDEN_DIM": HIDDEN_DIM,
        "DROPOUT_P": DROPOUT_P,
        "EPOCHS": EPOCHS,
        "BATCH_SIZE": BATCH_SIZE,
        "LR": LR,
        "WEIGHT_DECAY": WEIGHT_DECAY,
        "DEVICE": str(DEVICE),
        "N_COMPONENTS": N_COMPONENTS,
        "REPRESENTATION": "PCA"
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
        dt = time.time() - t1

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
