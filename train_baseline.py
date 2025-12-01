#!/usr/bin/env python3
# Predict isoform expression from gene expression (bulk dataset)
# DTU HPC-friendly: uses GPU if available, caches preprocessed arrays, logs to CSV.

import os
import math
import time
import json
import csv
import datetime
import pathlib
import numpy as np
import anndata as ad
from scipy import sparse

import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader

# =========================
# Config (edit as needed)
# =========================
GENE_PATH = "/work3/s193518/scIsoPred/data/bulk_processed_genes.h5ad"
ISOFORM_PATH = "/work3/s193518/scIsoPred/data/bulk_processed_transcripts.h5ad"

# You can override SEED at runtime:  SEED=123 python train_baseline.py
SEED = int(os.getenv("SEED", "42"))

# Scale gradually once it works:
N_SAMPLES    = 5000       # number of samples (rows) to use
TOP_GENES    = 10000      # keep top-K most-variable genes as inputs
MAX_ISOFORMS = 10000      # cap number of isoform targets (columns)
VAL_FRACTION = 0.1

# Training
HIDDEN_DIM   = 1024
DROPOUT_P    = 0.10
EPOCHS       = 10
BATCH_SIZE   = 128
LR           = 1e-3
WEIGHT_DECAY = 1e-5
ZSCORE_INPUTS = True      # z-score inputs per gene after log1p
PRINT_EVERY  = 1

# Directories
CACHE_DIR = "cache"
RUNS_DIR  = "runs"

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
# Utilities
# =========================
def densify_slice(arr, rows=None, cols=None):
    """Slice first, then convert to dense numpy array to save memory."""
    view = arr
    if rows is not None:
        view = view[:rows]
    if cols is not None:
        view = view[:, :cols]
    if sparse.issparse(view):
        return view.toarray()
    return np.asarray(view)

def torch_pearsonr(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Pearson correlation across all elements (flattened)."""
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
    os.makedirs(CACHE_DIR, exist_ok=True)
    x_path = os.path.join(CACHE_DIR, f"X_n{n_samples}_g{top_genes}.npy")
    y_path = os.path.join(CACHE_DIR, f"Y_n{n_samples}_t{max_iso}.npy")
    idx_path = os.path.join(CACHE_DIR, f"Xidx_g{top_genes}.npy")  # indices of selected genes
    mu_path  = os.path.join(CACHE_DIR, f"mu_n{n_samples}_g{top_genes}.npy")
    sd_path  = os.path.join(CACHE_DIR, f"sd_n{n_samples}_g{top_genes}.npy")
    return x_path, y_path, idx_path, mu_path, sd_path

# =========================
# Data loading / Caching
# =========================
def load_or_build_data():
    t0 = time.time()
    x_cache, y_cache, idx_cache, mu_cache, sd_cache = cache_paths(N_SAMPLES, TOP_GENES, MAX_ISOFORMS)

    # If cached arrays exist, load them (fast path)
    if os.path.exists(x_cache) and os.path.exists(y_cache):
        print("[INFO] Loading cached arrays...")
        X = np.load(x_cache, mmap_mode="r")
        Y = np.load(y_cache, mmap_mode="r")
        print(f"[INFO] Cached X: {X.shape}, Y: {Y.shape}")
        print(f"[INFO] Setup time: {time.time() - t0:.1f}s\n")
        return X, Y, None, None  # we'll recompute z-score stats on train split

    # Cache miss → build once
    print("[INFO] Loading AnnData (first run, building cache)...")
    genes = ad.read_h5ad(GENE_PATH)
    iso   = ad.read_h5ad(ISOFORM_PATH)
    print(f"[INFO] genes:    {genes.X.shape}")
    print(f"[INFO] isoforms: {iso.X.shape}")

    print("[INFO] Slicing, densifying, log1p...")
    X_full = densify_slice(genes.X, rows=N_SAMPLES, cols=None)           # [N, G]
    Y      = densify_slice(iso.X,   rows=N_SAMPLES, cols=MAX_ISOFORMS)   # [N, T]
    X_full = np.log1p(X_full).astype(np.float32)
    Y      = np.log1p(Y).astype(np.float32)

    # Select top-K variable genes (keeps columns with highest variance)
    if TOP_GENES is not None and TOP_GENES < X_full.shape[1]:
        vars_ = X_full.var(axis=0)
        top_idx = np.argsort(vars_)[-TOP_GENES:]
        top_idx.sort()  # keep column order
        np.save(idx_cache, top_idx)
        X = X_full[:, top_idx]
    else:
        X = X_full

    # Save cache for fast future runs
    print("[INFO] Saving cache...")
    np.save(x_cache, X)
    np.save(y_cache, Y)

    print(f"[INFO] Using N_SAMPLES={X.shape[0]}, IN={X.shape[1]}, OUT={Y.shape[1]}")
    print(f"[INFO] Setup time: {time.time() - t0:.1f}s\n")
    return X, Y, None, None  # we'll compute z-score stats after splitting

# =========================
# Model
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

# =========================
# Train / Eval
# =========================
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
# Main
# =========================
def main():
    # Per-run folders & metadata
    RUN_ID = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    RUN_DIR = pathlib.Path(RUNS_DIR) / f"run_{RUN_ID}_seed{SEED}"
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
        "ZSCORE_INPUTS": ZSCORE_INPUTS,
        "DEVICE": str(DEVICE),
    }
    with open(CONFIG_PATH, "w") as f:
        json.dump(config_dict, f, indent=2)

    # Load data (or from cache)
    X, Y, _, _ = load_or_build_data()

    # Train/val split
    n = X.shape[0]
    n_val = max(1, int(math.ceil(VAL_FRACTION * n)))
    n_train = n - n_val
    idx = np.arange(n)
    np.random.shuffle(idx)
    train_idx, val_idx = idx[:n_train], idx[n_train:]

    X_train = X[train_idx]
    Y_train = Y[train_idx]
    X_val   = X[val_idx]
    Y_val   = Y[val_idx]

    # Optional z-scoring (fit on train, apply to val)
    if ZSCORE_INPUTS:
        mu = X_train.mean(axis=0, keepdims=True)
        sd = X_train.std(axis=0, keepdims=True) + 1e-6
        X_train = (X_train - mu) / sd
        X_val   = (X_val   - mu) / sd

    # Tensors & loaders
    X_train = torch.tensor(X_train, dtype=torch.float32)
    Y_train = torch.tensor(Y_train, dtype=torch.float32)
    X_val   = torch.tensor(X_val,   dtype=torch.float32)
    Y_val   = torch.tensor(Y_val,   dtype=torch.float32)

    train_loader = DataLoader(TensorDataset(X_train, Y_train),
                              batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
    val_loader   = DataLoader(TensorDataset(X_val,   Y_val),
                              batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    # Model
    in_dim  = X_train.shape[1]
    out_dim = Y_train.shape[1]
    model = IsoformMLP(in_dim, out_dim, hidden=HIDDEN_DIM, p_drop=DROPOUT_P).to(DEVICE)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[INFO] Model params: {n_params/1e6:.2f}M  (IN={in_dim}, OUT={out_dim})")

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    loss_fn = nn.MSELoss()
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3, verbose=True
    )

    # Init CSV header
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

        scheduler.step(val_mse)  # adjust LR based on val loss

        if epoch % PRINT_EVERY == 0:
            print(f"[E{epoch:02d}] train_MSE={train_mse:.6f}  val_MSE={val_mse:.6f}  "
                  f"val_Pearson={val_r:.4f}  ({dt:.1f}s)")

        # Append to CSV
        with open(CSV_PATH, "a", newline="") as f:
            w = csv.writer(f)
            w.writerow([epoch, f"{train_mse:.6f}", f"{val_mse:.6f}", f"{val_r:.6f}", f"{dt:.2f}"])

        # Save best model by validation Pearson
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
