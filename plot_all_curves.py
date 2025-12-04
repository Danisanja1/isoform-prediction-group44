#!/usr/bin/env python3
"""
plot_all_curves.py

Plots validation Pearson and validation MSE curves for
RAW, PCA and VAE models on the same graphs.

It reads the metrics.csv files from:
- runs_raw/...
- runs_pca/...
- runs_vae_repr/...
Adjust the paths below if your run folder names differ.
"""

import matplotlib
matplotlib.use("Agg")  # use non-interactive backend (important on HPC)

import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

# === EDIT THESE PATHS IF NEEDED ===
RAW_DIR = Path("runs_raw") / "run_20251204_165511_seed42_raw"
PCA_DIR = Path("runs_pca") / "run_20251204_180009_seed42_pca800"
VAE_DIR = Path("runs_vae_repr") / "run_20251127_205646_seed42_vae512"

RAW_CSV = RAW_DIR / "metrics.csv"
PCA_CSV = PCA_DIR / "metrics.csv"
VAE_CSV = VAE_DIR / "metrics.csv"


def load_metrics(path: Path):
    df = pd.read_csv(path)
    # Ensure numeric in case of strings
    df["epoch"] = pd.to_numeric(df["epoch"])
    df["val_pearson"] = pd.to_numeric(df["val_pearson"])
    df["val_mse"] = pd.to_numeric(df["val_mse"])
    return df


def main():
    print("[INFO] Loading metrics...")
    df_raw = load_metrics(RAW_CSV)
    df_pca = load_metrics(PCA_CSV)
    df_vae = load_metrics(VAE_CSV)

    # -------- Pearson figure --------
    plt.figure(figsize=(10, 5))
    plt.plot(df_raw["epoch"], df_raw["val_pearson"], label="RAW", marker="o")
    plt.plot(df_pca["epoch"], df_pca["val_pearson"], label="PCA (800)", marker="o")
    plt.plot(df_vae["epoch"], df_vae["val_pearson"], label="VAE (512)", marker="o")

    plt.xlabel("Epoch")
    plt.ylabel("Validation Pearson")
    plt.title("Validation Pearson – RAW vs PCA vs VAE")
    plt.grid(True)
    plt.legend(loc="best")
    plt.tight_layout()

    pearson_out = "combined_learning_curves_pearson.png"
    plt.savefig(pearson_out)
    print(f"[INFO] Saved Pearson figure to {pearson_out}")

    # -------- MSE figure --------
    plt.figure(figsize=(10, 5))
    plt.plot(df_raw["epoch"], df_raw["val_mse"], label="RAW", marker="o")
    plt.plot(df_pca["epoch"], df_pca["val_mse"], label="PCA (800)", marker="o")
    plt.plot(df_vae["epoch"], df_vae["val_mse"], label="VAE (512)", marker="o")

    plt.xlabel("Epoch")
    plt.ylabel("Validation MSE")
    plt.title("Validation MSE – RAW vs PCA vs VAE")
    plt.grid(True)
    plt.legend(loc="best")
    plt.tight_layout()

    mse_out = "combined_learning_curves_mse.png"
    plt.savefig(mse_out)
    print(f"[INFO] Saved MSE figure to {mse_out}")


if __name__ == "__main__":
    main()
