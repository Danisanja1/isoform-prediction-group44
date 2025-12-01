#!/usr/bin/env python3
"""
inspect_run.py
Compares Raw vs PCA models in ./runs_raw and ./runs_pca.
Saves learning curve plots as PNG files for the best Raw and PCA models only.
"""

import os
import json
import glob
import pandas as pd
import matplotlib.pyplot as plt

RUNS_RAW_DIR = "runs_raw"
RUNS_PCA_DIR = "runs_pca"
RUNS_VAE_DIR = "runs_vae_repr"

def load_run(run_path):
    """Load config and metrics, return dict summary."""
    cfg_path = os.path.join(run_path, "config.json")
    csv_path = os.path.join(run_path, "metrics.csv")
    if not (os.path.exists(cfg_path) and os.path.exists(csv_path)):
        return None

    # Load config
    with open(cfg_path) as f:
        cfg = json.load(f)

    # Load metrics
    df = pd.read_csv(csv_path)
    best_row = df.loc[df["val_pearson"].idxmax()]
    summary = {
        "run": os.path.basename(run_path),
        "seed": cfg.get("SEED"),
        "val_pearson": best_row["val_pearson"],
        "val_mse": best_row["val_mse"],
        "epoch": int(best_row["epoch"]),
        "N_SAMPLES": cfg.get("N_SAMPLES"),
        "TOP_GENES": cfg.get("TOP_GENES"),
        "MAX_ISOFORMS": cfg.get("MAX_ISOFORMS"),
        "EPOCHS": cfg.get("EPOCHS"),
        "BATCH_SIZE": cfg.get("BATCH_SIZE"),
        "LR": cfg.get("LR"),
        "DROPOUT_P": cfg.get("DROPOUT_P"),
        "REPRESENTATION": cfg.get("REPRESENTATION")
    }
    return summary, df


def main():
    # Load and compare raw and PCA runs
    raw_runs = sorted(glob.glob(os.path.join(RUNS_RAW_DIR, "run_*_seed*")))
    pca_runs = sorted(glob.glob(os.path.join(RUNS_PCA_DIR, "run_*_seed*")))
    vae_runs = sorted(glob.glob(os.path.join(RUNS_VAE_DIR, "run_*_seed*")))

    all_runs = []
    for r in raw_runs + pca_runs + vae_runs:
        result = load_run(r)
        if result is None:
            continue
        summary, df = result
        all_runs.append(summary)

    df_summary = pd.DataFrame(all_runs)
    df_summary = df_summary.sort_values(by="val_pearson", ascending=False)

    print("\n=== Run Summary (sorted by best val Pearson) ===\n")
    print(df_summary.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    # Compare PCA vs Raw performance
    raw_best = df_summary[df_summary["REPRESENTATION"] == "RAW"].iloc[0]
    pca_best = df_summary[df_summary["REPRESENTATION"] == "PCA"].iloc[0]
    vae_best = df_summary[df_summary["REPRESENTATION"] == "VAE"].iloc[0]

    print("\n=== PCA vs Raw Comparison ===")
    print(f"Best Raw Model: {raw_best['run']}, Pearson: {raw_best['val_pearson']}, MSE: {raw_best['val_mse']}")
    print(f"Best PCA Model: {pca_best['run']}, Pearson: {pca_best['val_pearson']}, MSE: {pca_best['val_mse']}")
    print(f"Best VAE Model: {vae_best['run']}, Pearson: {vae_best['val_pearson']}, MSE: {vae_best['val_mse']}")

    if raw_best["val_pearson"] > pca_best["val_pearson"]:
        print("\nThe raw gene model performed better.")
    else:
        print("\nThe PCA model performed better.")

    # Save learning curve for the best Raw model
    best_run = raw_best['run']
    run_dir = f"runs_raw/{best_run}"
    df = pd.read_csv(os.path.join(run_dir, "metrics.csv"))
    
    
    # Plot validation Pearson
    plt.figure(figsize=(10, 5))
    plt.plot(df["epoch"], df["val_pearson"], label="Validation Pearson", color="C0")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Pearson")
    plt.title(f"Learning Curve - Best Raw Model {best_run}")
    plt.legend(loc="best")
    plt.grid(True)
    
    plot_path_pearson = f"{run_dir}/learning_curve_{best_run}_pearson.png"
    plt.tight_layout()
    plt.savefig(plot_path_pearson)
    print(f"[INFO] Saved Pearson learning curve to {plot_path_pearson}")

    # Plot validation MSE
    plt.figure(figsize=(10, 5))
    plt.plot(df["epoch"], df["val_mse"], label="Validation MSE", color="C1")
    plt.xlabel("Epoch")
    plt.ylabel("Validation MSE")
    plt.title(f"Validation MSE - Best Raw Model {best_run}")
    plt.legend(loc="best")
    plt.grid(True)
    
    plot_path_mse = f"{run_dir}/learning_curve_{best_run}_mse.png"
    plt.tight_layout()
    plt.savefig(plot_path_mse)
    print(f"[INFO] Saved MSE learning curve to {plot_path_mse}")

    # Save learning curve for the best PCA model
    best_run = pca_best['run']
    run_dir = f"runs_pca/{best_run}"
    df = pd.read_csv(os.path.join(run_dir, "metrics.csv"))
    
    # Plot validation Pearson
    plt.figure(figsize=(10, 5))
    plt.plot(df["epoch"], df["val_pearson"], label="Validation Pearson", color="C0")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Pearson")
    plt.title(f"Learning Curve - Best PCA Model {best_run}")
    plt.legend(loc="best")
    plt.grid(True)
    
    plot_path_pearson = f"{run_dir}/learning_curve_{best_run}_pearson.png"
    plt.tight_layout()
    plt.savefig(plot_path_pearson)
    print(f"[INFO] Saved Pearson learning curve to {plot_path_pearson}")

    # Plot validation MSE
    plt.figure(figsize=(10, 5))
    plt.plot(df["epoch"], df["val_mse"], label="Validation MSE", color="C1")
    plt.xlabel("Epoch")
    plt.ylabel("Validation MSE")
    plt.title(f"Validation MSE - Best PCA Model {best_run}")
    plt.legend(loc="best")
    plt.grid(True)
    
    plot_path_mse = f"{run_dir}/learning_curve_{best_run}_mse.png"
    plt.tight_layout()
    plt.savefig(plot_path_mse)
    print(f"[INFO] Saved MSE learning curve to {plot_path_mse}")

    # Save learning curve for the best VAE model
    best_run = vae_best['run']
    run_dir = f"runs_vae_repr/{best_run}"
    df = pd.read_csv(os.path.join(run_dir, "metrics.csv"))
    
    # Plot validation Pearson
    plt.figure(figsize=(10, 5))
    plt.plot(df["epoch"], df["val_pearson"], label="Validation Pearson", color="C0")
    plt.xlabel("Epoch")
    plt.ylabel("Validation Pearson")
    plt.title(f"Learning Curve - Best VAE Model {best_run}")
    plt.legend(loc="best")
    plt.grid(True)
    
    plot_path_pearson = f"{run_dir}/learning_curve_{best_run}_pearson.png"
    plt.tight_layout()
    plt.savefig(plot_path_pearson)
    print(f"[INFO] Saved Pearson learning curve to {plot_path_pearson}")

    # Plot validation MSE
    plt.figure(figsize=(10, 5))
    plt.plot(df["epoch"], df["val_mse"], label="Validation MSE", color="C1")
    plt.xlabel("Epoch")
    plt.ylabel("Validation MSE")
    plt.title(f"Validation MSE - Best VAE Model {best_run}")
    plt.legend(loc="best")
    plt.grid(True)
    
    plot_path_mse = f"{run_dir}/learning_curve_{best_run}_mse.png"
    plt.tight_layout()
    plt.savefig(plot_path_mse)
    print(f"[INFO] Saved MSE learning curve to {plot_path_mse}")

    # Show the plots (optional, may not work on headless systems)
    plt.show()


if __name__ == "__main__":
    main()
