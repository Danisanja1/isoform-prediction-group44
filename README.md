Deep Learning — DTU 02456 · Group 44

This project explores how different representation learning techniques affect the prediction of isoform-level expression from gene-level expression. 
The task is motivated by the biological fact that a single gene can produce multiple isoforms via alternative splicing, making isoform expression more informative but harder to measure experimentally.

We evaluate three input representations:
- Raw gene expression values (full dimensionality)
- PCA-reduced gene expression (800 principal components)
- VAE latent representation (512-dimensional bottleneck)

A feed-forward MLP is trained on each representation to predict isoform expression. Models are compared using Validation Pearson correlation and MSE.
We also implement **residual MLP variants**, where skip connections are added between hidden layers. These residual networks are trained on the same representations (RAW, PCA, VAE) to evaluate whether residual connections improve optimization and prediction performance compared to standard MLPs.



Repository Structure:

train_raw.py               # Standard MLP using raw gene expression

train_pca_800.py           # Standard MLP on PCA (800 components)

train_baseline.py          # Original baseline (used to generate cached X/Y)

train_VAE.py               # Standard MLP on VAE latent space

residual/train_baseline_residual.py   # Baseline but with residual blocks
residual/train_raw_residual.py        # Residual MLP on raw expression
residual/train_pca_800_residual.py    # Residual MLP on PCA features
residual/train_VAE_residual.py        # Residual MLP on VAE latent features

inspect_run.py              # Inspect model performance across runs
plot_all_curves.py          # Compare Raw, PCA, VAE (and residual versions)
visualizecurves.py          # Utility plotting script



How to Run the Code:
(All scripts assume access to the DTU HPC cluster and the preprocessed data provided for the course)

1. Run the baseline once to create cached matrices
   
python train_baseline.py

3. Train individual models
   
python train_raw.py

python train_pca_800.py

python train_VAE.py

2b. Train residual MLP models

python residual/train_raw_residual.py

python residual/train_pca_800_residual.py

python residual/train_VAE_residual.py


4. Inspect training results
python inspect_run.py

5. Generate comparison plots (Pearson, MSE)
python plot_all_curves.py

All models are saved under:
runs_raw/
runs_pca/
runs_vae_repr/

Each run folder contains:
metrics.csv
best_model.pt
config.json


Results (Summary):
Representation	      Pearson ↑  	MSE ↓

Raw genes	            0.882	      1.249

PCA (800 components)	0.900	      1.015

VAE (512 latent dims)	0.846	      1.507

Residual MLP models were also trained on all three representations. Their performance is included and discussed in the project report, allowing a direct comparison between standard and residual architectures.


PCA offers the strongest performance, with the best balance between dimensionality reduction and information retention.

Authors:
Group 44 — DTU 02456 Deep Learning
- Daniel Sánchez Javaloyes
- Fanny Gómez Martín
- Guillermo Moya Fernández
