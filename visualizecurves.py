import pandas as pd
import matplotlib.pyplot as plt

# Load the PCA model's metrics CSV
df = pd.read_csv("runs_pca/run_20251124_153151_seed123_pca1000/metrics.csv")

# Plot validation Pearson
plt.figure(figsize=(10, 5))
plt.plot(df["epoch"], df["val_pearson"], label="Validation Pearson", color="C0")
plt.xlabel("Epoch")
plt.ylabel("Validation Pearson")
plt.title("Learning Curve - PCA Model")
plt.legend(loc="best")
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot validation MSE
plt.figure(figsize=(10, 5))
plt.plot(df["epoch"], df["val_mse"], label="Validation MSE", color="C1")
plt.xlabel("Epoch")
plt.ylabel("Validation MSE")
plt.title("Validation MSE - PCA Model")
plt.legend(loc="best")
plt.grid(True)
plt.tight_layout()
plt.show()
