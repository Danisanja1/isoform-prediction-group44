import torch
import numpy as np
from torch import nn
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Assuming the VAE model class and loss function are defined somewhere in your code.
# Ensure the VAE class is imported if it's defined in another file
from train_VAE import VAE  # Adjust the import if necessary

# Set device (make sure it's using GPU if available)
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the trained VAE model
vae = VAE(input_dim=10000, latent_dim=256).to(DEVICE)  # Adjust input_dim as needed

# Load the model weights (ensure you have saved the model after training)
vae.load_state_dict(torch.load("vae_model.pt"))
vae.eval()

# =========================
# Visualization of the latent space
# =========================
# Assuming you have X_train data
X_train = np.random.rand(5000, 10000)  # Example: replace with actual data
X_train = torch.tensor(X_train, dtype=torch.float32).to(DEVICE)

# Get the latent space using the encoder
mu, log_var = vae.encode(X_train)
latent_space = vae.reparameterize(mu, log_var).cpu().detach().numpy()

# Apply t-SNE to the latent space for 2D visualization
latent_2d = TSNE(n_components=2).fit_transform(latent_space)

# Use the mean isoform expression value for coloring (adjust based on your data)
mean_expression = np.random.rand(5000)  # Replace with actual isoform data

# Plot the latent space
plt.scatter(latent_2d[:, 0], latent_2d[:, 1], c=mean_expression, cmap='viridis')
plt.colorbar(label='Isoform Expression (Mean)')
plt.title("Latent Space Visualization")
plt.show()
