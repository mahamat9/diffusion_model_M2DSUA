# Imports
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt


T = 10000  # nbre d'étapes de diff
beta_start = 1e-4
beta_end = 0.02
betas = torch.linspace(beta_start, beta_end, T)  # HYPERPARMETRES du noise
alphas = 1.0 - betas
alpha_bar = torch.cumprod(alphas, dim=0)  # produit cum alphas

def diffusion_process(x0, t):
    """Ajoute du bruit à une image x0 selon l'étape t."""
    bruit = torch.randn_like(x0)
    sqrt_alpha_bar_t = torch.sqrt(alpha_bar[t]).view(-1, 1, 1, 1)
    root_un_moins_alpha_bar_t = torch.sqrt(1 - alpha_bar[t]).view(-1, 1, 1, 1)
    return sqrt_alpha_bar_t * x0 + root_un_moins_alpha_bar_t * bruit, bruit

def simul_diffusion():
    # Chargement d'une image (ici, une image aléatoire pour illustration)
    x0 = torch.rand(1, 1, 28, 28)  # Image aléatoire 28x28
    fig, axs = plt.subplots(1, 5, figsize=(15, 5))
    
    axs[0].set_title("Image originale")
    axs[0].imshow(x0[0, 0].detach().numpy(), cmap="gray")
    for i, t in enumerate([2500, 5000, 7500, 9999]):
        xt, _ = diffusion_process(x0, t)
        axs[i+1].imshow(xt[0, 0].detach().numpy(), cmap="gray")
        axs[i+1].set_title(f"Étape {t}")
        axs[i+1].axis("off")

    plt.savefig("simu_diffusion.png")
    plt.show()

simul_diffusion()
