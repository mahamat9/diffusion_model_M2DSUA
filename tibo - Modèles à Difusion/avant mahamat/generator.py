import torch
import matplotlib.pyplot as plt
from unet2 import UNetMNIST
import yaml

# Charger la configuration
config = yaml.safe_load(open('../le retour du roi/config_ddpm.yaml', 'r'))

# Charger les paramètres de diffusion
T = config['num_timesteps']
beta = torch.linspace(config['beta_start'], config['beta_end'], T)
alpha = torch.cumprod(1 - beta, dim=0)
im_channels = config['im_channels']
im_size = config['im_size']

def load_model(path) :
    # Charger le modèle
    model = UNetMNIST(im_channels=im_channels)
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.eval()
    
    return model

# Générer une image
@torch.no_grad()
def generate_sample(model):
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    model.to(device)

    X = torch.randn(1, im_channels, im_size, im_size).to(device)  # Bruit initial

    for t in reversed(range(T)):
        t_tensor = torch.tensor([t], dtype=torch.float32).to(device)
        Zt = torch.randn_like(X) if t > 0 else torch.zeros_like(X)  # Pas de bruit à t=0

        inv_one_minus_beta_sqrt = 1 / ((1 - beta[t]).sqrt())
        model_factor = beta[t] / ((1 - alpha[t]).sqrt())

        X = inv_one_minus_beta_sqrt * (X - model_factor * model(X, t_tensor)) + beta[t].sqrt() * Zt

    img = X.squeeze().cpu().detach().numpy()
    plt.figure(figsize=(2, 2))
    plt.imshow(img, cmap='gray')
    plt.axis('off')
    plt.show()

# Générer une image
if __name__ == "__main__":
    loaded_model = load_model("ddpm.pth")
    generate_sample(loaded_model)

