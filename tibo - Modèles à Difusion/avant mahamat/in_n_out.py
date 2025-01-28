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



def load_model(path):
    # Charger le modèle
    model = UNetMNIST(im_channels=im_channels)
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.eval()
    
    return model





if __name__ == "__main__":
    config = get_config()
    dataloader = get_dataloader(config['batch_size'])

    model = UNetMNIST(im_channels=config['im_channels'])
    train_model(model, dataloader, config)

    generate_samples(model, config)
    test_reconstruction(model, dataloader, config)
