import torch
import torchvision
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import nn

import torchvision.transforms as transforms
from torch.utils.data import DataLoader, SubsetRandomSampler

import yaml
from unet2 import UNetMNIST
from newUNET import UNet

#### CHARGEMENT DES DONNÉES
def get_dataloader(config):
    transform = transforms.Compose([transforms.ToTensor()])
    TRANSFORMS = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,)),
    ])

    ds = torchvision.datasets.MNIST("/tmp/mnist", train=True, transform=TRANSFORMS,
                                    target_transform=None, download=True)
    K = len(ds)  # dataset réduit pour epochs plus courtes
    subsample_train_indices = torch.randperm(len(ds))[:K]
    dataloader = DataLoader(ds, batch_size=config['batch_size'], drop_last=True,
                            sampler=SubsetRandomSampler(subsample_train_indices))

    
    return dataloader



#### ENTRAÎNEMENT
def train(dataloader_, model_):
    
    mps_device = torch.device("mps")
    
    criterion = nn.MSELoss()
    model_.train()
    model_.to(mps_device)
    for epoch in range(config['num_epochs']):
        for X0, _ in tqdm(dataloader_, desc=f"Epoch {epoch}/{config['num_epochs']}", leave=True):
            e = torch.randn_like(X0)
            t = torch.randint(0, config['num_timesteps'], size=[config['batch_size']])
            
            Xt = alpha[t].sqrt().view(-1, 1, 1, 1) * X0 + (1 - alpha[t]).sqrt().view(-1, 1, 1, 1) * e
            
            Xt = Xt.to(mps_device)
            t = t.to(mps_device)
            e = e.to(mps_device)
    
            e_theta = model_(Xt, t)
            loss = criterion(e, e_theta)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        # Visualisation des échantillons intermédiaires après chaque epoch
        if epoch % 1 == 0:  # Afficher tous les 1 epochs (tu peux ajuster selon ton besoin)
            model_.eval()
            with torch.no_grad():
                # Visualisation des reconstructions intermédiaires à différents niveaux de bruit
                visualize_intermediate_samples(model_, Xt, alpha)
            model_.train()
            
        # generate sample every epoch
        model_.eval()
        generate_sample(model_)
        model_.train()
    
    torch.save(model_.state_dict(), config['model_path'])
    


### LOAD DE MODÈLE ENREGISTRÉ
def load_model(path):
    # Charger le modèle
    model = UNetMNIST(im_channels=config['im_channels'])
    model.load_state_dict(torch.load(path, map_location=torch.device('cpu')))
    model.eval()
    
    return model


### TESTS DE RECONSTRUCTION
@torch.no_grad()
def reconstruction(model, dataloader, config):
    
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    model.to(device)
    model.eval()

    T = config['num_timesteps']
    beta = torch.linspace(config['beta_start'], config['beta_end'], T).to(device)
    alpha = torch.cumprod(1 - beta, dim=0).to(device)

    for X0, _ in dataloader:
        X0 = X0.to(device)
        t = torch.randint(0, T, (X0.size(0),), device=device)

        Xt = (alpha[t].view(-1, 1, 1, 1).sqrt() * X0 +
              (1 - alpha[t]).view(-1, 1, 1, 1).sqrt() * torch.randn_like(X0))

        X_reconstructed = model(Xt, t)

        for i in range(min(5, X0.size(0))):
            plt.subplot(2, 5, i + 1)
            plt.imshow(X0[i].squeeze().cpu().numpy(), cmap='gray')
            plt.title("Original")
            plt.axis('off')

            plt.subplot(2, 5, i + 6)
            plt.imshow(X_reconstructed[i].squeeze().cpu().numpy(), cmap='gray')
            plt.title("Reconstructed")
            plt.axis('off')

        plt.show()
        break


def visualize_intermediate_samples(model, Xt, alpha):
    """Visualisation des reconstructions intermédiaires à différents niveaux de bruit"""
    
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    
    num_steps_to_visualize = 5  # Par exemple, visualiser 5 étapes de bruit différentes
    step_interval = len(alpha) // num_steps_to_visualize
    
    # Créer une figure pour afficher les échantillons
    plt.figure(figsize=(10, 10))
    
    for i, step in enumerate(range(0, len(alpha), step_interval)):
        t_step = step
        Xt_step = (Xt * alpha[t_step].sqrt().view(-1, 1, 1, 1) +
                   (1 - alpha[t_step]).sqrt().view(-1, 1, 1, 1) * torch.randn_like(Xt))
        
        # Passer à travers le modèle pour obtenir la reconstruction
        Xt_step_reconstructed = model(Xt_step, torch.tensor([t_step] * Xt_step.size(0)).to(mps_device))
        
        # Afficher l'image reconstruite à cette étape
        plt.subplot(1, num_steps_to_visualize, i + 1)
        plt.imshow(Xt_step_reconstructed[0].squeeze().cpu().detach().numpy(), cmap='gray')
        plt.title(f"Step {t_step}")
        plt.axis('off')
    
    plt.show()
    
    
### GÉNÉRATION
@torch.no_grad()
def generate_sample(model):
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    model.to(device)

    X = torch.randn(1, config['im_channels'], config['im_size'], config['im_size']).to(device)  # Bruit initial

    for t in reversed(range(config['num_timesteps'])):
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
    

if __name__ == '__main__':
    
    config = yaml.safe_load(open('../le retour du roi/config_ddpm.yaml', 'r'))
    
    # Je suis sur mac, mps est l'équivalent cuda
    if torch.backends.mps.is_available():
        mps_device = torch.device("mps")  # MACBOOK MPS
    else:
        print("MPS device not found.")
        mps_device = torch.device("cpu")
    
    #model = UNetMNIST(im_channels=config['im_channels'])
    model = UNet(image_channels=config['im_channels'])
    optimizer = torch.optim.Adam(model.parameters(), lr=config['lr'])
    
    beta = torch.linspace(config['beta_start'], config['beta_end'], config['num_timesteps'])
    alpha = torch.cumprod(1 - beta, dim=0)
    
    
    ### APPELS AUX FONCTIONS
    
    dataloader = get_dataloader(config)
    train(dataloader, model)
    #model = load_model(config['model_path'])
    
    reconstruction(model, dataloader, config)
    
    
    