import torch
import torchvision
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from matplotlib import pyplot as plt
from tqdm import tqdm

# Configuration
def get_config():
    return {
        'num_timesteps': 1000,
        'beta_start': 0.0001,
        'beta_end': 0.02,
        'im_channels': 1,
        'im_size': 28,
        'batch_size': 64,
        'num_epochs': 10,
        'lr': 0.001,
        'num_samples': 5
    }

# Définition de l'architecture UNet
class UNetMNIST(nn.Module):
    def __init__(self, im_channels):
        super(UNetMNIST, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(im_channels, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, im_channels, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x, t):
        t_embedding = t.view(t.size(0), 1, 1, 1).expand_as(x)
        x = torch.cat([x, t_embedding], dim=1)
        x = self.encoder(x)
        x = self.bottleneck(x)
        x = self.decoder(x)
        return x

# Chargement des données
def get_dataloader(batch_size):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])
    dataset = torchvision.datasets.MNIST(root="/tmp/mnist", train=True, transform=transform, download=True)
    return DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Entraînement du modèle
def train_model(model, dataloader, config):
    
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    T = config['num_timesteps']
    beta = torch.linspace(config['beta_start'], config['beta_end'], T).to(device)
    alpha = torch.cumprod(1 - beta, dim=0).to(device)

    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    criterion = nn.MSELoss()

    model.to(device)
    model.train()

    for epoch in range(config['num_epochs']):
        for X0, _ in tqdm(dataloader, desc=f"Epoch {epoch+1}/{config['num_epochs']}"):
            X0 = X0.to(device)
            e = torch.randn_like(X0)
            t = torch.randint(0, T, (X0.size(0),), device=device)

            Xt = (alpha[t].view(-1, 1, 1, 1).sqrt() * X0 +
                  (1 - alpha[t]).view(-1, 1, 1, 1).sqrt() * e)

            print(Xt.shape)
            optimizer.zero_grad()
            e_theta = model(Xt, t)
            loss = criterion(e, e_theta)
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1} completed. Loss: {loss.item():.4f}")

    torch.save(model.state_dict(), "ddpm_unet.pth")
    print("Model saved as ddpm_unet.pth")

# Génération d'images
@torch.no_grad()
def generate_samples(model, config):
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    
    T = config['num_timesteps']
    beta = torch.linspace(config['beta_start'], config['beta_end'], T).to(device)
    alpha = torch.cumprod(1 - beta, dim=0).to(device)

    
    model.to(device)
    model.eval()

    samples = []
    for _ in range(config['num_samples']):
        X = torch.randn(1, config['im_channels'], config['im_size'], config['im_size']).to(device)

        for t in reversed(range(T)):
            t_tensor = torch.tensor([t], dtype=torch.float32).to(device)
            Zt = torch.randn_like(X) if t > 0 else torch.zeros_like(X)

            inv_one_minus_beta_sqrt = 1 / ((1 - beta[t]).sqrt())
            model_factor = beta[t] / ((1 - alpha[t]).sqrt())

            X = inv_one_minus_beta_sqrt * (X - model_factor * model(X, t_tensor)) + beta[t].sqrt() * Zt

        samples.append(X.squeeze().cpu().numpy())

    for i, sample in enumerate(samples):
        plt.subplot(1, config['num_samples'], i + 1)
        plt.imshow(sample, cmap='gray')
        plt.axis('off')
    plt.show()

# Validation de la reconstruction
@torch.no_grad()
def validate_reconstruction(model, dataloader, config):
    
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

if __name__ == "__main__":
    config = get_config()
    dataloader = get_dataloader(config['batch_size'])

    model = UNetMNIST(im_channels=config['im_channels'])
    train_model(model, dataloader, config)

    generate_samples(model, config)
    validate_reconstruction(model, dataloader, config)

