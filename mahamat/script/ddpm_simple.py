import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
import pickle
import numpy as np
import os

# ---- CHARGEMENT DES DONNÉES ----
def load_cifar_batch(file_path):
    try:
        with open(file_path, 'rb') as f:
            batch = pickle.load(f, encoding='bytes')
        images = batch[b'data'].reshape(-1, 3, 32, 32).astype(np.float32) / 255.0
        return images
    except Exception as e:
        print(f"Erreur lors du chargement du fichier {file_path}: {e}")
        raise

def load_cifar_data(data_dir):
    train_images = []
    for i in range(1, 6):
        batch_path = os.path.join(data_dir, f"data_batch_{i}")
        images = load_cifar_batch(batch_path)
        train_images.append(images)
    train_images = np.concatenate(train_images)
    return torch.tensor(train_images).permute(0, 2, 3, 1)  # [N, H, W, C]

# Normalisation et dataset
data_dir = "./data/cifar-10-batches-py"
train_images = load_cifar_data(data_dir)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Entre -1 et 1
])

class CIFAR10Dataset(Dataset):
    def __init__(self, images, transform=None):
        self.images = images
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        if self.transform:
            if not isinstance(image, torch.Tensor):
                image = self.transform(image)
        return image


# ---- ARCHITECTURE DU MODÈLE ----
class SimpleUNet(nn.Module):
    def __init__(self):
        super(SimpleUNet, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        self.middle = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, kernel_size=3, stride=1, padding=1)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.middle(x)
        x = self.decoder(x)
        return x

# ---- PIPELINE DDPM ----
class DDPM:
    def __init__(self, model, betas, timesteps=1000):
        self.model = model
        self.timesteps = timesteps
        self.betas = betas
        self.alphas = 1.0 - betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        
        print(f"x_start shape: {x_start.shape}")  # Doit être [batch_size, 3, 32, 32]
        print(f"noise shape: {noise.shape}")      # Doit être identique à x_start
        print(f"alpha_bar[t] shape: {self.alpha_bar[t].shape}")  # Doit être [batch_size]
        
        alpha_bar_t = self.alpha_bar[t].view(-1, 1, 1, 1)  # Diffusion des dimensions

        return (
            torch.sqrt(alpha_bar_t) * x_start +
            torch.sqrt(1 - alpha_bar_t) * noise
        )

    def p_sample(self, x, t):
        pred_noise = self.model(x)
        return (
            torch.sqrt(self.alpha_bar[t]) * x -
            torch.sqrt(1 - self.alpha_bar[t]) * pred_noise
        )

    def train_step(self, x):
        t = torch.randint(0, self.timesteps, (x.size(0),)).to(x.device)
        noise = torch.randn_like(x)
        x_noisy = self.q_sample(x, t, noise)
        pred_noise = self.model(x_noisy)
        loss = F.mse_loss(pred_noise, noise)
        return loss
    
    
# ---- GÉNÉRATION D'IMAGES ----
def generate_images(ddpm, num_samples=16):
    with torch.no_grad():
        x = torch.randn(num_samples, 3, 32, 32).to(device)
        for t in reversed(range(ddpm.timesteps)):
            x = ddpm.p_sample(x, t)
    return x



if __name__ == '__main__':
    
    dataset = CIFAR10Dataset(train_images, transform=transform)
    data_loader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=0)
    
    # ---- INITIALISATION ----
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    betas = torch.linspace(0.0001, 0.02, 1000).to(device)  # Planification de bruit
    model = SimpleUNet().to(device)
    ddpm = DDPM(model=model, betas=betas)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    # ---- ENTRAÎNEMENT ----
    epochs = 10
    for epoch in range(epochs):
        for images in data_loader:
            images = images.to(device)
            optimizer.zero_grad()
            loss = ddpm.train_step(images)
            loss.backward()
            optimizer.step()

        print(f"Époque {epoch + 1}/{epochs}, Perte : {loss.item():.4f}")



    # Exemple de génération
    generated_images = generate_images(ddpm, num_samples=8)
    print("Images générées !")
