import torch
import torchvision
from matplotlib import pyplot as plt
from tqdm import tqdm
from torch import nn

import torchvision.transforms as transforms
from torch.utils.data import DataLoader, SubsetRandomSampler

import numpy as np

import yaml
from UNET import Unet

config = yaml.safe_load(open('configDDPM.yaml', 'r'))

# Je suis sur mac, mps est l'équivalent cuda
if torch.backends.mps.is_available():
    mps_device = torch.device("mps")  # MACBOOK MPS
else:
    print("MPS device not found.")
    mps_device = torch.device("cpu")





#### CONFIGURATION

# paramètres de diffusion
T = config['num_timesteps']
beta = torch.linspace(config['beta_start'], config['beta_end'], config['num_timesteps'])
alpha = torch.cumprod(1 - beta, dim=0)

# paramètres du modèle
im_channels = config['im_channels']
im_size = config['im_size']

# paramètres du train
batch_size = config['batch_size']
num_epochs = config['num_epochs']
num_samples = config['num_samples']
lr = config['lr']





#### CHARGEMENT DES DONNÉES

TRANSFORMS = transforms.Compose([
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
])

ds = torchvision.datasets.MNIST("/tmp/mnist", train=True, transform=TRANSFORMS,
                                target_transform=None, download=True)
K = 6000  # dataset réduit pour epochs plus courtes
subsample_train_indices = torch.randperm(len(ds))[:K]
dataloader = DataLoader(ds, batch_size=batch_size, drop_last=True,
                        sampler=SubsetRandomSampler(subsample_train_indices))





#### CHARGEMENT DU MODÈLE

model = Unet(im_channels=im_channels, )
optimizer = torch.optim.Adam(model.parameters(), lr=lr)





#### GÉNÉRATEUR

model.eval()
def generate_sample():
    X = torch.randn([im_size, im_size])
    X = X.view(im_channels, im_size, im_size)
    
    for t in reversed(range(T)):
        print("RUN ", t)
        
        t = torch.as_tensor([t])
        Zt = torch.randn_like(X)
        print(X.shape)
        out = model(X, t)
        inv_one_minus_beta_sqrt = 1 / ((1 - beta[t]).sqrt())
        model_factor = beta[t] / ((1 - alpha[t]).sqrt())
        print((X - model_factor*model(X, t)).shape)
        X = inv_one_minus_beta_sqrt * (X - model_factor*model(X, t))
        + beta[t] * Zt
        
    
    img = X.squeeze().cpu().detach().numpy()
    plt.figure(figsize=(2, 2))
    plt.imshow(img, cmap='gray')
    plt.show()


#generate_sample()




#### ENTRAÎNEMENT

criterion = nn.MSELoss()
model.train()
model.to(mps_device)
for epoch in range(num_epochs):
    for X0, _ in tqdm(dataloader, desc=f"Epoch {epoch}/{num_epochs}", leave=True):
        e = torch.randn_like(X0)
        t = torch.randint(1, T, size=[batch_size])
        
        Xt = alpha[t].sqrt().view(-1, 1, 1, 1) * X0 + (1 - alpha[t]).sqrt().view(-1, 1, 1, 1) * e
        
        Xt = Xt.to(mps_device)
        t = t.to(mps_device)
        e = e.to(mps_device)

        e_theta = model(Xt, t)
        loss = criterion(e, e_theta)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    #generate_sample()

torch.save(model.state_dict(), 'ddpm.pth')

