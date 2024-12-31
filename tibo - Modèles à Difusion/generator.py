import torch
import torchvision
from matplotlib import pyplot as plt
from torch import nn

import torchvision.transforms as transforms
from torch.utils.data import DataLoader, SubsetRandomSampler

import numpy as np

import yaml
from UNET import Unet

model = Unet(im_channels=1)
model.load_state_dict(torch.load('ddpm.pth', map_location=torch.device('cpu')))

X = torch.randn(1, 28, 28)
t = torch.randint(1, 1000, size=[1])

out = model(X, t)

