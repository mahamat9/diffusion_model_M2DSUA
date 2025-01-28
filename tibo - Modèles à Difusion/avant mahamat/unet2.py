import torch
from torch import nn


class UNetMNIST(nn.Module):
    def __init__(self, im_channels):
        super(UNetMNIST, self).__init__()
        # Projection pour ajuster les canaux après concaténation
        self.input_proj = nn.Conv2d(im_channels + 1, 64, kernel_size=3, stride=1, padding=1)
        
        # Encoder
        self.encoder1 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.encoder2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder1 = nn.Sequential(
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )
        self.decoder2 = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU()
        )
        self.output_layer = nn.Conv2d(64, im_channels, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x, t):
        # Redimensionner et concaténer le tenseur temporel
        t_embedding = t.view(t.size(0), 1, 1, 1).expand(-1, 1, x.size(2), x.size(3))
        x = torch.cat([x, t_embedding], dim=1)
        
        # Adapter les canaux avec une couche convolutive
        x = self.input_proj(x)
        
        # UNet classique
        x = self.encoder1(x)
        x = self.encoder2(x)
        x = self.bottleneck(x)
        x = self.decoder1(x)
        x = self.decoder2(x)
        x = self.output_layer(x)
        return x
