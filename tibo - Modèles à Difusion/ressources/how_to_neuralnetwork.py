"""
Procédure typique d'entraînement d'un réseau de neurones :
    1 - Définir le réseau de neurones, ses paramètres (ou "poids")
    2 - Itérer sur des données en entrée
    3 - Faire passer ces données dans le réseau (forward pass)
    4 - Calculer la fonction de perte (loss = criterion(prediction, attendu))
    5 - Rétropropager les gradients dans l'attribut .grad des paramètres
    6 - Mettre à jour ces paramètres, usuellement avec une règle facile, comme weight -= learning_rate * gradient
"""

import torch
import torch.nn as nn  # classes parentes des réseaux, fonctions de perte
import torch.nn.functional as F  # fonctions d'activation
import torch.optim  # fonctions d'optimisation

""" 1 - Définir le réseau

On doit définir les paramètres du réseau (ses couches, ses convolutions...)
et aussi sa fonction forward. Backward est automatiquement construite par autograd héritée de nn.Module.

on va partir sur le modèle "nn_mnist.png" """


class Net(nn.Module):
    
    def __init__(self):
        super(Net, self).__init__()
        
        # input : image MNIST 32*32
        
        # deux étages de convolutions, de noyau 5*5 :
        # premier étage : 6 convolutions
        self.conv1 = nn.Conv2d(1, 6, 5)
        # deuxième étage :  16 convolutions
        self.conv2 = nn.Conv2d(6, 16, 5)
        
        # étages affines (FullyConnected) sur 16 sorties de dimension 5*5
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, input_):
        # Convolution layer C1: recoit 1 image 32*32,
        # en sort 6 de taille 28*28 (=bords tronqués)
        # 5x5 convolution, activation RELU
        # -> Tensor(N, 6, 28, 28), avec N taille du batch
        c1 = F.relu(self.conv1(input_))
        
        # Subsampling : maxpool 2x2 , purely functional,
        # -> Tensor(N, 6, 14, 14)
        s2 = F.max_pool2d(c1, (2, 2))
        
        # Convolution layer C3: 6 14*14 -> 16 10*10
        # 5x5 convolution, RELU
        # -> Tensor(N, 16, 10, 10)
        c3 = F.relu(self.conv2(s2))
        
        # Subsampling layer S4: maxpool 2*2, -> Tensor(N, 16, 5, 5)
        s4 = F.max_pool2d(c3, 2)
        
        # Flatten operation:  -> Tensor(N, 400)
        s4 = torch.flatten(s4, 1)
        
        # Fully connected layer F5: Tensor(N, 400) -> Tensor(N, 120)
        # activation RELU
        f5 = F.relu(self.fc1(s4))
        
        # Fully connected layer F6: Tensor(N, 120) -> Tensor(N, 84)
        # activation RELU
        f6 = F.relu(self.fc2(f5))
        
        # Gaussian layer OUTPUT: Tensor(N, 84) -> Tensor(N, 10)
        output = self.fc3(f6)
        
        return output


net = Net()
# print(net)

# on peut afficher les paramètres entraînables du réseau avec :
# params = list(net.parameters())
# print(f"{len(params)=}") # nombre de paramètres
# print(f"{params[0].size()=}")  # tailles des paramètre de la première couche de convolution


""" 4,5 - Calculer la fonction de perte, rétropropager """

# on va se donner une "image" au hasard. On pourra tester sur MNIST pus tard.
image = torch.randn(1, 1, 32, 32)

output = net(image)
target = torch.randn(10)  # a dummy target, for example
target = target.view(1, -1)  # make it the same shape as output
criterion = nn.MSELoss() # on choisit la fonction de perte ( = critère de la descente)

loss = criterion(output, target)

# loss a compris, d'après "output", quels paramètres sont en jeu (grace à autograd encore).
# on peut donc immédiatement faire le backward:
loss.backward()

# ATTENTION : il faut remmetre à 0 les .grad avant d'appeler à nouveau backward() !!!!!!
# la fonction est préécrite :
net.zero_grad()



""" 6 - Mettre à jour les paramètres du modèle """

# créer l'optimiseur. Il sait sur quels paramètres optimiser.
optim = torch.optim.SGD(net.parameters(),
                        lr=0.01)
optim.zero_grad()
optim.step()




