import pickle
import numpy as np
import os
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import torch

# path vers le dossier contenant CIFAR-10
data_dir = "./data/cifar-10-batches-py"  

# Fonction pour charger un fichier CIFAR-10
def load_cifar_batch(file_path):
    with open(file_path, 'rb') as f:
        batch = pickle.load(f, encoding='bytes')  # Lecture en binaire
    images = batch[b'data']  # Les images sont sous forme de vecteurs 1D
    labels = batch[b'labels']
    # Reshape pour correspondre à [Nombre d'images, Canaux, Hauteur, Largeur]
    images = images.reshape(-1, 3, 32, 32).astype(np.float32)
    return images, labels

# Charger toutes les données d'entraînement
def load_cifar_data(data_dir):
    train_images = []
    train_labels = []
    for i in range(1, 6):  # Les 5 batches d'entraînement
        batch_path = os.path.join(data_dir, f"data_batch_{i}")
        images, labels = load_cifar_batch(batch_path)
        train_images.append(images)
        train_labels.extend(labels)
    train_images = np.concatenate(train_images)
    train_labels = np.array(train_labels)
    return train_images, train_labels

# Charger les données de test
def load_cifar_test(data_dir):
    test_path = os.path.join(data_dir, "test_batch")
    return load_cifar_batch(test_path)




# **Étape 2 : Transformation pour Torch**

# Pre-traitement des données 
transform = transforms.Compose([
    transforms.ToTensor(), # Convertit l'image (PIL et NumPy) en tenseur PyTorch.
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # Normalisation entre -1 et 1 des RGB
])

# Dataset personnalisé
class CIFAR10Dataset(Dataset):
    def __init__(self, images, labels, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label



if __name__ == '__main__':
    train_images, train_labels = load_cifar_data(data_dir)
    test_images, test_labels = load_cifar_test(data_dir)

    print(f"Entraînement: {train_images.shape}, {train_labels.shape}")
    print(f"Test: {test_images.shape}, {len(test_labels)}")


    # Création des DataLoaders
    train_dataset = CIFAR10Dataset(train_images, train_labels, transform=transform)
    test_dataset = CIFAR10Dataset(test_images, test_labels, transform=transform)


    # divise  en batchs
    #shuffle : mixing aléatoire
    #Chargement parallèle
    # num_workers(option) : num_de_threads<num_core
    train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

    # Exemple d'itération sur les données
    for images, labels in train_loader:
        print(f"Images batch: {images.size()}, Labels batch: {labels.size()}")
        break
