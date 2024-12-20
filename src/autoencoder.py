# models.py

import torch.nn as nn
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

class Autoencoder(nn.Module):
    def __init__(self, 
                 in_dim: int,
                 latent_dim: int,
                 encode_neurons: list,
                 decode_neurons: list = None,
                 batch_norm: bool = True):
        super().__init__()

        encode_layers = []
        decode_layers = []

        self.latent_dim = latent_dim

        for i, neuron_count in enumerate(encode_neurons):
            if i == 0:
                encode_layers.append(nn.Linear(in_dim, neuron_count))
                encode_layers.append(nn.ReLU())
            elif i == (len(encode_neurons) - 1):
                encode_layers.append(nn.Linear(neuron_count, latent_dim))
                if batch_norm: encode_layers.append(nn.BatchNorm1d(latent_dim))
            else:
                encode_layers.append(nn.Linear(encode_neurons[i - 1],
                                               neuron_count))
                encode_layers.append(nn.ReLU())
        
        if decode_neurons:
            for i, neuron_count in enumerate(decode_neurons):
                if i == 0:
                    decode_layers.append(nn.Linear(latent_dim, neuron_count))
                    decode_layers.append(nn.ReLU())
                elif i == (len(decode_neurons) - 1):
                    decode_layers.append(nn.Linear(neuron_count, in_dim))
                else:
                    decode_layers.append(nn.Linear(decode_neurons[i - 1],
                                                neuron_count))
                    decode_layers.append(nn.ReLU())
        else:
            reversed_encode = list(reversed(encode_neurons))
            for i, neuron_count in enumerate(reversed_encode):
                if i == 0:
                    decode_layers.append(nn.Linear(latent_dim, neuron_count))
                    decode_layers.append(nn.ReLU())
                elif i == (len(encode_neurons) - 1):
                    decode_layers.append(nn.Linear(neuron_count, in_dim))
                else:
                    decode_layers.append(nn.Linear(reversed_encode[i - 1],
                                                neuron_count))
                    decode_layers.append(nn.ReLU())

        self.encode = nn.Sequential(*encode_layers)
        self.decode = nn.Sequential(*decode_layers)
        
    def forward(self,
                x: torch.Tensor):
        return self.decode(self.encode(x))

    def viz_latent_space(self, loader: DataLoader):
        self.eval()
        latent_space = []
        labels = []
        for (input, label) in loader:
            input = input.view(input.size(0), -1)
            output = self.encode(input)
            latent_space.append(output.detach().numpy())
            labels.append(label.detach().numpy())
        latent_space = np.concatenate(latent_space, axis = 0)
        labels = np.concatenate(labels, axis = 0)
        if self.latent_dim > 2:
            print(f'Latent space dim {self.latent_dim} > 2, using PCA.')
            pca = PCA(n_components=2)
            latent_space = pca.fit_transform(latent_space)
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.scatter(latent_space[:, 0], latent_space[:, 1], c = labels)
        ax.legend()

    def train_now(self, data: Dataset, batch_size: int = 32,
              epochs: int = 1, lr: float = 0.01,
              optimizer: optim.Optimizer = optim.Adam, 
              criterion: nn.Module = nn.MSELoss(),
              print_loss: bool = True):
        self.train()
        loader = DataLoader(data, batch_size=batch_size, shuffle=True)
        optimizer = optimizer(self.parameters(), lr=lr)
        for epoch in range(epochs):
            running_loss = 0.
            for (input, _) in loader:
                input = input.view(input.size(0), -1)
                pred = self(input)
                loss = criterion(input, pred)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                running_loss += loss.item()
            if print_loss: print(f'Epoch: {epoch} | Loss: {running_loss}')

    def test_loss(self, data: Dataset, batch_size: int = 32,
                 criterion: nn.Module = nn.MSELoss(),
                 print_loss: bool = True):
        self.eval()
        loader = DataLoader(data, batch_size=batch_size, shuffle=False)
        total_loss = 0.
        for (input, _) in loader:
            input = input.view(input.size(0), -1)
            pred = self(input)
            loss = criterion(input, pred)
            total_loss += loss.item()
        if print_loss: print(f'Test Loss: {total_loss}.')
        return total_loss
    
    def relative_representation(self, data: Dataset,
                                anchors: torch.Tensor):
        return data