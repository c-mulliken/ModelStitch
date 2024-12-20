# models.py

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt

class Autoencoder(nn.Module):
    def __init__(self, 
                 in_dim: int,
                 latent_dim: int,
                 encode_neurons: list,
                 decode_neurons: list = None
                 ):
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
        self.decode(self.encode(x))

    def viz_latent_space(self, loader: DataLoader):
        self.eval()
        outs = []
        labels = []
        if self.latent_dim == 2:
            for (input, label) in loader:
                out = self.encode(input.view(input.size(0), -1))
                outs.append(out.detach().numpy())
                labels.append(label.detach().numpy())
        elif self.latent_dim > 2:
            print(f'Latent space dim {self.latent_dim} > 2, using PCA.')
            for (input, label) in loader:
                out = self.encode(input.view(input.size(0), -1))
                outs.append(out.detach().numpy())
                labels.append(label.detach().numpy())
            pca = PCA(n_components=2)
            outs = pca.fit_transform(outs)
        outs.concatenate(outs, axis = 0)
        labels.concatenate(labels, axis = 0)
        fig, ax = plt.subplots(figsize=(10, 10))
        ax.scatter(outs[:, 0], outs[:, 1], c = labels)
        ax.legend()

ae = Autoencoder(5, 2, [5, 5, 5])
t = torch.tensor([1., 2., 3., 4., 5.], dtype=torch.float32)