# models.py

import torch.nn as nn
import torch
from torch.utils.data import DataLoader
from torchinfo import summary

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
        
    def forward(self, x):
        self.decode(self.encode(x))