import torch 
import torch.nn as nn
import torch.nn.functional as F 


class Generator(nn.Module):

    def __init__(self, layers_dim, activation_func):
        super(Generator, self).__init__()
        self.layers = self._build(layers_dim, activation_func)

    def _build(self, layers_dim, activation_func):
        layers = []
        # the first item in layers_dim would be the latent space dimension
        for i in range(1, len(layers_dim)):
            layer = nn.Linear(layers_dim[i-1], layers_dim[i])
            layers.append(layer)
            layers.append(activation_func)
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.layers(x)
        return x


class Discriminator(nn.Module):

    def __init__(self, layers_dim, internal_activation, output_activation):
        super(Discriminator, self).__init__()
        self.layers = self._build(layers_dim, internal_activation, output_activation)
    
    def _build(self, layers_dim, internal_activation, output_activation):
        layers = []
        for i in range(1, len(layers_dim)):
            layer = nn.Linear(layers_dim[i-1], layers_dim[i])
            layers.append(layer)
            # append internal activation if this is not the last layer, otherwise, append the 
            # output activation
            layers.append(internal_activation if i < len(layers_dim) - 1 else output_activation)

    def forward(self, x):
        x = self.layers(x)
        return x
