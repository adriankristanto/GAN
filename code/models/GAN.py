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
