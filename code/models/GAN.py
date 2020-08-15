import torch 
import torch.nn as nn
import torch.nn.functional as F 


class Generator(nn.Module):

    def __init__(self, layers_dim, internal_activation, output_activation):
        super(Generator, self).__init__()
        self.layers = self._build(layers_dim, internal_activation, output_activation)

    def _build(self, layers_dim, internal_activation, output_activation):
        layers = []
        # the first item in layers_dim would be the latent space dimension
        for i in range(1, len(layers_dim)):
            layer = nn.Linear(layers_dim[i-1], layers_dim[i])
            layers.append(layer)
            # append internal activation if this is not the last layer, otherwise, append the 
            # output activation
            # often, the output layer uses tanh activation function to normalise the image
            # from [0, 255] to [-1, 1]
            layers.append(internal_activation if i < len(layers_dim) - 1 else output_activation)
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
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.layers(x)
        return x


class GAN(nn.Module):

    def __init__(self, g_dims, g_in_activation, g_out_activation, d_dims, d_in_activation, d_out_activation):
        super(GAN, self).__init__()
        self.generator = Generator(g_dims, g_in_activation, g_out_activation)
        self.discriminator = Discriminator(d_dims, d_in_activation, d_out_activation)

    def generate(self, x):
        x = self.generator(x)
        return x

    def discriminate(self, x):
        x = self.discriminator(x)
        return x

    def forward(self, x):
        # combined model
        x = self.generate(x)
        x = self.discriminate(x)
        return x

    
if __name__ == "__main__":
    net = GAN(
        # the latent space dimension is 100
        g_dims=[100, 256, 400, 784],
        g_in_activation=nn.LeakyReLU(),
        g_out_activation=nn.Tanh(),
        # the final layer for the discriminator is just a neuron that returns [0, 1]
        d_dims=[784, 400, 256, 1],
        d_in_activation=nn.LeakyReLU(),
        d_out_activation=nn.Sigmoid()
    )
    print(net)
    sample = torch.randn((1, 100))
    print(net.generate(sample).shape)
    print(net.discriminate(net.generate(sample)).shape)