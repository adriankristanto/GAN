import torch 
import torch.nn as nn


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

    
if __name__ == "__main__":
    G = Generator(
        layers_dim=[100, 256, 256, 784],
        internal_activation=nn.ReLU(),
        output_activation=nn.Tanh()
    )
    D = Discriminator(
        layers_dim=[784, 256, 256, 1],
        internal_activation=nn.LeakyReLU(0.2),
        output_activation=nn.Sigmoid()
    )
    print(f"""
    Generator G: 
    {G}
    Discriminator D:
    {D}
    """)
    sample = torch.randn((1, 100))
    print(G(sample).shape)
    print(D(G(sample)).shape)