import torch
import torch.nn as nn

class Generator(nn.Module):

    def __init__(self, channels, kernels, strides, paddings, batch_norm, internal_activation, output_activation):
        super(Generator, self).__init__()
        self.layers = self._build(channels, kernels, strides, paddings, batch_norm, internal_activation, output_activation)

    def _build(self, channels, kernels, strides, paddings, batch_norm, internal_activation, output_activation):
        # the first input is obtained from the latent space z
        # which should be reshaped into (z_dim, 1, 1) where z_dim is the channel
        layers = []
        for i in range(1, len(channels)):
            # add convtranspose layer
            layer = nn.ConvTranspose2d(
                in_channels=channels[i-1],
                out_channels=channels[i],
                kernel_size=kernels[i],
                stride=strides[i],
                padding=paddings[i]
            )
            layers.append(layer)
            # add batchnorm layer if batch_norm == True and this is not the final layer
            if batch_norm and i < len(channels) - 1:
                layers.append(nn.BatchNorm2d(channels[i]))
            # add activation layer
            # use internal activation if this is still not the final layer, otherwise, use output activation
            layers.append(internal_activation if i < len(channels) - 1 else output_activation)
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.layers(x)
        return x

class Discriminator(nn.Module):

    def __init__(self, channels, kernels, strides, paddings, batch_norm, internal_activation, output_activation):
        super(Discriminator, self).__init__()
        self.layers = self._build(channels, kernels, strides, paddings, batch_norm, internal_activation, output_activation)
    
    def _build(self, channels, kernels, strides, paddings, batch_norm, internal_activation, output_activation):
        layers = []
        for i in range(1, len(channels)):
            # add convolutional layer or strided convolutional layer
            layer = nn.Conv2d(
                in_channels=channels[i-1],
                out_channels=channels[i],
                kernel_size=kernels[i],
                stride=strides[i],
                padding=paddings[i]
            )
            layers.append(layer)
            # add batch norm layer
            if batch_norm and i < len(channels) - 1:
                layers.append(nn.BatchNorm2d(channels[i]))
            # add activation function
            layers.append(internal_activation if i < len(channels) else output_activation)
    
    def forward(self, x):
        x = self.layers(x)
        return x


if __name__ == "__main__":
    Z_DIM = 100
    G = Generator(
        channels=[Z_DIM, 256, 128, 64, 1],
        kernels=[None, 7, 5, 4, 4],
        strides=[None, 1, 1, 2, 2],
        paddings=[None, 0, 2, 1, 1],
        batch_norm=True,
        internal_activation=nn.ReLU(),
        output_activation=nn.Tanh()
    )
    print(G)
    sample = torch.randn((1, Z_DIM, 1, 1))
    print(G(sample).shape)