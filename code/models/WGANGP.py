import torch
import torch.nn as nn

# Wasserstein GAN Gradient Penalty

class Generator(nn.Module):

    def __init__(self, channels, kernels, strides, paddings, batch_norm, internal_activation, output_activation):
        super(Generator, self).__init__()
        self.layers = self._build(channels, kernels, strides, paddings, batch_norm, internal_activation, output_activation)
        self._init_weights()
    
    def _build(self, channels, kernels, strides, paddings, batch_norm, internal_activation, output_activation):
        layers = []
        for i in range(1, len(channels)):
            layer = nn.ConvTranspose2d(
                in_channels=channels[i-1],
                out_channels=channels[i],
                kernel_size=kernels[i],
                stride=strides[i],
                padding=paddings[i],
                # when using batchnorm, bias can be set to False
                bias=False
            )
            layers.append(layer)
            # add batch norm layer
            if batch_norm and i < len(channels) - 1:
                layers.append(nn.BatchNorm2d(channels[i]))
            # add activation function
            layers.append(internal_activation if i < len(channels) - 1 else output_activation)
        return nn.Sequential(*layers)
    
    # weight initialisation of vanilla DCGAN
    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.ConvTranspose2d):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        x = self.layers(x)
        return x

class Critic(nn.Module):

    # note that we don't use batchnorm in the critic
    # because gradient penalty should be independently imposed on different samples, but batch normalization brings batch correlation
    def __init__(self, channels, kernels, strides, paddings, internal_activation):
        super(Critic, self).__init__()
        self.layers = self._build(channels, kernels, strides, paddings, internal_activation)
        self._init_weights()
    
    def _build(self, channels, kernels, strides, paddings, internal_activation):
        layers = []
        for i in range(1, len(channels)):
            layer = nn.Conv2d(
                in_channels=channels[i-1],
                out_channels=channels[i],
                kernel_size=kernels[i],
                stride=strides[i],
                padding=paddings[i],
            )
            layers.append(layer)
            # add activation function when it is not the final layer
            # the output layer doesn't need any activation
            if i < len(channels) - 1:
                layers.append(internal_activation)
        return nn.Sequential(*layers)
    
    def _init_weights(self):
        # no batch norm layer
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, x):
        x = self.layers(x)
        return x


if __name__ == "__main__":
    # Generator G
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

    # Critic D
    D = Critic(
        channels=[1, 64, 128, 256, 1],
        kernels=[None, 4, 4, 5, 7],
        strides=[None, 2, 2, 1, 1],
        paddings=[None, 1, 1, 2, 0],
        internal_activation=nn.LeakyReLU(0.2),
    )
    print(D)
    print(D(G(sample)).shape)