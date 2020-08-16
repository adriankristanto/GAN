import torch
import torch.nn as nn

class Generator(nn.Module):

    def __init__(self, channels, kernels, strides, paddings, batch_norm, internal_activation, output_activation):
        super(Generator, self).__init__()
        self.layers = self._build(channels, kernels, strides, paddings, batch_norm, internal_activation, output_activation)

    def _build(self, channels, kernels, strides, paddings, batch_norm, internal_activation, output_activation):
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
