import torch
import torch.nn as nn
import torchvision
from datetime import datetime
import os
import models.GAN as GAN
from collections import OrderedDict

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load the model
MODEL_PATH = os.path.dirname(os.path.realpath(__file__)) + '/../saved_models/'
MODEL_NAME = 'gan-model-epoch300.pth'
Z_DIM = 100

G = GAN.Generator(
    layers_dim=[100, 256, 256, 784],
    internal_activation=nn.ReLU(),
    output_activation=nn.Tanh()
)

checkpoint = torch.load(MODEL_PATH + MODEL_NAME, map_location=device)
old_G_state_dict = checkpoint.get('G_state_dict')

if 'module.' in list(old_G_state_dict.keys())[0]:
    new_G_state_dict = OrderedDict()
    for key, value in old_G_state_dict.items():
        name = key[7:]
        new_G_state_dict[name] = value
    G.load_state_dict(new_G_state_dict)
else:
    G.load_state_dict(old_G_state_dict)

sample = torch.randn((1, Z_DIM))
print(f'Sample: {sample}\n')
sample = G(sample)
# reshape the prediction into an image
# as this is a FC GAN
sample = sample.view(-1, *(1, 28, 28))
filename = datetime.now().strftime('%d_%m_%Y_%H%M%S.png')
torchvision.utils.save_image(sample, filename)