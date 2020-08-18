import torch
import torch.nn as nn
import torchvision
from datetime import datetime
import models.WGANGP as WGANGP
import os
from collections import OrderedDict

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load the model
MODEL_PATH = os.path.dirname(os.path.realpath(__file__)) + '/../saved_models/'
MODEL_NAME = 'wgangp-model-epoch300.pth'
Z_DIM = 100
G = WGANGP.Generator(
    channels=[Z_DIM, 256, 128, 64, 1],
    kernels=[None, 7, 5, 4, 4],
    strides=[None, 1, 1, 2, 2],
    paddings=[None, 0, 2, 1, 1],
    batch_norm=True,
    internal_activation=nn.ReLU(),
    output_activation=nn.Tanh()
)
# reference: https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686/3
# loading a model that was wrapped by nn.DataParallel for training
checkpoint = torch.load(MODEL_PATH + MODEL_NAME, map_location=device)
old_G_state_dict = checkpoint.get('G_state_dict')
# if the model was wrapped by nn.DataParallel
if 'module.' in list(old_G_state_dict.keys())[0]:
    new_G_state_dict = OrderedDict()
    for key, value in old_G_state_dict.items():
        # remove "module." from each key
        name = key[7:]
        new_G_state_dict[name] = value
    # load the newly created state dict
    G.load_state_dict(new_G_state_dict)
else:
    G.load_state_dict(old_G_state_dict)

# generate new image
sample = torch.randn((1, Z_DIM))
print(f'Sample: {sample}\n')
sample = sample.view((-1, Z_DIM, 1, 1))
sample = G(sample)
filename = datetime.now().strftime('%d_%m_%Y_%H%M%S')
torchvision.utils.save_image(sample, f'{filename}.png')