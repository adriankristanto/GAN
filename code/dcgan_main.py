import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm
import os
import models.DCGAN as DCGAN
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Current device: {device}', flush=True)

# 1. load the data
DATA_PATH = os.path.dirname(os.path.realpath(__file__)) + '/../data/'
BATCH_SIZE = 128

train_transform = transforms.Compose([
    transforms.ToTensor(),
    # each pixel of each image from the MNIST dataset is within the range [0, 1]
    # normalizing with mean=0.5 and std=0.5
    # 0 <= x <= 1
    # -0.5 <= x - 0.5 <= 0.5
    # -1 <= (x - 0.5) / 0.5 <= 1
    # thus, changing its range to [-1, 1]
    # this is because the generator uses tanh on the output layer, which has the range [-1, 1]
    transforms.Normalize((0.5,), (0.5,))
])

test_transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

trainset = datasets.MNIST(root=DATA_PATH, train=True, transform=train_transform, download=True)
testset = datasets.MNIST(root=DATA_PATH, train=False, transform=test_transform, download=True)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

print(f"""
Total training data: {len(trainset)}
Total testing data: {len(testset)}
Total data: {len(trainset) + len(testset)}
""", flush=True)

# 2. instantiate the model
Z_DIM = 100
# create the generator G
G = Generator(
    channels=[Z_DIM, 256, 128, 64, 1],
    kernels=[None, 7, 5, 4, 4],
    strides=[None, 1, 1, 2, 2],
    paddings=[None, 0, 2, 1, 1],
    batch_norm=True,
    internal_activation=nn.ReLU(),
    output_activation=nn.Tanh()
)
# create the discrimintor D
D = Discriminator(
    channels=[1, 64, 128, 256, 1],
    kernels=[None, 4, 4, 5, 7],
    strides=[None, 2, 2, 1, 1],
    paddings=[None, 1, 1, 2, 0],
    batch_norm=True,
    internal_activation=nn.LeakyReLU(0.2),
    output_activation=nn.Sigmoid()
)

print(f"""
Generator G:
{G}

Discriminator D:
{D}
""", flush=True)

multigpu = False
if torch.cuda.device_count() > 1:
    print(f'Number of GPUs: {torch.cuda.device_count()}\n', flush=True)
    G = nn.DataParallel(G)
    D = nn.DataParallel(D)
    multigpu = True

G.to(device)
D.to(device)