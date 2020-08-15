import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm
import os 
from models.GAN import GAN
from datetime import datetime

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'Current device: {device}', flush=True)

# 1. load the data
DATA_PATH = os.path.dirname(os.path.realpath(__file__)) + '/../data/'
BATCH_SIZE = 128

train_transform = transforms.Compose([
    transforms.ToTensor()
])

test_transform = transforms.Compose([
    transforms.ToTensor()
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

# 2. instantiate the network model
Z_DIM = 100

net = GAN(
    # the latent space dimension is 100
    g_dims=[Z_DIM, 256, 400, 784],
    g_in_activation=nn.LeakyReLU(),
    g_out_activation=nn.Tanh(),
    # the final layer for the discriminator is just a neuron that returns [0, 1]
    d_dims=[784, 400, 256, 1],
    d_in_activation=nn.LeakyReLU(),
    d_out_activation=nn.Sigmoid()
)

# if we train with multiple GPUs, we need to use 
# net.module.generate()
# with only one GPU, we can simply use
# net.generate()
multigpu = False
if torch.cuda.device_count() > 1:
    print(f'Number of GPUs: {torch.cuda.device_count()}\n', flush=True)
    net = nn.DataParallel(net)
    multigpu = True

net.to(device)