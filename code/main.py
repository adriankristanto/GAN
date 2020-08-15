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
import numpy as np

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

# 3. define the loss function
# the discriminator will output a value in the range of [0, 1]
# where 0 means the input is fake, while 1 means the input is real
# therefore, we can use binary cross entropy loss
# the generator is trained by combining it with the discriminator
# however, we don't change the discriminator's weights
# therefore, the output that will be used by the loss function is also in the range [0, 1]
criterion = nn.BCELoss()

# 4. define the optimiser
# a. the optimiser for the generator
g_lr = 0.0002
g_optimizer = optim.Adam(net.generator.parameters(), lr=g_lr)
# b. the optimiser for the discriminator
d_lr = 0.0004
d_optimizer = optim.Adam(net.discriminator.parameters(), lr=d_lr)

# 5. train the model
MODEL_DIRPATH = os.path.dirname(os.path.realpath(__file__)) + '/../saved_models/'
GENERATED_DIRPATH = os.path.dirname(os.path.realpath(__file__)) + '/../generated_images/'
CONTINUE_TRAIN = False
CONTINUE_TRAIN_NAME = MODEL_DIRPATH + 'gan-model-epoch10.pth'
EPOCH = 50
SAVE_INTERVAL = 5
# for generation
SAMPLE = torch.randn((BATCH_SIZE, Z_DIM))

IMAGE_SIZE = (1, 28, 28)
FLATTEN_SIZE = np.prod(IMAGE_SIZE)

next_epoch = 0
if CONTINUE_TRAIN:
    checkpoint = torch.load(CONTINUE_TRAIN_NAME)
    net.load_state_dict(checkpoint.get('net_state_dict'))
    g_optimizer.load_state_dict(checkpoint.get('g_optimizer_state_dict'))
    d_optimizer.load_state_dict(checkpoint.get('d_optimizer_state_dict'))
    next_epoch = checkpoint.get('epoch')

def generate(sample, filename):
    net.eval()
    with torch.no_grad():
        sample = sample.to(device)
        sample = net.module.generate(sample) if multigpu else net.generate(sample)
        sample = sample.view(BATCH_SIZE, *IMAGE_SIZE)
        torchvision.utils.save_image(sample, filename)

def save_training_progress(net, g_optimizer, d_optimizer, epoch, target_dir):
    torch.save({
        'epoch' : epoch + 1,
        'net_state_dict' : net.state_dict(),
        'g_optimizer_state_dict' : g_optimizer.state_dict(),
        'd_optimizer_state_dict' : d_optimizer.state_dict()
    }, target_dir)

generate(SAMPLE, GENERATED_DIRPATH + 'gan_sample_0.png')