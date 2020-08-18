import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm
import os 
import models.GAN as GAN
from datetime import datetime
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

# 2. instantiate the network model
Z_DIM = 100
# reference: https://machinelearningmastery.com/how-to-train-stable-generative-adversarial-networks/
G = GAN.Generator(
    layers_dim=[Z_DIM, 256, 256, 784],
    internal_activation=nn.ReLU(),
    output_activation=nn.Tanh()
)
D = GAN.Discriminator(
    layers_dim=[784, 256, 256, 1],
    internal_activation=nn.LeakyReLU(0.2),
    output_activation=nn.Sigmoid()
)
print(f"""
Generator G:
{G}

Discriminator D:
{D}
""", flush=True)

# if we train with multiple GPUs, we need to use 
# net.module.generate()
# with only one GPU, we can simply use
# net.generate()
multigpu = False
if torch.cuda.device_count() > 1:
    print(f'Number of GPUs: {torch.cuda.device_count()}\n', flush=True)
    G = nn.DataParallel(G)
    D = nn.DataParallel(D)
    multigpu = True

G.to(device)
D.to(device)

# 3. define the loss function
# the discriminator will output a value in the range of [0, 1]
# where 0 means the input is fake, while 1 means the input is real
# therefore, we can use binary cross entropy loss
# the generator is trained by combining it with the discriminator
# however, we don't change the discriminator's weights
# therefore, the output that will be used by the loss function is also in the range [0, 1]
criterion = nn.BCELoss(reduction='sum')

# 4. define the optimiser
# a. the optimiser for the generator
g_lr = 0.0002
g_optimizer = optim.Adam(net.generator.parameters(), lr=g_lr, betas=(0.5, 0.999))
# b. the optimiser for the discriminator
d_lr = 0.0002
d_optimizer = optim.Adam(net.discriminator.parameters(), lr=d_lr, betas=(0.5, 0.999))

# 5. train the model
MODEL_DIRPATH = os.path.dirname(os.path.realpath(__file__)) + '/../saved_models/'
GENERATED_DIRPATH = os.path.dirname(os.path.realpath(__file__)) + '/../generated_images/'
CONTINUE_TRAIN = False
CONTINUE_TRAIN_NAME = MODEL_DIRPATH + 'gan-model-epoch10.pth'
EPOCH = 600
SAVE_INTERVAL = 100
# for generation
SAMPLE_SIZE = 64
SAMPLE = torch.randn((SAMPLE_SIZE, Z_DIM))

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
        sample = sample.view(SAMPLE_SIZE, *IMAGE_SIZE)
        torchvision.utils.save_image(sample, filename)

def save_training_progress(net, g_optimizer, d_optimizer, epoch, target_dir):
    torch.save({
        'epoch' : epoch + 1,
        'net_state_dict' : net.state_dict(),
        'g_optimizer_state_dict' : g_optimizer.state_dict(),
        'd_optimizer_state_dict' : d_optimizer.state_dict()
    }, target_dir)

generate(SAMPLE, GENERATED_DIRPATH + 'gan_sample_0.png')

# actual training script
for epoch in range(next_epoch, EPOCH):
    # for loss calculation
    d_real_loss = 0.0
    d_fake_loss = 0.0
    d_loss = 0.0
    g_loss = 0.0
    d_n = 0
    g_n = 0

    net.train()
    for train_data in tqdm(trainloader, desc=f"Epoch {epoch + 1}/{EPOCH}"):
        inputs = train_data[0].to(device)
        inputs = inputs.view(-1, FLATTEN_SIZE)

        # 1. train the discriminator
        # zeroes gradients
        d_optimizer.zero_grad()
        g_optimizer.zero_grad()
        # a. train on real images
        real_outputs = net.module.discriminate(inputs) if multigpu else net.discriminate(inputs)
        # the last batch might not have BATCH_SIZE number of data
        real_labels = torch.ones(inputs.shape[0], 1).to(device)
        real_loss = criterion(real_outputs, real_labels)
        d_real_loss += real_loss.item()
        # b. train on fake images
        # generate fake images from samples
        samples = torch.randn((inputs.shape[0], Z_DIM)).to(device)
        fake_inputs = net.module.generate(samples) if multigpu else net.generate(samples)
        fake_outputs = net.module.discriminate(fake_inputs.detach()) if multigpu else net.discriminate(fake_inputs.detach())
        # the last batch might not have BATCH_SIZE number of data
        fake_labels = torch.zeros((inputs.shape[0], 1)).to(device)
        fake_loss = criterion(fake_outputs, fake_labels)
        d_fake_loss += fake_loss.item()
        # compute total loss
        total_loss = (real_loss + fake_loss)
        d_loss += total_loss.item()
        d_n += len(inputs)
        # compute gradients
        total_loss.backward()
        # update discriminator weights
        d_optimizer.step()

        # 2. train the generator
        # zeroes gradients
        g_optimizer.zero_grad()
        d_optimizer.zero_grad()
        # generate samples
        samples = torch.randn((inputs.shape[0], Z_DIM)).to(device)
        # generate images based on the samples and discriminate them
        outputs = net(samples)
        # the last batch might not have BATCH_SIZE number of data
        labels = torch.ones((inputs.shape[0], 1)).to(device)
        # compute loss
        loss = criterion(outputs, labels)
        g_loss += loss.item()
        g_n += len(inputs)
        loss.backward()
        # update generator weights
        g_optimizer.step()
    
    generate(SAMPLE, GENERATED_DIRPATH + f'gan_sample_{epoch + 1}.png')

    if (epoch + 1) % SAVE_INTERVAL == 0:
        save_training_progress(net, g_optimizer, d_optimizer, epoch, MODEL_DIRPATH + f"gan-model-epoch{epoch + 1}.pth")
    
    print(f"""
    Discriminator loss on real images: {d_real_loss/d_n}
    Discriminator loss on fake images: {d_fake_loss/d_n}
    Discriminator loss: {d_loss/d_n}
    Generator loss: {g_loss/g_n}
    """, flush=True)

save_training_progress(net, g_optimizer, d_optimizer, epoch, MODEL_DIRPATH + f"gan-model-epoch{EPOCH}.pth")