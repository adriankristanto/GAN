import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from tqdm import tqdm
import os
import models.DCGAN as DCGAN

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
G = DCGAN.Generator(
    channels=[Z_DIM, 256, 128, 64, 1],
    kernels=[None, 7, 5, 4, 4],
    strides=[None, 1, 1, 2, 2],
    paddings=[None, 0, 2, 1, 1],
    batch_norm=True,
    internal_activation=nn.ReLU(),
    output_activation=nn.Tanh()
)
# create the discrimintor D
D = DCGAN.Discriminator(
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

# 3. define the loss function
criterion = nn.BCELoss(reduction='sum')

# 4. define the optimisers
# a. the optimiser for the generator
g_lr = 0.0002
g_optimizer = optim.Adam(G.parameters(), lr=g_lr, betas=(0.5, 0.999))
# b. the optimiser for the discriminator
d_lr = 0.0002
d_optimizer = optim.Adam(D.parameters(), lr=d_lr, betas=(0.5, 0.999))

# 5. train the model
MODEL_DIRPATH = os.path.dirname(os.path.realpath(__file__)) + '/../saved_models/'
GENERATED_DIRPATH = os.path.dirname(os.path.realpath(__file__)) + '/../generated_images/'
CONTINUE_TRAIN = False
CONTINUE_TRAIN_NAME = MODEL_DIRPATH + 'gan-model-epoch10.pth'
EPOCH = 600
SAVE_INTERVAL = 50
# for generation
SAMPLE_SIZE = 64
# the generator accepts input (100, 1, 1)
SAMPLE = torch.randn((SAMPLE_SIZE, Z_DIM, 1, 1))

IMAGE_SIZE = (1, 28, 28)

next_epoch = 0
if CONTINUE_TRAIN:
    checkpoint = torch.load(CONTINUE_TRAIN_NAME)
    net.load_state_dict(checkpoint.get('net_state_dict'))
    g_optimizer.load_state_dict(checkpoint.get('g_optimizer_state_dict'))
    d_optimizer.load_state_dict(checkpoint.get('d_optimizer_state_dict'))
    next_epoch = checkpoint.get('epoch')

def generate(sample, filename):
    G.eval()
    with torch.no_grad():
        sample = sample.to(device)
        sample = G(sample)
        torchvision.utils.save_image(sample, filename, pad_value=1)

def save_training_progress(G, D, g_optimizer, d_optimizer, epoch, target_dir):
    torch.save({
        'epoch' : epoch + 1,
        'G_state_dict' : G.state_dict(),
        'D_state_dict' : D.state_dict(),
        'g_optimizer_state_dict' : g_optimizer.state_dict(),
        'd_optimizer_state_dict' : d_optimizer.state_dict()
    }, target_dir)

generate(SAMPLE, GENERATED_DIRPATH + 'dcgan_sample_0.png')

for epoch in range(next_epoch, EPOCH):
    d_real_loss = 0.0
    d_fake_loss = 0.0
    d_loss = 0.0
    g_loss = 0.0
    n = 0

    D.train()
    G.train()
    for train_data in tqdm(trainloader, desc=f"Epoch {epoch + 1}/{EPOCH}"):
        inputs = train_data[0].to(device)

        # 1. train the discriminator
        # zeroes gradients
        d_optimizer.zero_grad()
        g_optimizer.zero_grad()
        # a. train on real images
        real_outputs = D(inputs)
        real_labels = torch.ones(inputs.shape[0], 1).to(device)
        real_loss = criterion(real_outputs, real_labels)
        d_real_loss += real_loss.item()
        # b. train on fake images
        samples = torch.randn((inputs.shape[0], Z_DIM, 1, 1)).to(device)
        fake_outputs = D(G(samples).detach())
        fake_labels = torch.zeros((inputs.shape[0], 1)).to(device)
        fake_loss = criterion(fake_outputs, fake_labels)
        d_fake_loss += fake_loss.item()
        # compute total loss
        total_loss = (real_loss + fake_loss)
        d_loss += total_loss.item()
        # compute gradients
        total_loss.backward()
        # update discriminator weights
        d_optimizer.step()

        # 2. train the generator
        g_optimizer.zero_grad()
        d_optimizer.zero_grad()
        # generate samples
        samples = torch.randn((inputs.shape[0], Z_DIM, 1, 1)).to(device)
        # generate images based on the samples and discriminate them
        outputs = D(G(samples))
        labels = torch.ones((inputs.shape[0], 1)).to(device)
        # compute loss
        loss = criterion(outputs, labels)
        g_loss += loss.item()
        # compute gradients
        loss.backward()
        # update generator weights
        g_optimizer.step()

        n += len(inputs)

    generate(SAMPLE, GENERATED_DIRPATH + f'dcgan_sample_{epoch + 1}.png')

    if (epoch + 1) % SAVE_INTERVAL == 0:
        save_training_progress(G, D, g_optimizer, d_optimizer, epoch, MODEL_DIRPATH + f'dcgan-model-epoch{epoch + 1}.pth')

    print(f"""
    Discriminator loss on real images: {d_real_loss/n}
    Discriminator loss on fake images: {d_fake_loss/n}
    Discriminator loss: {d_loss/n}
    Generator loss: {g_loss/n}
    """, flush=True)

save_training_progress(G, D, g_optimizer, d_optimizer, epoch, MODEL_DIRPATH + f'dcgan-model-epoch{epoch + 1}.pth')