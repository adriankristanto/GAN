import torch
import torch.nn as nn 
import torch.optim as optim
import torchvision 
import torchvision.datasets as datasets 
import torchvision.transforms as transforms
from tqdm import tqdm
import os 
import models.WGANGP as WGANGP

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
# Generator G
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
# Critic D
D = WGANGP.Critic(
    channels=[1, 64, 128, 256, 1],
    kernels=[None, 4, 4, 5, 7],
    strides=[None, 2, 2, 1, 1],
    paddings=[None, 1, 1, 2, 0],
    internal_activation=nn.LeakyReLU(0.2),
)
print(f"""
Generator G:
{G}

Critic D:
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
def WassersteinLoss(prediction, truth, reduction='mean'):
    reduction_func = None
    if reduction == 'mean':
        reduction_func = torch.mean
    elif reduction == 'sum':
        reduction_func = torch.sum
    # since we are minimising the loss function, we multiply it with -1
    return -1 * reduction_func(prediction * truth)

def GradientPenaltyLoss(D, real_samples, fake_samples, reduction='mean'):
    # generate a random number epsilon from uniform distribution [0,1]
    # as the weight for each sample
    # there are real_samples.shape[0] samples in the batch
    epsilon = torch.rand((real_samples.shape[0], 1, 1, 1)).to(device)
    # interpolates real and fake samples
    inputs = (epsilon * real_samples + (1 - epsilon) * fake_samples).requires_grad_(True)
    inputs = inputs.to(device)
    # fed the interpolated samples to the critic
    outputs = D(inputs)
    # compute the gradients of the outputs w.r.t the inputs
    gradients = torch.autograd.grad(
        outputs=outputs,
        inputs=inputs,
        # reference: https://stackoverflow.com/questions/58059268/pytorch-autograd-grad-how-to-write-the-parameters-for-multiple-outputs
        # reference: https://stackoverflow.com/questions/54166206/grad-outputs-in-torch-autograd-grad-crossentropyloss
        # reference: https://discuss.pytorch.org/t/what-does-grad-outputs-do-in-autograd-grad/18014
        grad_outputs=torch.ones_like(outputs).to(device),
        create_graph=True,
        retain_graph=True
    )[0]
    # to easily compute the norm, flatten the gradients of shape (batch_size, channels, heights, widths)
    # for example, torch.Size([1, 1, 28, 28])
    gradients = gradients.view(real_samples.shape[0], -1)
    # calculate the gradients norm
    # and compute the distance between the gradients norm and 1
    # the aim is to satisfy the Lipschitz constraint
    gradient_penalty = (gradients.norm(2, dim=1) - 1) ** 2

    reduction_func = None
    if reduction == 'mean':
        reduction_func = torch.mean
    elif reduction == 'sum':
        reduction_func = torch.sum
    
    return reduction_func(gradient_penalty)

# 4. define the optimisers
# a. the optimiser for the generator
g_lr = 0.0002
g_optimizer = optim.Adam(G.parameters(), lr=g_lr, betas=(0.5, 0.999))
# b. the optimiser for the critic
d_lr = 0.0002
d_optimizer = optim.Adam(D.parameters(), lr=d_lr, betas=(0.5, 0.999))

# 5. train the model
MODEL_DIRPATH = os.path.dirname(os.path.realpath(__file__)) + '/../saved_models/'
GENERATED_DIRPATH = os.path.dirname(os.path.realpath(__file__)) + '/../generated_images/'
CONTINUE_TRAIN = False
CONTINUE_TRAIN_NAME = MODEL_DIRPATH + 'wgangp-model-epoch10.pth'
EPOCH = 400
SAVE_INTERVAL = 20
# for generation
SAMPLE_SIZE = 64
# the generator accepts input (100, 1, 1)
SAMPLE = torch.randn((SAMPLE_SIZE, Z_DIM, 1, 1))

# WGAN-GP global variables
# the recommended ratio used is 5 critic updates to 1 generator update
CRITIC_ITER = 5
# the following is the penalty coefficient
LAMBDA = 10

IMAGE_SIZE = (1, 28, 28)

next_epoch = 0
if CONTINUE_TRAIN:
    checkpoint = torch.load(CONTINUE_TRAIN_NAME)
    G.load_state_dict(checkpoint.get('G_state_dict'))
    D.load_state_dict(checkpoint.get('D_state_dict'))
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
    d_real_wloss = 0.0
    d_fake_wloss = 0.0
    d_loss = 0.0
    g_loss = 0.0
    d_n = 0
    g_n = 0

    D.train()
    G.train()

    # to count how many times the gradient updates have been performed
    # whenever counter % CRITIC_ITER == 0, we update the generator
    # for example, CRITIC_ITER = 5, then
    # counter = 0, update both G & D
    # counter = 1 until 4, update D
    # counter = 5, update both G & D, and so on
    counter = 0
    for train_data in tqdm(trainloader):
        inputs = train_data[0].to(device)
        d_n += len(inputs)

        # 1. train the discriminator
        # zeroes gradients
        d_optimizer.zero_grad()
        g_optimizer.zero_grad()

        # a. train on real images
        real_outputs = D(inputs)
        real_labels = torch.ones(inputs.shape[0], 1, 1, 1).to(device)
        # similar to -1 * reduction_func(real_outputs)
        real_loss = WassersteinLoss(real_outputs, real_labels, reduction='sum')
        d_real_wloss += real_loss

        samples = torch.randn((inputs.shape[0], Z_DIM, 1, 1)).to(device)
        fake_inputs = G(samples)
        # b. train on fake images
        fake_outputs = D(fake_inputs.detach())
        fake_labels = -1 * torch.ones((inputs.shape[0], 1, 1, 1)).to(device)
        # similar to reduction_func(fake_outputs)
        fake_loss = WassersteinLoss(fake_outputs, fake_labels, reduction='sum')
        d_fake_wloss += fake_loss

        # compute total loss
        total_loss = real_loss + fake_loss + LAMBDA * GradientPenaltyLoss(D, inputs, fake_inputs, reduction='sum')
        d_loss += total_loss

        # compute gradients
        total_loss.backward()
        # update parameters
        d_optimizer.step()

        if counter % CRITIC_ITER == 0:
            g_n += len(inputs)
            # 2. train the generator
            g_optimizer.zero_grad()
            d_optimizer.zero_grad()
            
            samples = torch.randn((inputs.shape[0], Z_DIM, 1, 1)).to(device)
            outputs = D(G(samples))
            labels = torch.ones((inputs.shape[0], 1, 1, 1)).to(device)
            # compute loss
            loss = WassersteinLoss(outputs, labels, reduction='sum')
            g_loss += loss
            # compute gradients
            loss.backward()
            # update generator weights
            g_optimizer.step()
        
        counter += 1
    
    generate(SAMPLE, GENERATED_DIRPATH + f'wgangp_sample_{epoch + 1}.png')

    if (epoch + 1) % SAVE_INTERVAL == 0:
        save_training_progress(G, D, g_optimizer, d_optimizer, epoch, MODEL_DIRPATH + f'dcgan-model-epoch{epoch + 1}.pth')

    print(f"""
    Discriminator loss on real images: {d_real_loss/d_n}
    Discriminator loss on fake images: {d_fake_loss/d_n}
    Discriminator loss: {d_loss/d_n}
    Generator loss: {g_loss/g_n}
    """, flush=True)

save_training_progress(G, D, g_optimizer, d_optimizer, epoch, MODEL_DIRPATH + f'dcgan-model-epoch{epoch + 1}.pth')