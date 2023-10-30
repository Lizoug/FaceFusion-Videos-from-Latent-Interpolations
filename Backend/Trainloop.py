import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader
from Dataset import dataset 
from Critic import Critic
from Generator import Generator

# Hyperparameters
lr = 0.0002
beta1 = 0.5
num_epochs = 10
batch_size = 32
nz = 100
ngf = 64
ndf = 64

# Set up DataLoader
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Initialize networks
critic = Critic(nc=3, ndf=ndf)
generator = Generator(nz=nz, ngf=ngf, nc=3)

# Set up Loss function (Wasserstein loss)
def critic_loss(real_output, fake_output):
    return torch.mean(real_output) - torch.mean(fake_output)

# Set up the generator loss
def generator_loss(fake_output):
    return -torch.mean(fake_output)

# Set up Optimizers
# By setting this value close to 1, the optimizer gives more 
# weight to past squared gradients, making the estimate more stable over time.
optimizerD = torch.optim.Adam(critic.parameters(), lr=lr, betas=(beta1, 0.999))
optimizerG = torch.optim.Adam(generator.parameters(), lr=lr, betas=(beta1, 0.999))
