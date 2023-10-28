import torch.nn as nn

class Critic(nn.Module):
    def __init__(self, nc, ndf):
        super(Critic, self).__init__()
        
        # Defining individual layers

        # Input: nc x 64 x 64
        self.conv1 = nn.Conv2d(nc, 
                               ndf,
                               kernel_size=4,
                               stride=2,
                               padding=1,
                               bias=False
                               )
        self.lr1 = nn.LeakyReLU(0.2, inplace=True)

        # ndf x 32 x 32
        self.conv2 = nn.Conv2d(ndf, 
                               ndf*2, 
                               kernel_size=4, 
                               stride=2, 
                               padding=1, 
                               bias=False)
        self.bn2 = nn.BatchNorm2d(ndf*2)
        self.lr2 = nn.LeakyReLU(0.2, inplace=True)

        # (ndf*2) x 16 x 16
        self.conv3 = nn.Conv2d(ndf*2, 
                               ndf*4, 
                               kernel_size=4, 
                               stride=2, 
                               padding=1, 
                               bias=False)
        self.bn3 = nn.BatchNorm2d(ndf*4)
        self.lr3 = nn.LeakyReLU(0.2, inplace=True)

        # (ndf*4) x 8 x 8
        self.conv4 = nn.Conv2d(ndf*4, 
                               ndf*8, 
                               kernel_size=4, 
                               stride=2, 
                               padding=1, 
                               bias=False)
        self.bn4 = nn.BatchNorm2d(ndf*8)
        self.lr4 = nn.LeakyReLU(0.2, inplace=True)

        # (ndf*8) x 4 x 4
        self.fc = nn.Linear(ndf*8*4*4, 1)
        
    def forward(self, x):
        x = self.lr1(self.conv1(x))
        x = self.lr2(self.bn2(self.conv2(x)))
        x = self.lr3(self.bn3(self.conv3(x)))
        x = self.lr4(self.bn4(self.conv4(x)))
        
        # Flatten
        x = x.view(-1, ndf*8*4*4)
        
        x = self.fc(x)
        return x
        
ndf = 64
critic = Critic(nc=3, ndf=ndf)