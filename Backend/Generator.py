import torch
import torch.nn as nn
import torch.optim as optim

class Generator(nn.Module):
    """ nz: (size of latent vector z): This is the dimension of the random 
        vector that you provide as input to the generator to produce an image. 
        It's a crucial parameter since it determines how many different "kinds" 
        of images the generator can potentially produce. A common value for 
        nz in many DCGAN implementations is 100.
        
        nc: Number of channels in the training images.
        
        ngf: (Generator Feature Map Size): This parameter determines the depth 
        of the feature maps that pass through the generator. It affects the 
        complexity and size of the generator model. A typical value for ngf is 64.
        """
    def __init__(self, nz, nc, ngf) -> None:

        # used to call the __init__ method of the parent class (nn.Module in this 
        # case). This is necessary to correctly initialize the base class and to 
        # make sure all functionalities of the nn.Module class are available in 
        # our Generator subclass.
        super(Generator, self).__init__()
        

       # Defining individual layers
        
        # Input 512 x 4 x 4 
        self.convT1 = nn.ConvTranspose2d(in_channels=512, 
                                          out_channels=ngf*4, 
                                          kernel_size=4, 
                                          stride=2, 
                                          padding=1, 
                                          bias=False)
        self.bn1 = nn.BatchNorm2d(ngf*4)
        
        # (ngf*4) x 8 x 8 -> in this example 256x8x8, because im using ngf=64
        self.convT2 = nn.ConvTranspose2d(in_channels=ngf*4,
                                          out_channels=ngf*2,
                                          kernel_size=4,
                                          stride=2,
                                          padding=1,
                                          bias=False)
        self.bn2 = nn.BatchNorm2d(ngf*2)
        
        # (ngf*2) x 16 x 16 -> 128x16x16
        self.convT3 = nn.ConvTranspose2d(in_channels=ngf*2, 
                                          out_channels=ngf,
                                          kernel_size=4,
                                          stride=2,
                                          padding=1,
                                          bias=False)
        self.bn3 = nn.BatchNorm2d(ngf)
        
        # ngf x 32 x 32 -> 64x32x32
        self.convT4 = nn.ConvTranspose2d(in_channels=ngf,
                                          out_channels=nc,
                                          kernel_size=4,
                                          stride=2,
                                          padding=1,
                                          bias=False)
        
        # Final convolution layer
        self.conv = nn.Conv2d(nc, nc, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(nc)


        self.linear = nn.Linear(100, 4*4*512)


    def forward(self, x):
        x = self.linear(x)
        x = torch.reshape(x, (x.shape[0], 512, 4, 4))
        x = self.convT1(x)
        #print(x.shape)
        x = self.bn1(x)
        x = nn.ReLU(True)(x)

        x = self.convT2(x)
        #print(x.shape)
        x = self.bn2(x)
        x = nn.ReLU(True)(x)

        x = self.convT3(x)
        #print(x.shape)
        x = self.bn3(x)
        x = nn.ReLU(True)(x)

        x = self.convT4(x)
        #print(x.shape)
        x = self.bn4(x)
        x = nn.ReLU(True)(x)

        x = self.conv(x)
        #print(x.shape)
        x = nn.Tanh()(x)
    
        return x



nz = 100
generator = Generator(nz=nz, ngf=128, nc=3)

# Random vector
random_vector = torch.randn(1, nz)


generated_image = generator(random_vector)

# check the size
#print(generated_image.size())  # must be torch.Size([1, 3, 64, 64])
