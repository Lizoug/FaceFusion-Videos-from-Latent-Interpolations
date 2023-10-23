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
        
        # stack different layers sequentially
        self.main = nn.Sequential(
            # nz is the input 
            nn.ConvTranspose2d(in_channels=nz, 
                               out_channels=ngf*8, 
                               kernel_size=4, 
                               stride=1, 
                               padding=0, 
                               bias=False)

        )