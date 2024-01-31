import torch
import torch.nn as nn


class Generator(nn.Module):
    """ nz: (size of latent vector z): This is the dimension of the random
        vector that you provide as input to the generator to produce an image.
        It's a crucial parameter since it determines how many different "kinds"
        of images the generator can potentially produce. A common value for
        nz in many DCGAN implementations is 100.

        nc: Number of channels in the training images.
 
        ngf: (Generator Feature Map Size): This parameter determines the depth
        of the feature maps that pass through the generator. It affects the
        complexity and size of the generator model. A typical value
        for ngf is 64.
        """
    def __init__(self, nz, nc, ngf) -> None:

        # used to call the __init__ method of the parent class (nn.Module in
        # this case). This is necessary to correctly initialize the base class
        # and to make sure all functionalities of the nn.Module class are
        # available in our Generator subclass.
        super(Generator, self).__init__()

        # Linear layer to transform the input latent vector
        # into a suitable shape
        self.linear = nn.Linear(nz, 4*4*512)

        # Defining individual layers
        # First transposed convolutional layer: Upscales from 4x4 to 8x8
        self.convT1 = nn.ConvTranspose2d(in_channels=512,
                                          out_channels=ngf*8,
                                          kernel_size=4,
                                          stride=2,
                                          padding=1,
                                          bias=False)
        self.bn1 = nn.BatchNorm2d(ngf*8)

        # Second transposed convolutional layer: Upscales from 8x8 to 16x16
        self.convT2 = nn.ConvTranspose2d(in_channels=ngf*8,
                                          out_channels=ngf*4,
                                          kernel_size=4,
                                          stride=2,
                                          padding=1,
                                          bias=False)
        self.bn2 = nn.BatchNorm2d(ngf*4)

        # Third transposed convolutional layer: Upscales from 16x16 to 32x32
        self.convT3 = nn.ConvTranspose2d(in_channels=ngf*4,
                                          out_channels=ngf*2,
                                          kernel_size=4,
                                          stride=2,
                                          padding=1,
                                          bias=False)
        self.bn3 = nn.BatchNorm2d(ngf*2)

        # Fourth transposed convolutional layer: Upscales from 32x32 to 64x64
        self.convT4 = nn.ConvTranspose2d(in_channels=ngf*2,
                                          out_channels=ngf,
                                          kernel_size=4,
                                          stride=2,
                                          padding=1,
                                          bias=False)
        self.bn4 = nn.BatchNorm2d(ngf)

        # Fifth transposed convolutional layer: Upscales from 64x64 to 128x128
        self.convT5 = nn.ConvTranspose2d(in_channels=ngf,
                                         out_channels=ngf//2,
                                         kernel_size=4,
                                         stride=2,
                                         padding=1,
                                         bias=False)
        self.bn5 = nn.BatchNorm2d(ngf//2)

        # Final convolution layer: Adjusts the channel size to match 
        # the target image channels
        self.conv = nn.Conv2d(ngf//2, nc, 3, 1, 1)

    def forward(self, x):
        x = self.linear(x)
        x = torch.reshape(x, (x.shape[0], 512, 4, 4))
        x = self.convT1(x)
        # print(x.shape)
        x = self.bn1(x)
        x = nn.ReLU(True)(x)

        x = self.convT2(x)
        # print(x.shape)
        x = self.bn2(x)
        x = nn.ReLU(True)(x)

        x = self.convT3(x)
        # print(x.shape)
        x = self.bn3(x)
        x = nn.ReLU(True)(x)

        x = self.convT4(x)
        # print(x.shape)
        x = self.bn4(x)
        x = nn.ReLU(True)(x)

        x = self.convT5(x)
        # print(x.shape)
        x = self.bn5(x)
        x = nn.ReLU(True)(x)

        x = self.conv(x)
        # print(x.shape)
        x = nn.Tanh()(x)  # Output layer with Tanh activation
        return x


nz = 100
ngf = 128
generator = Generator(nz=nz, ngf=ngf, nc=3)

# Random vector
random_vector = torch.randn(1, nz)
generated_image = generator(random_vector)
