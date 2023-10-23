import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self,channel_img,features_d):
        super(Discriminator,self).__init__()
        self.disc = nn.Sequential(
            nn.Conv2d(channel_img,features_d,kernel_size=4,stride=2,padding=1),
            nn.LeakyReLU(0.2),
            self._block(features_d,features_d*2,4,2,1),
            self._block(features_d*2,features_d*4,4,2,1),
            self._block(features_d*4,features_d*8,4,2,1),
            nn.Conv2d(features_d*8,1,kernel_size=4,stride=2,padding=0),
        )
            
    def _block(self,in_channels,out_channels,kernel_size,stride,padding):
        return nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False
            ),
            nn.InstanceNorm2d(out_channels,affine = True),
            nn.LeakyReLU(0.2),
        )
    
    def forward(self,x):
        return self.disc(x)

class Generator(nn.Module):
    def  __init__(self,channels_noise,channels_img,features_g):
        super(Generator,self).__init__()
        self.gen = nn.Sequential(
            self._block(channels_noise,features_g*16,4,1,0),
            self._block(features_g*16,features_g*8,4,2,1),
            self._block(features_g*8,features_g*4,4,2,1),
            self._block(features_g*4,features_g*2,4,2,1),
            nn.ConvTranspose2d(features_g*2,channels_img,4,2,1),
            nn.Tanh(),
        )
    
    def _block(self,in_channels,out_channels,kernel_size,stride,padding):
        return nn.Sequential(
            nn.ConvTranspose2d(
                in_channels,
                out_channels,
                kernel_size,
                stride,
                padding,
                bias=False,
            ),
            nn.InstanceNorm2d(out_channels,affine=True),
            nn.ReLU(),
        )
        
    def forward(self,x):
        return self.gen(x)
    
