from config import cfg
import torch
import torch.nn as nn
import torch.nn.functional as F
from misc.layer import Conv2d, FC
from torchvision import models
from misc.utils import *
import pdb

class RAZ_loc(nn.Module):
    def __init__(self):
        super(RAZ_loc, self).__init__()
        vgg = models.vgg16(pretrained=True)

        features = list(vgg.features.children())
        self.features4 = nn.Sequential(*features[0:23])

        self.de_pred = nn.Sequential(Conv2d( 512, 512, 3, same_padding=True, NL='relu', dilation=2),
                                    Conv2d( 512, 512, 3, same_padding=True, NL='relu', dilation=2),
                                    Conv2d( 512, 512, 3, same_padding=True, NL='relu', dilation=2),
                                    nn.ConvTranspose2d(512,256,4,stride=2,padding=1,output_padding=0,bias=True),
                                    nn.ReLU(),
                                    Conv2d( 256, 256, 3, same_padding=True, NL='relu', dilation=2),
                                    nn.ConvTranspose2d(256,128,4,stride=2,padding=1,output_padding=0,bias=True),
                                    nn.ReLU(),
                                    Conv2d( 128, 128, 3, same_padding=True, NL='relu', dilation=2),
                                    nn.ConvTranspose2d(128,64,4,stride=2,padding=1,output_padding=0,bias=True),
                                    nn.ReLU(),
                                    Conv2d( 64, 64, 3, same_padding=True, NL='relu', dilation=2),
                                    Conv2d(64, 2, 1, same_padding=True, NL='relu'))



    def forward(self, x):
        x = self.features4(x)       
        x = self.de_pred(x)
        

        return x