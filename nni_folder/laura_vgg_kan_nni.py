from torchvision.models import vgg16
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
from torchsummary import summary
import gc
import matplotlib.pyplot as plt
import math

from kcn_nni import KANLinear

import nni
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

class VGG16_KAN(nn.Module):

    def __init__(self, num_classes=1000, params=None, pretrained=True):
        super(VGG16_KAN, self).__init__()
        # Load pre-trained EfficientNet-B1 model
        self.vgg = vgg16(pretrained=pretrained)

        # Remove the final MLP layers (usually a Linear layer)
        in_features = self.vgg.classifier[0].in_features

        num_features = 512

        if params is None:
            self.kan_layer1 = KANLinear(num_features, num_features)
            self.kan_layer2 = KANLinear(num_features, num_classes)

        else:
            self.kan_layer1 = KANLinear(num_features, num_classes,
                                        grid_size=params['grid_size'],
                                        spline_order=params['spline_order'],
                                        scale_noise=params['scale_noise'],
                                        scale_base=params['scale_base'],
                                        scale_spline=params['scale_spline']
                                        )
            self.kan_layer2 = KANLinear(num_classes, num_features,
                                        grid_size=params['grid_size'],
                                        spline_order=params['spline_order'],
                                        scale_noise=params['scale_noise'],
                                        scale_base=params['scale_base'],
                                        scale_spline=params['scale_spline'])

        # Define KAN layers to replace MLP
        self.kan_layer1 = KANLinear(in_features, 512)
        self.kan_layer2 = KANLinear(512, num_classes)

    def forward(self, x):
        x = self.vgg.features(x)
        x = self.vgg.avgpool(x)
        x = torch.flatten(x, 1)
        # Forward pass through KAN layers
        x = self.kan_layer1(x)  
        x = self.kan_layer2(x)
        return x


