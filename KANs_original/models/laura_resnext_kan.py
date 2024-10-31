import torch, gc, math, nni
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

from torchvision import datasets, transforms, models
from torchvision.models import resnext50_32x4d
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchsummary import summary

from kcn import KANLinear


class ResNext_KAN(nn.Module):

    def __init__(self, num_classes=1000, params=None, pretrained=True):
        super(ResNext_KAN, self).__init__()
        # Load pre-trained EfficientNet-B1 model
        self.resnext = resnext50_32x4d(pretrained=pretrained)

        # Remove the final MLP layers (usually a Linear layer)
        in_features = self.resnext.fc.in_features

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
        x = self.resnext.conv1(x)
        x = self.resnext.bn1(x)
        x = self.resnext.relu(x)
        x = self.resnext.maxpool(x)
        x = self.resnext.layer1(x)
        x = self.resnext.layer2(x)
        x = self.resnext.layer3(x)
        x = self.resnext.layer4(x)
        x = self.resnext.avgpool(x)
        x = torch.flatten(x, 1)
        # Forward pass through KAN layers
        x = self.kan_layer1(x)  
        x = self.kan_layer2(x)
        return x

