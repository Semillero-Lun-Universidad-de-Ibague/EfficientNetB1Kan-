import torch, gc, math
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

    def __init__(self, num_classes=1000, params=None, pretrained=True, num_features=32):
        super(ResNext_KAN, self).__init__()
        # Load pre-trained EfficientNet-B1 model
        self.resnext = resnext50_32x4d(pretrained=pretrained)

        # Remove the final MLP layers (usually a Linear layer)
        in_features = self.resnext.fc.in_features

        kan_layers = []
        if params is None:
            kan_layers.append(KANLinear(in_features, num_features))
            kan_layers.append(KANLinear(num_features, in_features))

        else:
            kan_layers.append(KANLinear(in_features, num_features,
                                        grid_size=params['grid_size'],
                                        spline_order=params['spline_order'],
                                        scale_noise=params['scale_noise'],
                                        scale_base=params['scale_base'],
                                        scale_spline=params['scale_spline']
                                        ))
            for layer in list(range(params['num_layers'] - 1)):
                if layer == list(range(params['num_layers'] - 1))[-1]:
                    output_size = in_features
                else:
                    output_size = num_features
                kan_layers.append(KANLinear(num_features, output_size,
                                        grid_size=params['grid_size'],
                                        spline_order=params['spline_order'],
                                        scale_noise=params['scale_noise'],
                                        scale_base=params['scale_base'],
                                        scale_spline=params['scale_spline']
                                        ))

        self.feature_extractor = nn.Sequential(*kan_layers)
        # self.kan_layers = nn.ModuleList(kan_layers)

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
        x = self.feature_extractor(x)
        # for layer in self.kan_layers:
        #     x = layer(x)
        return x

