import torch
import torch.nn as nn
## import torch.nn.functional as F
## import torch.optim as optim
## from torchvision import datasets, transforms, models
## from torch.utils.data import DataLoader
## from torchsummary import summary
## from torchvision.models import efficientnet_b2, efficientnet_b1
from torchvision.models import efficientnet_b1
from kcn import KANLinear

## import gc
## import matplotlib.pyplot as plt
## import math

## import nni
## from torch.utils.data import DataLoader
## from torchvision import datasets
## from torchvision.transforms import ToTensor

class EfficientNetB1_KAN(nn.Module):

    def __init__(self, num_classes=1000, params=None, pretrained=True):
        super(EfficientNetB1_KAN, self).__init__()
        # Load pre-trained EfficientNet-B1 model
        efficientnet = efficientnet_b1(pretrained=pretrained)

        self.backbone = nn.Sequential(*list(efficientnet.children())[:-1])

        # Remove the final MLP layers (usually a Linear layer)
        in_features = efficientnet.classifier[1].in_features

        num_features = 512

        if params is None:
            self.kan_layer1 = KANLinear(num_features, num_features)
            self.kan_layer2 = KANLinear(num_features, num_features)

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
        self.kan_layer2 = KANLinear(512, in_features)

        self.classifier = efficientnet.classifier


    def forward(self, x):
        x.requires_grad = True
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = self.kan_layer1(x)  
        x = self.kan_layer2(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x


