import torch, gc, math, nni
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt

from torchvision import datasets, transforms, models
from torchvision.models import vgg16
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torchsummary import summary

from kcn import KANLinear


class VGG16_KAN_Mid(nn.Module):

    def __init__(self, num_classes=1000, params=None, pretrained=True):
        super(VGG16_KAN_Mid, self).__init__()
        # Load pre-trained EfficientNet-B1 model
        vgg = vgg16(pretrained=pretrained)

        # print('Normal VGG model:')
        # print(vgg)

        # Step 2: Freeze the backbone's parameters (optional)
        for param in vgg.parameters():
            param.requires_grad = False

        # Step 3: Extract the feature extraction part (everything except the final FC layer)
        self.backbone = nn.Sequential(*list(vgg.children())[:-1])

        # print('Backbone of VGG model:')
        # print(self.backbone)
        # Remove the final MLP layers (usually a Linear layer)
        in_features = vgg.classifier[0].in_features

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
        self.classifier = vgg.classifier

        # print('Classifier of VGG model:')
        # print(self.classifier)

    def forward(self, x):
        x.requires_grad = True
        x = self.backbone(x)
        x = torch.flatten(x, 1)
        x = self.kan_layer1(x)  
        x = self.kan_layer2(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        
        return x


