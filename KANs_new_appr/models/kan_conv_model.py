import torch
from torch import nn
from torchvision import datasets, transforms
from tqdm import tqdm

from convkan import ConvKAN, LayerNorm2D

class ConvKAN_Model(nn.Module):

    def __init__(self, num_classes=4, params=None, num_features=32):
        super(ConvKAN_Model, self).__init__()

        kan_layers = []
        if params is None:
            kan_layers = [ConvKAN(1, num_features), LayerNorm2D(num_features), ConvKAN(num_features, num_features), LayerNorm2D(num_features)]
            # lns = [LayerNorm2D(num_features), LayerNorm2D(num_features)]
        else:
            kan_layers.append(ConvKAN(1, num_features,
                                        padding=params['padding'],
                                        kernel_size=params['kernel_size'],
                                        stride=params['stride']
                                        ))
            kan_layers.append(LayerNorm2D(1))
            # lns.append(LayerNorm2D(1))
            
            for layer in list(range(params['num_layers'] - 1)):
                kan_layers.append(ConvKAN(num_features, num_features,
                                        padding=params['padding'],
                                        kernel_size=params['kernel_size'],
                                        stride=params['stride']
                                        ))
                kan_layers.append(LayerNorm2D(num_features))
                # lns.append(LayerNorm2D(num_features))
                
        # self.kan_layers = nn.ModuleList(kan_layers)
        # self.lns = nn.ModuleList(lns)
        self.feature_extractor = nn.Sequential(*kan_layers)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.classifier = nn.Linear(num_features, num_classes)  


    def forward(self, x):
        x.requires_grad = True
        x = self.feature_extractor(x)
        # for i, layer in enumerate(self.kan_layers):
        #     # print(x.shape)
        #     x = layer(x)  
        #     x = self.lns[i](x)
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.classifier(x)
        
        return x