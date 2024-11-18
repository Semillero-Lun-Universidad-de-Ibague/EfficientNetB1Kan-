import torch
import torch.nn as nn

from kcn import KANLinear


class KAN_Model(nn.Module):

    def __init__(self, num_classes=4, params=None, num_features=32, size_img=128):
        super(KAN_Model, self).__init__()

        kan_layers = []
        if params is None:
            kan_layers.append(KANLinear(3*size_img*size_img, num_features))
            kan_layers.append(KANLinear(num_features, num_features))

        else:
            kan_layers.append(KANLinear(3*size_img*size_img, num_features,
                                        grid_size=params['grid_size'],
                                        spline_order=params['spline_order'],
                                        scale_noise=params['scale_noise'],
                                        scale_base=params['scale_base'],
                                        scale_spline=params['scale_spline']
                                        ))
            for layer in list(range(params['num_layers'] - 1)):
                kan_layers.append(KANLinear(num_features, num_features,
                                        grid_size=params['grid_size'],
                                        spline_order=params['spline_order'],
                                        scale_noise=params['scale_noise'],
                                        scale_base=params['scale_base'],
                                        scale_spline=params['scale_spline']
                                        ))

        
        self.flatten = nn.Flatten()
        self.feature_extractor = nn.Sequential(*kan_layers)
        # self.kan_layers = nn.ModuleList(kan_layers)   
        self.classifier = nn.Linear(num_features, num_classes)  

    def forward(self, x):
        x.requires_grad = True
        # x = torch.flatten(x, 1)
        # print(x.shape)
        x = self.feature_extractor(x)
        # for layer in self.kan_layers:
        #     x = layer(x)
        x = self.classifier(x)
        
        return x


