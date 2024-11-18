import torch
import torch.nn as nn
from torchvision.models import efficientnet_b1
from kcn import KANLinear


class EfficientNetB1_KAN(nn.Module):

    def __init__(self, num_classes=1000, params=None, pretrained=True, num_features=32):
        super(EfficientNetB1_KAN, self).__init__()
        # Load pre-trained EfficientNet-B1 model
        self.efficientnet = efficientnet_b1(pretrained=pretrained)

        # Remove the final MLP layers (usually a Linear layer)
        in_features = self.efficientnet.classifier[1].in_features

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
        x = self.efficientnet.features(x)
        x = self.efficientnet.avgpool(x)
        x = self.feature_extractor(x)
        # for layer in self.kan_layers:
        #     # layer.cuda()
        #     x = layer(x)
        return x


