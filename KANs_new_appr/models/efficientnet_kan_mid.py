import torch
import torch.nn as nn
from torchvision.models import efficientnet_b1
from kcn import KANLinear


class EfficientNetB1_KAN_Mid(nn.Module):

    def __init__(self, num_classes=4, params=None, pretrained=True, num_features=32):
        super(EfficientNetB1_KAN_Mid, self).__init__()
        # Load pre-trained EfficientNet-B1 model
        efficientnet = efficientnet_b1(pretrained=pretrained)

        self.backbone = nn.Sequential(*list(efficientnet.children())[:-1])

        # Remove the final MLP layers (usually a Linear layer)
        in_features = efficientnet.classifier[1].in_features

        kan_layers = []
        if params is None:
            kan_layers.append(KANLinear(in_features, num_features))
            kan_layers.append(KANLinear(num_features, num_features))

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
        self.classifier = efficientnet.classifier


    def forward(self, x):
        x.requires_grad = True
        x = self.backbone(x)
        x = self.feature_extractor(x)
        # # x = torch.flatten(x, 1)
        # for layer in self.kan_layers:
        #     layer.cuda()
        #     x = layer(x)
        # # x = torch.flatten(x, 1)
        x = self.classifier(x)

        return x


