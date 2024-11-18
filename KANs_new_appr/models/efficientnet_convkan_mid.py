import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import efficientnet_b1
from convkan import ConvKAN, LayerNorm2D


class EfficientNetB1_ConvKAN_Mid(nn.Module):

    def __init__(self, num_classes=4, params=None, pretrained=True, num_features=32):
        super(EfficientNetB1_ConvKAN_Mid, self).__init__()
        # Load pre-trained EfficientNet-B1 model
        efficientnet = efficientnet_b1(pretrained=pretrained)

        self.backbone = nn.Sequential(*list(efficientnet.children())[:-1])

        # Remove the final MLP layers (usually a Linear layer)
        in_features = efficientnet.classifier[1].in_features

        kan_layers = []
        if params is None:
            kan_layers = [ConvKAN(in_features, num_features), LayerNorm2D(num_features), ConvKAN(num_features, in_features), LayerNorm2D(in_features)]
            # kan_layers = [ConvKAN(in_features, num_features), ConvKAN(num_features, in_features)]
            # self.lns = [LayerNorm2D(num_features), LayerNorm2D(in_features)]

        else:
            kan_layers.append(ConvKAN(in_features, num_features,
                                        padding=params['padding'],
                                        kernel_size=params['kernel_size'],
                                        stride=params['stride']
                                        ))
            kan_layers.append(LayerNorm2D(num_features))
            for layer in list(range(params['num_layers'] - 1)):
                if layer == list(range(params['num_layers'] - 1))[-1]:
                    output_size = in_features
                else:
                    output_size = num_features
                kan_layers.append(ConvKAN(num_features, output_size,
                                        padding=params['padding'],
                                        kernel_size=params['kernel_size'],
                                        stride=params['stride']
                                        ))
                kan_layers.append(LayerNorm2D(output_size))
                
        self.feature_extractor = nn.Sequential(*kan_layers)
        # self.kan_layers = nn.ModuleList(kan_layers)
        # self.ln = LayerNorm2D(num_features)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.classifier = efficientnet.classifier


    def forward(self, x):
        x.requires_grad = True
        x = self.backbone(x)
        x = self.feature_extractor(x)
        # for i, layer in enumerate(self.kan_layers):
        #     print(x.shape)
        #     x = layer(x)
        #     # self.lns[i].cuda()
        #     x = self.lns[i](x)
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.classifier(x)

        return x


