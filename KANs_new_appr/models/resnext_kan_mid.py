import torch
import torch.nn as nn
from torchvision.models import resnext50_32x4d
from kcn_try import KANLinear


class ResNext_KAN_Mid(nn.Module):

    def __init__(self, num_classes=4, params=None, pretrained=True, num_features=32):
        super(ResNext_KAN_Mid, self).__init__()
        # Load pre-trained EfficientNet-B1 model
        self.resnext = resnext50_32x4d(pretrained=pretrained)

        # self.backbone = nn.Sequential(*list(resnext.children())[:-1])

        # Remove the final MLP layers (usually a Linear layer)
        in_features = self.resnext.fc.in_features

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

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.feature_extractor = nn.Sequential(*kan_layers)
        # self.kan_layers = nn.ModuleList(kan_layers)
        self.fc = self.resnext.fc
        print(self.resnext.fc)


    def forward(self, x):
        x.requires_grad = True
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
        # x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        # for layer in self.kan_layers:
        #     layer.cuda()
        #     x = layer(x)
        # x = torch.flatten(x, 1)
        x = self.fc(x)

        return x