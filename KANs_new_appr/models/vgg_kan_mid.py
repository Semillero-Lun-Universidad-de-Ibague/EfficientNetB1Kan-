import torch
import torch.nn as nn
from torchvision.models import vgg16
from kcn import KANLinear


class VGG16_KAN_Mid(nn.Module):

    def __init__(self, num_classes=4, params=None, pretrained=True, num_features=32):
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

        kan_layers = []
        if params is None:
            kan_layers.append(KANLinear(in_features, num_features))
            kan_layers.append(KANLinear(num_features, in_features))
            # kan_layer1 = KANLinear(in_features, num_features)
            # kan_layer2 = KANLinear(num_features, in_features)
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
        self.classifier = vgg.classifier


    def forward(self, x):
        x.requires_grad = True
        x = self.backbone(x)
        x = self.feature_extractor(x)
        # for layer in self.kan_layers:
        #     # layer.cuda()
        #     x = layer(x)
        x = self.classifier(x)
        
        return x