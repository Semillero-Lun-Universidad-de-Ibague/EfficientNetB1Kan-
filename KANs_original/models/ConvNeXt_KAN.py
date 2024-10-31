import torch
import torch.nn as nn

from torchvision import models
from kcn import KANLinear


class ConvNeXtKAN(nn.Module):
    def __init__(self, params=None):
        super(ConvNeXtKAN, self).__init__()
        # Load pre-trained ConvNeXt model
        self.convnext = models.convnext_tiny(pretrained=True)

        # Freeze ConvNeXt layers (if required)
        for param in self.convnext.parameters():
            param.requires_grad = False

        # Modify the classifier part of ConvNeXt
        num_features = self.convnext.classifier[2].in_features
        self.convnext.classifier = nn.Identity()

        if params is None:
            self.kan1 = KANLinear(num_features, 256)
            self.kan2 = KANLinear(256, 4)

        else:
            self.kan1 = KANLinear(num_features, 256,
                                  grid_size=params['grid_size'],
                                  spline_order=params['spline_order'],
                                  scale_noise=params['scale_noise'],
                                  scale_base=params['scale_base'],
                                          scale_spline=params['scale_spline']
                                  )
            self.kan2 = KANLinear(256, 4,
                                  grid_size=params['grid_size'],
                                  spline_order=params['spline_order'],
                                  scale_noise=params['scale_noise'],
                                  scale_base=params['scale_base'],
                                  scale_spline=params['scale_spline']
                                  )

    def forward(self, x):
        x = self.convnext(x)
        x = x.view(x.size(0), -1)  # Flatten the tensor
        x = self.kan1(x)
        x = self.kan2(x)
        return x
