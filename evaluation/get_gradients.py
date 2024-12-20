import torch, sys
import torch.nn as nn
from torchvision.models import vgg16, efficientnet_b1

sys.path.append('..')
from models.laura_vgg_kan_try import VGG16_KAN
from models.laura_resnext_kan import ResNext_KAN
from models.efficientnet_kan_try import EfficientNetB1_KAN

params = {
        'grid_size': 32,
        'spline_order': 2,
        'scale_noise': 0.66,
        'scale_base': 0.6,
        'scale_spline': 0.62
    }

# Load a pretrained VGG16 model
# model = vgg16(pretrained=True)
model = EfficientNetB1_KAN(4, params)
# model = efficientnet_b1(pretrained=True)

# Set the model to evaluation mode
model.eval()

# Store gradients here
gradients = []

# Function to capture gradients
def save_gradient(grad):
    gradients.append(grad)

# Register a hook on the last layer (classifier's last linear layer in VGG16)
# layer = model.features[-1]
layer = model.kan_layer2
hook = layer.register_backward_hook(lambda module, grad_input, grad_output: save_gradient(grad_output[0]))

# Create a dummy input (e.g., an image of the appropriate size for VGG16)
input_tensor = torch.randn(1, 3, 224, 224, requires_grad=True)

# Perform a forward pass
output = model(input_tensor)

# Choose a specific class to backpropagate
target_class = output[0, 1]  # Example: class index 281

# Perform backward pass to compute gradients
target_class.backward()
print(model)

# Access the gradients
print("Extracted Gradients:", gradients[0])  # (1, 1000) for VGG16's output layer
print("Extracted Gradients Shape:", gradients[0].shape)  # (1, 1000) for VGG16's output layer
print("Extracted Gradients Length:", len(gradients[0].shape))  # (1, 1000) for VGG16's output layer

# Don't forget to remove the hook when done
hook.remove()
