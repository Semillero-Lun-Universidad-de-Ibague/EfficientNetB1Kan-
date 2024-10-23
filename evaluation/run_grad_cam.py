import torch, cv2, argparse, sys
import numpy as np
import tensorflow as tf
from torchvision import transforms
from torchvision.models import vgg16, resnext50_32x4d, efficientnet_b1

sys.path.append('..')
from utils import load_model_from_state 
from models.laura_vgg_kan import VGG16_KAN
# from models.laura_vgg_kan_try import VGG16_KAN
from models.laura_resnext_kan import ResNext_KAN
from models.efficientnet_kan import EfficientNetB1_KAN
# from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
# from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
# from pytorch_grad_cam.utils.image import show_cam_on_image
from custom_grad_cam.pytorch_grad_cam.grad_cam import GradCAM
from custom_grad_cam.pytorch_grad_cam.score_cam import ScoreCAM
from custom_grad_cam.pytorch_grad_cam.eigen_cam import EigenCAM
from custom_grad_cam.pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from custom_grad_cam.pytorch_grad_cam.utils.image import show_cam_on_image


def visualize_image_with_model(path_to_model, model_type, path_to_image):

    sizeof_picture = 240
        
    # Define the preprocessing transformation
    test_transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((sizeof_picture, sizeof_picture)),
        transforms.ToTensor()
    ])

    # Load and preprocess the image
    img = cv2.imread(path_to_image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = test_transform(img).unsqueeze(0)

    params = {
        'grid_size': 32,
        'spline_order': 2,
        'scale_noise': 0.66,
        'scale_base': 0.6,
        'scale_spline': 0.62
    }

    if model_type == 'vgg':
        model = vgg16(pretrained=True)
    elif model_type == 'vgg_kan':
        model = VGG16_KAN(4, params)
    elif model_type == 'resnext':
        model = resnext50_32x4d(pretrained=True)
    elif model_type == 'resnext_kan':
        model = ResNext_KAN(4, params)
    elif model_type == 'efficientnet':     
        model = efficientnet_b1(pretrained=True)
    elif model_type == 'efficientnet_kan':     
        model = EfficientNetB1_KAN(4, params)

    model.load_state_dict(torch.load(path_to_model, map_location='cpu'))
    
    # Perform the forward pass
    model.eval() # Set the model to evaluation mode

    # Identify the target layer
    # target_layer = model.layer4[-1]
    if model_type.endswith('kan'):    
        target_layers = [model.kan_layer2]
    elif model_type == 'resnext':
        target_layers = [model.layer4[-1]]
    else:
        target_layers = [model.features[-1]]

    # We have to specify the target we want to generate the CAM for.
    # COMMENT: Whenever this is set to > 3, I get the following error later:
    # An exception occurred in CAM with block: <class 'IndexError'>. Message: index 4 is out of bounds for dimension 0 with size 4
    # If it's set to =< 3 I get this one:
    # ValueError: Invalid grads shape.Shape of grads should be 4 (2D image) or 5 (3D image).
    targets = [ClassifierOutputTarget(3)]


    # Construct the CAM object once, and then re-use it on many images.
    with GradCAM(model=model, target_layers=target_layers) as cam:
    # with EigenCAM(model=model, target_layers=target_layers) as cam:
        # You can also pass aug_smooth=True and eigen_smooth=True, to apply smoothing.
        grayscale_cam = cam(input_tensor=img_tensor, targets=targets)
        # In this example grayscale_cam has only one image in the batch:
        grayscale_cam = grayscale_cam[0, :]
        norm_img = (img-np.min(img))/(np.max(img)-np.min(img))
        norm_img = cv2.resize(norm_img, dsize=(240, 240))
        visualization = show_cam_on_image(norm_img, grayscale_cam, use_rgb=True)
        # You can also get the model outputs without having to redo inference
        model_outputs = cam.outputs   

        # print(model_outputs)
        _, predicted = torch.max(model_outputs, 1)
        print(predicted)

        cv2.imwrite('grad_cam_test.jpeg', visualization)

    # # Display the result
    # cv2.imshow('Grad-CAM', superimposed_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run this script in order to perform hyperparameter tuning on the desired model.')
    parser.add_argument('model_path', type=str, help='pass the path to the model\'s checkpoint')
    parser.add_argument('model_type', type=str, help='pass the of model')
    parser.add_argument('image_path', type=str, help='pass the path to the desired image')

    args = parser.parse_args()

    visualize_image_with_model(args.model_path, args.model_type, args.image_path)