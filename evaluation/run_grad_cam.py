import torch, cv2, argparse, sys
import numpy as np
from torchvision import transforms
from torchvision.models import vgg16, resnext50_32x4d, efficientnet_b1

sys.path.append('..')
from utils import load_model_from_state 
from KANs_new_appr.models.vgg_kan_mid import VGG16_KAN_Mid
from KANs_new_appr.models.resnext_kan_mid import ResNext_KAN_Mid
from KANs_new_appr.models.efficientnet_kan_mid import EfficientNetB1_KAN_Mid
from KANs_new_appr.models.efficientnet_convkan_mid import EfficientNetB1_ConvKAN_Mid
from KANs_new_appr.models.kan_model import KAN_Model
from KANs_new_appr.models.kan_conv_model import ConvKAN_Model
from KANs_original.models.vgg_kan import VGG16_KAN
from KANs_original.models.resnext_kan import ResNext_KAN
from KANs_original.models.efficientnet_kan import EfficientNetB1_KAN
from pytorch_grad_cam import GradCAM, HiResCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, FullGrad
# from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
# from pytorch_grad_cam.utils.image import show_cam_on_image
# from custom_grad_cam.pytorch_grad_cam.grad_cam import GradCAM
# from custom_grad_cam.pytorch_grad_cam.score_cam import ScoreCAM
# from custom_grad_cam.pytorch_grad_cam.eigen_cam import EigenCAM
from custom_grad_cam.pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from custom_grad_cam.pytorch_grad_cam.utils.image import show_cam_on_image

def visualize_image_with_model(path_to_model, model_type, path_to_image, out_image, grayscale=False):

    sizeof_picture = 240

    if grayscale:  
        # Define the preprocessing transformation
        test_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((sizeof_picture, sizeof_picture)),
            transforms.Grayscale(),
            transforms.ToTensor()
        ])
    else:
        # Define the preprocessing transformation
        test_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((sizeof_picture, sizeof_picture)),
            transforms.ToTensor()
        ])

    # Load and preprocess the image
    img = cv2.imread(path_to_image)
    # img = cv2.resize(img, (240, 240))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor = test_transform(img).unsqueeze(0)
    print(img_tensor)

    if model_type == 'vgg':
        model = vgg16(pretrained=True)
    elif model_type == 'vgg_kan':
        params = {
        'num_layers': 3,
        'grid_size': 16, # 8,
        'spline_order': 2, # 3,
        'scale_noise': 0.4, # 0.5,
        'scale_base': 0.7, # 0.8,
        'scale_spline': 0.7 # 0.76
        }
        model = VGG16_KAN(4, params)
    elif model_type == 'vgg_kan_mid':
        params = {
        'num_layers': 3,
        'grid_size':  16,
        'spline_order':  3,
        'scale_noise':  0.75,
        'scale_base':  0.75,
        'scale_spline':  0.75
        }
        model = VGG16_KAN_Mid(4, params)
    elif model_type == 'resnext':
        model = resnext50_32x4d(pretrained=True)
    elif model_type == 'resnext_kan':
        model = ResNext_KAN_Mid(4, params)
    elif model_type == 'efficientnet':     
        model = efficientnet_b1(pretrained=True)
    elif model_type == 'efficientnet_kan':     
        params = {
        'num_layers': 3,
        'grid_size':  16,
        'spline_order':  3,
        'scale_noise':  0.4,
        'scale_base':  0.65,
        'scale_spline':  0.8
        }    
        model = EfficientNetB1_KAN(4, params)
    elif model_type == 'efficientnet_kan_mid':     
        params = {
        'num_layers': 3,
        'grid_size':  16,
        'spline_order':  3,
        'scale_noise':  0.4,
        'scale_base':  0.65,
        'scale_spline':  0.8
        }    
        model = EfficientNetB1_KAN_Mid(4, params)
    elif args.model_type == 'efficientnet_convkan_mid':
        params = {
        'num_layers': 3,
        'padding': 1,
        'kernel_size': 3,
        'stride': 1
        }
        model = EfficientNetB1_ConvKAN_Mid(4, params)
    elif model_type == 'kan':     
        model = KAN_Model(4, params)
    elif model_type == 'conv_kan': 
        params = {
        'num_layers': 3,
        'padding': 1,
        'kernel_size': 3,
        'stride': 1
        }    
        model = ConvKAN_Model(4, params)

    # model.load_state_dict(torch.load(path_to_model, map_location='cpu'))
    checkpoint = torch.load(path_to_model)
    model.load_state_dict(checkpoint['model_state_dict'])
    # Perform the forward pass
    model.eval() # Set the model to evaluation mode

    # Identify the target layer
    # target_layer = model.layer4[-1]
    if model_type.endswith('kan') or model_type.endswith('kan_mid'):    
        # target_layers = [model.kan_layer2]
        target_layers = [model.feature_extractor[-1]]
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
        visualization = show_cam_on_image(norm_img, grayscale_cam, use_rgb=False)
        # You can also get the model outputs without having to redo inference
        model_outputs = cam.outputs   

        # print(model_outputs)
        _, predicted = torch.max(model_outputs, 1)
        print(model_type)
        print(predicted)

        cv2.imwrite('grad_cam_pics/' + out_image, visualization)

    # # Display the result
    # cv2.imshow('Grad-CAM', superimposed_img)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Run this script in order to perform hyperparameter tuning on the desired model.')
    parser.add_argument('model_path', type=str, help='pass the path to the model\'s checkpoint')
    parser.add_argument('model_type', type=str, help='pass the of model')
    parser.add_argument('image_path', type=str, help='pass the path to the desired image')
    parser.add_argument('out_image_path', type=str, help='pass the path to the desired image')

    args = parser.parse_args()

    if 'conv_kan' in args.model_type or 'conv_kan' in args.model_type:
        visualize_image_with_model(args.model_path, args.model_type, args.image_path, args.out_image_path, True)
    else:
        visualize_image_with_model(args.model_path, args.model_type, args.image_path, args.out_image_path)