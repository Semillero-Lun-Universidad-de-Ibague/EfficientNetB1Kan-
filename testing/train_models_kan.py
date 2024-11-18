import torch, argparse, sys

from datetime import date
from torchvision.models import vgg16, resnext50_32x4d, efficientnet_b1

sys.path.append('..')
import common_training
from common_training import train_model
from KANs_original.models.vgg_kan import VGG16_KAN
from KANs_original.models.resnext_kan import ResNext_KAN
from KANs_original.models.efficientnet_kan import EfficientNetB1_KAN
from KANs_new_appr.models.vgg_kan_mid import VGG16_KAN_Mid
from KANs_new_appr.models.resnext_kan_mid import ResNext_KAN_Mid
from KANs_new_appr.models.efficientnet_kan_mid import EfficientNetB1_KAN_Mid
from KANs_new_appr.models.kan_model import KAN_Model
from KANs_new_appr.models.kan_conv_model import ConvKAN_Model
from KANs_new_appr.models.efficientnet_convkan_mid import EfficientNetB1_ConvKAN_Mid


NAME_JSON_FILE = 'data_experiment_params.json'

MODEL_SAVING_POSTFIX = "_checkpoint.pth"

common_training.batch_size = 8

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(0)

    parser = argparse.ArgumentParser(description='Run this script in order to perform hyperparameter tuning on the desired model.')
    parser.add_argument('model_name', type=str,
                        help='pass the name of the model (vgg|vgg_kan|vgg_kan_mid|resnext|resnext_kan|efficientnet|efficientnet_kan|efficientnet_kan_mid)')
    parser.add_argument('num_epochs', type=str,
                        help='pass the number of epochs the model should train for')
    
    args = parser.parse_args()

    num_epochs = args.num_epochs

    # TODO: continue here, add significance tests       
    if args.model_name == 'vgg':
        model = vgg16(pretrained=True)
        train_model(model, 'VGG16_{}epochs_{}'.format(num_epochs, date.today()), num_epochs=int(num_epochs), progress_bar=False)
    elif args.model_name == 'vgg_kan':
        params = {
        'num_layers': 3,
        'grid_size': 16, # 8,
        'spline_order': 2, # 3,
        'scale_noise': 0.4, # 0.5,
        'scale_base': 0.7, # 0.8,
        'scale_spline': 0.7 # 0.76
        }
        model = VGG16_KAN(4, params)
        train_model(model, 'VGG16_KAN_{}epochs_{}'.format(num_epochs, date.today()), num_epochs=int(num_epochs), progress_bar=False)
    elif args.model_name == 'vgg_kan_mid':
        params = {
        'num_layers': 3,
        'grid_size':  16,
        'spline_order':  3,
        'scale_noise':  0.75,
        'scale_base':  0.75,
        'scale_spline':  0.75
        }
        model = VGG16_KAN_Mid(4, params)
        train_model(model, 'VGG16_KAN_Mid_{}epochs_{}'.format(num_epochs, date.today()), num_epochs=int(num_epochs), progress_bar=False)
    elif args.model_name == 'resnext':
        resnext = resnext50_32x4d(pretrained=True)
        train_model(resnext, 'ResNext_{}epochs_{}'.format(num_epochs, date.today()), num_epochs=int(num_epochs), progress_bar=False)
    elif args.model_name == 'resnext_kan':
        params = {
        'num_layers': 3,
        'grid_size':  64,
        'spline_order':  3,
        'scale_noise':  0.6,
        'scale_base':  0.8,
        'scale_spline':  0.81
        }
        model = ResNext_KAN(4, params)
        train_model(model, 'ResNext_KAN_{}epochs_{}'.format(num_epochs, date.today()), num_epochs=int(num_epochs), progress_bar=False)
    elif args.model_name == 'resnext_kan_mid':
        params = {
        'num_layers': 3,
        'grid_size':  64,
        'spline_order':  3,
        'scale_noise':  0.6,
        'scale_base':  0.8,
        'scale_spline':  0.81
        }
        model = ResNext_KAN_Mid(4, params)
        train_model(model, 'ResNext_KAN_Mid_{}epochs_{}'.format(num_epochs, date.today()), num_epochs=int(num_epochs), progress_bar=False)
    elif args.model_name == 'efficientnet':     
        model = efficientnet_b1(pretrained=True)
        train_model(model, 'EfficientNet_{}epochs_{}'.format(num_epochs, date.today()), num_epochs=int(num_epochs), progress_bar=False)
    elif args.model_name == 'efficientnet_kan': 
        params = {
        'num_layers': 3,
        'grid_size':  16,
        'spline_order':  3,
        'scale_noise':  0.4,
        'scale_base':  0.65,
        'scale_spline':  0.8
        }    
        model = EfficientNetB1_KAN(4, params)
        train_model(model, 'EfficientNet_KAN_{}epochs_{}'.format(num_epochs, date.today()), num_epochs=int(num_epochs), progress_bar=False)
    elif args.model_name == 'efficientnet_kan_mid':
        params = {
        'num_layers': 3,
        'grid_size':  32,
        'spline_order':  3,
        'scale_noise':  0.54,
        'scale_base':  0.61,
        'scale_spline':  0.68
        }
        model = EfficientNetB1_KAN_Mid(4, params)
        train_model(model, 'EfficientNet_KAN_Mid_{}epochs_{}'.format(num_epochs, date.today()), num_epochs=int(num_epochs), progress_bar=False)
    elif args.model_name == 'efficientnet_convkan_mid':
        params = {
        'num_layers': 3,
        'padding': 1,
        'kernel_size': 3,
        'stride': 1
        }
        model = EfficientNetB1_ConvKAN_Mid(4, params)
        train_model(model, 'EfficientNet_ConvKAN_Mid_{}epochs_{}'.format(num_epochs, date.today()), num_epochs=int(num_epochs), progress_bar=False)
    elif args.model_name == 'kan':
        params = {
        'num_layers': 3,
        'grid_size':  32,
        'spline_order':  3,
        'scale_noise':  0.54,
        'scale_base':  0.61,
        'scale_spline':  0.68
        }
        model = KAN_Model(4, params)
        print('training model')
        train_model(model, 'KAN_model_{}epochs_{}'.format(num_epochs, date.today()), num_epochs=int(num_epochs), progress_bar=False)
    elif args.model_name == 'conv_kan':
        params = {
        'num_layers': 3,
        'padding': 1,
        'kernel_size': 3,
        'stride': 1
        }   
        model = ConvKAN_Model(4, params)
        print('training model')
        train_model(model, 'ConvKAN_model_{}epochs_{}'.format(num_epochs, date.today()), num_epochs=int(num_epochs), progress_bar=False)