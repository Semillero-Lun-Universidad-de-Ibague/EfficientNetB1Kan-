import torch, argparse, sys

from datetime import date
from torchvision.models import vgg16, resnext50_32x4d, efficientnet_b1

sys.path.append('..')
import common_training
from common_training import train_model
from KANs_original.models.laura_vgg_kan import VGG16_KAN
from KANs_original.models.laura_resnext_kan import ResNext_KAN
from KANs_original.models.efficientnet_kan import EfficientNetB1_KAN
from KANs_new_appr.models.vgg_kan_mid import VGG16_KAN_Mid
from KANs_new_appr.models.efficientnet_kan_mid import EfficientNetB1_KAN_Mid

NAME_JSON_FILE = 'data_experiment_params.json'

MODEL_SAVING_POSTFIX = "_checkpoint.pth"

common_training.batch_size = 210

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(1)

    parser = argparse.ArgumentParser(description='Run this script in order to perform hyperparameter tuning on the desired model.')
    parser.add_argument('model_name', type=str,
                        help='pass the name of the model (vgg|vgg_kan|resnext|resnext_kan)')
    parser.add_argument('num_epochs', type=str,
                        help='pass the number of epochs the model should train for')
    

    args = parser.parse_args()

    num_epochs = args.num_epochs

    # TODO: continue here, add significance tests       
    if args.model_name == 'vgg':
        vgg = vgg16(pretrained=True)
        train_model(vgg, 'VGG16_{}epochs_{}_2try'.format(num_epochs, date.today()), num_epochs=int(num_epochs), progress_bar=False)
    elif args.model_name == 'vgg_kan':
        params = {
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
        'grid_size':  64,
        'spline_order':  3,
        'scale_noise':  0.6,
        'scale_base':  0.8,
        'scale_spline':  0.81
        }
        model = ResNext_KAN(4, params)
        train_model(model, 'ResNext_KAN_{}epochs_{}'.format(num_epochs, date.today()), num_epochs=int(num_epochs), progress_bar=False)
    elif args.model_name == 'efficientnet':     
        model = efficientnet_b1(pretrained=True)
        train_model(model, 'EfficientNet_{}epochs_{}'.format(num_epochs, date.today()), num_epochs=int(num_epochs), progress_bar=False)
    elif args.model_name == 'efficientnet_kan': 
        params = {
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
        'grid_size':  32,
        'spline_order':  3,
        'scale_noise':  0.54,
        'scale_base':  0.61,
        'scale_spline':  0.68
        }
        model = EfficientNetB1_KAN_Mid(4, params)
        train_model(model, 'EfficientNet_KAN_Mid_{}epochs_{}'.format(num_epochs, date.today()), num_epochs=int(num_epochs), progress_bar=False)


