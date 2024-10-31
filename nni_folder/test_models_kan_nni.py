import os, random, itertools, torch, torchvision, nni, argparse, sys

from datetime import date
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from torchvision.models import vgg16, resnext50_32x4d, efficientnet_b1

# sys.path.append('..')
import common_testing_nni
from common_testing_nni import test_model
from utils_nni import get_data_from_json
from laura_vgg_kan_nni import VGG16_KAN
from vgg_kan_mid_nni import VGG16_KAN_Mid
from laura_resnext_kan_nni import ResNext_KAN
from efficientnet_kan_nni import EfficientNetB1_KAN
from efficientnet_kan_mid_nni import EfficientNetB1_KAN_Mid

NAME_JSON_FILE = 'data_experiment_params.json'

MODEL_SAVING_POSTFIX = "_checkpoint.pth"

common_testing_nni.batch_size = 210

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   
    torch.cuda.set_device(0)

    parser = argparse.ArgumentParser(description='Run this script in order to perform hyperparameter tuning on the desired model.')
    parser.add_argument('model_name', type=str,
                        help='pass the name of the model (vgg|vgg_kan|resnext|resnext_kan)')
    parser.add_argument('num_epochs', type=str,
                        help='pass the number of epochs the model should train for')

    args = parser.parse_args()

    # Obtener hiperparámetros desde NNI
    params_nni = nni.get_next_parameter()

    params = {
        'grid_size': params_nni.get('grid_size', 16),
        'spline_order': params_nni.get('spline_order', 3),
        'scale_noise': params_nni.get('scale_noise', 0.73),
        'scale_base': params_nni.get('scale_base', 0.76),
        'scale_spline': params_nni.get('scale_spline', 0.5)
    }

    num_epochs = args.num_epochs

    # TODO: continue here, add significance tests       
    if args.model_name == 'vgg':
        vgg = vgg16(pretrained=True)
        test_model(vgg, 'VGG16_{}epochs_{}'.format(num_epochs, date.today()), num_epochs=int(num_epochs), progress_bar=False)
    elif args.model_name == 'vgg_kan':
        model = VGG16_KAN(4, params)
        # print('VGG KAN model:')
        # print(model)
        test_model(model, 'VGG16_KAN_{}epochs_{}'.format(num_epochs, date.today()), num_epochs=int(num_epochs), progress_bar=False)
    elif args.model_name == 'vgg_kan_mid':
        model = VGG16_KAN_Mid(4, params)
        # print('VGG KAN model:')
        # print(model)
        test_model(model, 'VGG16_KAN_Mid_{}epochs_{}'.format(num_epochs, date.today()), num_epochs=int(num_epochs), progress_bar=False)
    elif args.model_name == 'resnext':
        resnext = resnext50_32x4d(pretrained=True)
        test_model(resnext, 'ResNext_{}epochs_{}'.format(num_epochs, date.today()), num_epochs=int(num_epochs), progress_bar=False)
    elif args.model_name == 'resnext_kan':
        model = ResNext_KAN(4, params)
        test_model(model, 'ResNext_KAN_{}epochs_{}'.format(num_epochs, date.today()), num_epochs=int(num_epochs), progress_bar=False)
    elif args.model_name == 'efficientnet':     
        model = efficientnet_b1(pretrained=True)
        test_model(model, 'EfficientNet_{}epochs_{}'.format(num_epochs, date.today()), num_epochs=int(num_epochs), progress_bar=False)
    elif args.model_name == 'efficientnet_kan':     
        model = EfficientNetB1_KAN(4, params)
        test_model(model, 'EfficientNet_KAN_{}epochs_{}'.format(num_epochs, date.today()), num_epochs=int(num_epochs), progress_bar=False)
    elif args.model_name == 'efficientnet_kan_mid':     
        model = EfficientNetB1_KAN_Mid(4, params)
        test_model(model, 'EfficientNet_KAN_Mid_{}epochs_{}'.format(num_epochs, date.today()), num_epochs=int(num_epochs), progress_bar=False)
