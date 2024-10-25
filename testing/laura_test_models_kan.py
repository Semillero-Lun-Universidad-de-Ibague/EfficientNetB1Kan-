import os, random, itertools, torch, torchvision, nni, argparse, sys

from datetime import date
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from torchvision.models import vgg16, resnext50_32x4d, efficientnet_b1

sys.path.append('..')
import common_testing
from common_testing import test_model
from utils import get_data_from_json
from models.laura_vgg_kan_try import VGG16_KAN
from models.laura_resnext_kan import ResNext_KAN
from models.efficientnet_kan import EfficientNetB1_KAN

NAME_JSON_FILE = 'data_experiment_params.json'

MODEL_SAVING_POSTFIX = "_checkpoint.pth"

common_testing.batch_size = 210

if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(0)

    parser = argparse.ArgumentParser(description='Run this script in order to perform hyperparameter tuning on the desired model.')
    parser.add_argument('model_name', type=str,
                        help='pass the name of the model (vgg|vgg_kan|resnext|resnext_kan)')

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

    # TODO: continue here, add significance tests       
    if args.model_name == 'vgg':
        vgg = vgg16(pretrained=True)
        test_model(vgg, 'VGG16_10epochs_{}'.format(date.today()), num_epochs=1, progress_bar=False)
    elif args.model_name == 'vgg_kan':
        model = VGG16_KAN(4, params)
        # print('VGG KAN model:')
        # print(model)
        test_model(model, 'VGG16_KAN_try_1epochs_{}'.format(date.today()), num_epochs=1, progress_bar=False)
    elif args.model_name == 'resnext':
        resnext = resnext50_32x4d(pretrained=True)
        test_model(resnext, 'ResNext_10epochs_{}'.format(date.today()), num_epochs=10, progress_bar=False)
    elif args.model_name == 'resnext_kan':
        model = ResNext_KAN(4, params)
        test_model(model, 'ResNext_KAN_10epochs_{}'.format(date.today()), num_epochs=10, progress_bar=False)
    elif args.model_name == 'efficientnet':     
        model = efficientnet_b1(pretrained=True)
        test_model(model, 'EfficientNet_10epochs_{}'.format(date.today()), num_epochs=10, progress_bar=False)
    elif args.model_name == 'efficientnet_kan':     
        model = EfficientNetB1_KAN(4, params)
        test_model(model, 'EfficientNet_10epochs_kan{}'.format(date.today()), num_epochs=1, progress_bar=False)


