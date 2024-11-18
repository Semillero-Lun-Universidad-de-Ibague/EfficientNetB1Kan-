import torch, time, argparse, sys
import numpy as np

from tqdm import tqdm
from torchsummary import summary
from torchvision.models import vgg16, resnext50_32x4d, efficientnet_b1

sys.path.append('..')
from data_preparation import prepare_dataset
from utils import update_json_with_key
from KANs_original.models.vgg_kan import VGG16_KAN
from KANs_original.models.resnext_kan import ResNext_KAN
from KANs_original.models.efficientnet_kan import EfficientNetB1_KAN
from KANs_new_appr.models.vgg_kan_mid import VGG16_KAN_Mid
from KANs_new_appr.models.resnext_kan_mid import ResNext_KAN_Mid
from KANs_new_appr.models.efficientnet_kan_mid import EfficientNetB1_KAN_Mid
from KANs_new_appr.models.efficientnet_convkan_mid import EfficientNetB1_ConvKAN_Mid
from KANs_new_appr.models.kan_model import KAN_Model
from KANs_new_appr.models.kan_conv_model import ConvKAN_Model

NAME_JSON_FILE = 'data.json'
NAME_PRED_FILE = 'preds.json'

MODEL_SAVING_POSTFIX = "_checkpoint.pth"

batch_size = 8

# try:
#     device
# except NameError:
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Vytiskněte souhrn modelu
def print_model_summary(model, input_size=(3, 128, 128)):
    model.train()
    summary(model, input_size=input_size)


def test_model(model, model_name, save_preds=False):

    print(model_name)
    if model_name.startswith('Conv'):
        datasets, loaders = prepare_dataset(batch_size, '~/kan/', grayscale=True)
    else:
        datasets, loaders = prepare_dataset(batch_size, '~/kan/')

    test_loader = loaders[2]

    model.eval()

    a_t, preds, targets = testing(model, test_loader)

    if save_preds:
        preds_json = {
            "preds": [el.item() for el in preds], 
            "labels": [el.item() for el in targets],
            "model_name": model_name
            }

        update_json_with_key(NAME_PRED_FILE, preds_json)

    # print('Saving final model: {}'.format(model))
    # # save_checkpoint(model, optimizer, '/home/semillerolun/kan/EfficientNetB1Kan-/models/model_checkpoints/' + model_name + MODEL_SAVING_POSTFIX, num_epochs)
    # torch.save(model.state_dict(), '/home/semillerolun/kan/model_checkpoints/' + model_name + '_final_' + MODEL_SAVING_POSTFIX)

    return a_t


def testing(model, test_loader):

    val_correct = 0
    val_total = 0
    preds, targets = [], []
    with tqdm(total=len(test_loader), desc='Testing', unit='batch') as pbar:
        with torch.no_grad():
            for inputs, labels in test_loader:

                if torch.cuda.is_available():
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                outputs = model(inputs)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                preds.extend(predicted)
                targets.extend(labels)
                pbar.set_postfix(val_accuracy=val_correct / val_total)
                pbar.update()

    # return avg loss and accuracy
    accuracy_testing = val_correct / val_total
    return accuracy_testing, preds, targets       


if __name__ == '__main__':
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")   
    torch.cuda.set_device(1)

    parser = argparse.ArgumentParser(description='Run this script in order to perform hyperparameter tuning on the desired model.')
    parser.add_argument('model_path', type=str,
                        help='pass the path to the desired model')
    parser.add_argument('model_type', type=str,
                        help='pass the type of the desired model')
    parser.add_argument('-p', '--save_preds', action='store_true',
                        help='set flags if predictions should be saved')
    

    args = parser.parse_args()

    model_name = args.model_path.split('/')[-1][:-4]

    if args.model_type == 'vgg':
        model = vgg16(pretrained=True)
    elif args.model_type == 'vgg_kan':
        params = {
        'num_layers': 3,
        'grid_size': 16, # 8,
        'spline_order': 2, # 3,
        'scale_noise': 0.4, # 0.5,
        'scale_base': 0.7, # 0.8,
        'scale_spline': 0.7 # 0.76
        }
        model = VGG16_KAN(4, params)
    elif args.model_type == 'vgg_kan_mid':
        params = {
        'num_layers': 3,
        'grid_size':  16,
        'spline_order':  3,
        'scale_noise':  0.75,
        'scale_base':  0.75,
        'scale_spline':  0.75
        }
        model = VGG16_KAN_Mid(4, params)
    elif args.model_type == 'resnext':
        model = resnext50_32x4d(pretrained=True)
    elif args.model_type == 'resnext_kan':
        params = {
        'num_layers': 3,
        'grid_size':  64,
        'spline_order':  3,
        'scale_noise':  0.6,
        'scale_base':  0.8,
        'scale_spline':  0.81
        }
        model = ResNext_KAN(4, params)
    elif args.model_type == 'resnext_kan_mid':
        params = {
        'num_layers': 3,
        'grid_size':  64,
        'spline_order':  3,
        'scale_noise':  0.6,
        'scale_base':  0.8,
        'scale_spline':  0.81
        }
        model = ResNext_KAN_Mid(4, params)
    elif args.model_type == 'efficientnet':     
        model = efficientnet_b1(pretrained=True)
    elif args.model_type == 'efficientnet_kan': 
        params = {
        'num_layers': 3,
        'grid_size':  16,
        'spline_order':  3,
        'scale_noise':  0.4,
        'scale_base':  0.65,
        'scale_spline':  0.8
        }    
        model = EfficientNetB1_KAN(4, params)
    elif args.model_type == 'efficientnet_kan_mid':
        params = {
        'num_layers': 3,
        'grid_size':  32,
        'spline_order':  3,
        'scale_noise':  0.54,
        'scale_base':  0.61,
        'scale_spline':  0.68
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
    elif args.model_type == 'kan':
        params = {
        'num_layers': 3,
        'grid_size':  32,
        'spline_order':  3,
        'scale_noise':  0.54,
        'scale_base':  0.61,
        'scale_spline':  0.68
        }
        model = KAN_Model(4, params)
    elif args.model_type == 'conv_kan':
        params = {
        'num_layers': 3,
        'padding': 1,
        'kernel_size': 3,
        'stride': 1
        }
        model = ConvKAN_Model(4, params)
    checkpoint = torch.load(args.model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    acc = test_model(model, model_name, args.save_preds)

    print(args.model_type)
    print(acc)
