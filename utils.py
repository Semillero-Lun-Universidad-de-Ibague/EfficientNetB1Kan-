import torch, json, time
import numpy as np
import torch.optim as optim

from torchvision.models import vgg16, resnext50_32x4d, efficientnet_b1

from KANs_original.models.vgg_kan import VGG16_KAN
from KANs_original.models.resnext_kan import ResNext_KAN
from KANs_original.models.efficientnet_kan import EfficientNetB1_KAN
from KANs_new_appr.models.vgg_kan_mid import VGG16_KAN_Mid
from KANs_new_appr.models.resnext_kan_mid import ResNext_KAN_Mid
from KANs_new_appr.models.efficientnet_kan_mid import EfficientNetB1_KAN_Mid
from KANs_new_appr.models.kan_model import KAN_Model
from KANs_new_appr.models.kan_conv_model import ConvKAN_Model
from KANs_new_appr.models.efficientnet_convkan_mid import EfficientNetB1_ConvKAN_Mid

# --------
# CLASSES
# --------

class EarlyStopping:
    def __init__(self, patience=5, verbose=False):
        self.patience = patience
        self.verbose = verbose
        self.best_score = None
        self.early_stop_counter = 0
        self.best_model_wts = None

    def __call__(self, val_loss, model):
        if self.best_score is None:
            self.best_score = val_loss
            self.best_model_wts = model.state_dict()
        elif val_loss < self.best_score:
            self.best_score = val_loss
            self.best_model_wts = model.state_dict()
            self.early_stop_counter = 0
        else:
            self.early_stop_counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.early_stop_counter} out of {self.patience}')
            if self.early_stop_counter >= self.patience:
                if self.verbose:
                    print('Early stopping')
                model.load_state_dict(self.best_model_wts)
                return True
        return False


class Timer:
    def __init__(self, use_cuda=False):
        self.use_cuda = use_cuda
        if self.use_cuda:
            self.start_event = torch.cuda.Event(enable_timing=True)
            self.end_event = torch.cuda.Event(enable_timing=True)
        else:
            self.start_time = None
            self.end_time = None

        self.__elapsed_time = 0

    def start(self):
        if self.use_cuda:
            self.start_event.record()
        else:
            self.start_time = time.time()

    def end(self):
        if self.use_cuda:
            self.end_event.record()
            torch.cuda.synchronize()  # Ensure that the events are completed
            self.__elapsed_time = self.start_event.elapsed_time(self.end_event)  # in milliseconds
        else:
            self.end_time = time.time()
            self.__elapsed_time = (self.end_time - self.start_time) * 1000  # Convert to milliseconds
        return self.__elapsed_time

    @property
    def elapsed_time(self):
        return self.__elapsed_time


# ----------------
# PARAMETER STUFF
# ----------------

def print_parameter_details(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            params = parameter.numel()  # Number of elements in the tensor
            total_params += params
            print(f"{name}: {params}")
    print(f"Total trainable parameters: {total_params}")


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# -----------------------------
# SAVING & LOADING CHECKPOINTS
# -----------------------------

class SaveBestModel:
    """
    Class to save the best model while training. If the current epoch's 
    validation loss is less than the previous least less, then save the
    model state.
    """
    def __init__(
        self, best_valid_loss=float('inf')
    ):
        self.best_valid_loss = best_valid_loss
    def __call__(
        self, current_valid_loss, 
        epoch, model, optimizer, criterion, save_path
    ):
        if current_valid_loss < self.best_valid_loss:
            self.best_valid_loss = current_valid_loss
            print(f"\nBest validation loss: {self.best_valid_loss}")
            print(f"\nSaving best model for epoch: {epoch+1}\n")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': criterion,
                }, save_path)


def save_checkpoint(model, optimizer, save_path, epoch):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }, save_path)


def load_checkpoint(model, optimizer, load_path):
    checkpoint = torch.load(load_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']

    return model, optimizer, epoch


# -----------------------
# LOADING MODEL & PARAMS
# -----------------------

def load_model_and_params(model, optimizer, accuracy_testing, accuracy_training, loss_training, loss_validation,
                          model_name, num_epochs, json_file, model_saving_postfix):
    
    epochs_accuracy, accuracy_training, accuracy_testing, epochs_loss, loss_training, loss_validation, name_of_mode = get_data_from_json(
        json_file, model_name)
    model, optimizer, num_epochs = load_checkpoint(model, optimizer, model_name + model_saving_postfix, )
    return accuracy_testing, accuracy_training, loss_training, loss_validation, num_epochs


def load_model_poc():
    model = ConvNeXtKAN()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    model_name = "ConvNeXtKAN"
    accuracy_training = []
    accuracy_testing = []
    loss_validation = []
    loss_training = []
    num_epochs = 0
    accuracy_testing, accuracy_training, loss_training, loss_validation, num_epochs = (
        load_model_and_params
            (
            model,
            optimizer,
            accuracy_testing, accuracy_training, loss_training, loss_validation, model_name, num_epochs
        ))


def load_model_from_state(state_dict_path: str, model_type: str):

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
        resnext = resnext50_32x4d(pretrained=True)
    elif model_type == 'resnext_kan':
        params = {
        'num_layers': 3,
        'grid_size':  64,
        'spline_order':  3,
        'scale_noise':  0.6,
        'scale_base':  0.8,
        'scale_spline':  0.81
        }
        model = ResNext_KAN(4, params)
    elif model_type == 'resnext_kan_mid':
        params = {
        'num_layers': 3,
        'grid_size':  64,
        'spline_order':  3,
        'scale_noise':  0.6,
        'scale_base':  0.8,
        'scale_spline':  0.81
        }
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
        'grid_size':  32,
        'spline_order':  3,
        'scale_noise':  0.54,
        'scale_base':  0.61,
        'scale_spline':  0.68
        }
        model = EfficientNetB1_KAN_Mid(4, params)
    elif model_type == 'efficientnet_convkan_mid':
        params = {
        'num_layers': 3,
        'padding': 1,
        'kernel_size': 3,
        'stride': 1
        }
        model = EfficientNetB1_ConvKAN_Mid(4, params)
    elif model_type == 'kan':
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
    elif model_type == 'conv_kan':
        params = {
        'num_layers': 3,
        'padding': 1,
        'kernel_size': 3,
        'stride': 1
        }   
        model = ConvKAN_Model(4, params)
    model_state = model.load_state_dict(torch.load(state_dict_path, map_location='cpu'))

    return model_state

# ----------------
# JSON OPERATIONS
# ----------------

def update_json_with_key(filename, new_data):
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        data = []

    added = False
    for i, dic in enumerate(data):
        if dic['model_name'] == new_data['model_name']:
            data[i] = new_data
            added = True
        
    if not added:
        # Update the data under the specified key
        data.append(new_data)

    # Save updated data back to JSON
    with open(filename, 'w') as f:
        json.dump(data, f, indent=4)


def get_data_from_json(filename, key):
    # Load data from JSON
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"The file {filename} does not exist.")

    # Retrieve data under the specified key
    if key not in data:
        raise KeyError(f"The key '{key}' was not found in the file.")

    model_data = data[key]

    # Extract and convert data back to numpy arrays
    epochs_accuracy = np.array(model_data["epochs_accuracy"])
    accuracy_training = np.array(model_data["accuracy_training"])
    accuracy_testing = np.array(model_data["accuracy_testing"])
    epochs_loss = np.array(model_data["epochs_loss"])
    loss_training = np.array(model_data["loss_training"])
    loss_validation = np.array(model_data["loss_validation"])
    name_of_mode = model_data.get("name_of_mode:", None)

    return epochs_accuracy, accuracy_training, accuracy_testing, epochs_loss, loss_training, loss_validation, name_of_mode


def merge_jsons(file1, file2, out_name):

    # List all JSON files you want to merge
    json_files = [file1, file2]

    # Empty dictionary for storing merged data
    combined_data = {}

    # Read each JSON file and update the dictionary
    for json_file in json_files:
        with open(json_file, "r") as f:
            data = json.load(f)
            combined_data.update(data)

    # Save the merged data into a single JSON file
    with open(out_name, "w") as f:
        json.dump(combined_data, f, indent=4)