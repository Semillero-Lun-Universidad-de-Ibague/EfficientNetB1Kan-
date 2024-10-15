import torch, json, time
import numpy as np
import torch.optim as optim

from models.ConvNeXt_KAN import ConvNeXtKAN


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


# ----------------
# JSON OPERATIONS
# ----------------

def update_json_with_key(filename, new_data):
    try:
        with open(filename, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        data = []

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