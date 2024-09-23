import torch
import numpy as np
import json
import numpy as np


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
