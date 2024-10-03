import json
import numpy as np
import matplotlib.pyplot as plt

## TODO: ask Jakub about this, this is a different data format!

# Load the JSON data
with open('../data.json', 'r') as file:
    data = json.load(file)

# Extract model names and their final testing accuracy
plt.figure(figsize=(12, 6))

data_array = []

for model in data:
    if len(model['accuracy_testing']) == 30:
        data_array.append(model)


def avg_func(i):
    return sum(i["accuracy_testing"]) / len(i["accuracy_testing"])
    # return max(i["accuracy_testing"])
    # return i["accuracy_testing"][0]


data_array.sort(key=avg_func, reverse=True)

for model in data_array[:3]:
    model_data = model

    # Extract relevant arrays
    # epochs_accuracy = model_data["epochs_accuracy"]
    epochs_accuracy = list(range(model_data["num_epochs"]))
    accuracy_training = model_data["accuracy_training"]
    accuracy_validation = model_data["accuracy_validation"]
    accuracy_testing = model_data["accuracy_testing"]
    # epochs_loss = model_data["epochs_loss"]
    loss_training = model_data["loss_training"]
    loss_validation = model_data["loss_validation"]

    # plt.plot(epochs_accuracy, accuracy_training, label=model_name)
    plt.plot(epochs_accuracy, accuracy_testing,
             label=model_data["model_name"])

# add the basicMLP model for comparison
model_data = data["BasicMLP"]
epochs_accuracy = model_data["epochs_accuracy"]
accuracy_testing = model_data["accuracy_testing"]
plt.plot(epochs_accuracy, accuracy_testing, label='MLP')

plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Testing Accuracy for different params')
plt.legend()
plt.grid(True)

plt.savefig('top_models_plot_benchamark.png')


# TODO: figure out what to do here!
def plot_graph(accuracy_testing, accuracy_training, loss_training, loss_validation, model_name, num_epochs):
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    epochs = np.arange(1, num_epochs + 1, num_epochs / len(accuracy_training))
    ax[0].plot(epochs, accuracy_training, 'g', label='Training Accuracy')
    epochs = np.arange(1, num_epochs + 1, num_epochs / len(accuracy_testing))
    ax[0].plot(epochs, accuracy_testing, 'y-', label='Validation Accuracy')
    ax[0].set_title('Model Training & Validation Accuracy')
    ax[0].legend(loc='lower right')
    ax[0].set_xlabel("Epochs")
    ax[0].set_ylabel("Accuracy")
    epochs = np.arange(0, num_epochs, num_epochs / len(loss_training))
    ax[1].plot(epochs, loss_training, 'g', label='Training Loss')
    epochs = np.arange(0, num_epochs, num_epochs / len(loss_validation))
    ax[1].plot(epochs, loss_validation, 'y-', label='Validation Loss')
    ax[1].set_title('Model Training & Validation & Loss')
    ax[1].legend()
    ax[1].set_xlabel("Epochs")
    ax[1].set_ylabel("Loss")
    plt.savefig(model_name + ".svg")
