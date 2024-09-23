import json
import matplotlib.pyplot as plt
import numpy as np

# Load the JSON data
with open('data.json', 'r') as file:
    data = json.load(file)

# Extract model names and their final testing accuracy
plt.figure(figsize=(12, 6))

data_array = []

for model_name in data.keys():
    if len(data[model_name]["accuracy_testing"]) == 30:
        data_array.append(data[model_name])


def avg_func(i):
    return sum(i["accuracy_testing"]) / len(i["accuracy_testing"])
    # return max(i["accuracy_testing"])
    # return i["accuracy_testing"][0]


data_array.sort(key=avg_func, reverse=True)

for model in data_array[:3]:
    model_data = model

    # Extract relevant arrays
    epochs_accuracy = model_data["epochs_accuracy"]
    accuracy_training = model_data["accuracy_training"]
    accuracy_testing = model_data["accuracy_testing"]
    epochs_loss = model_data["epochs_loss"]
    loss_training = model_data["loss_training"]
    loss_validation = model_data["loss_validation"]

    # plt.plot(epochs_accuracy, accuracy_training, label=model_name)
    plt.plot(epochs_accuracy, accuracy_testing,
             label=model_data["name_of_model"])

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
