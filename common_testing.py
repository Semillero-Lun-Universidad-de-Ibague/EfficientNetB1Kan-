import os
import random
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
import random
from IPython.display import Image
import imutils

from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

import torch.optim as optim
import numpy as np

from tqdm import tqdm

import torch.optim as optim

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.transforms import functional as F

import torchvision

from common import get_data_from_json
from kcn import ConvNeXtKAN

NAME_JSON_FILE = 'data.json'

MODEL_SAVING_POSTFIX = "_checkpoint.pth"

import time

batch_size = 32

try:
    device
except NameError:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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


# Vytiskněte souhrn modelu
def print_model_summary(model, input_size=(3, 128, 128)):
    from torchsummary import summary
    model.train()
    summary(model, input_size=input_size)


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


train_loader = None
valid_loader = None
test_loader = None
train_dataset = None
valid_dataset = None
test_dataset = None


def prepare_dataset():
    from pathlib import Path

    # `cwd`: current directory is straightforward
    cwd = Path.cwd()

    train_dir = str(cwd) + "/data/Training"
    test_dir = str(cwd) + "/data/Testing"

    sizeof_picture = 240

    # Define transformations
    train_transform = transforms.Compose([
        transforms.RandomRotation(degrees=10),
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=0, translate=(0, 0.2)),
        transforms.Resize((sizeof_picture, sizeof_picture)),
        transforms.ToTensor()
    ])

    valid_transform = transforms.Compose([
        transforms.Resize((sizeof_picture, sizeof_picture)),
        transforms.ToTensor()
    ])

    test_transform = transforms.Compose([
        transforms.Resize((sizeof_picture, sizeof_picture)),
        transforms.ToTensor()
    ])

    # Load datasets
    global train_dataset
    global valid_dataset
    global test_dataset
    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transform)

    # Split the training data into training and validation sets
    num_train = len(train_dataset)
    split = int(0.8 * num_train)  # 80% training, 20% validation
    train_size = split
    valid_size = num_train - split

    train_dataset, valid_dataset = random_split(train_dataset, [train_size, valid_size])

    print(f"batch size: {batch_size}")
    # Create DataLoader
    global train_loader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    global valid_loader
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    global test_loader
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    "train: ", len(train_dataset), "Valid: ", len(valid_dataset), "test: ", len(test_dataset)


def test_model(model, model_name, num_epochs=5, progress_bar=True):
    # Hope you don't be imprisoned by legacy Python code :)
    # # ## Compile Model
    #
    prepare_dataset()

    model = model.to(device)

    print_model_summary(model)

    print(f"batch size: {batch_size}")

    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Define the loss function
    criterion = nn.CrossEntropyLoss()

    # # ## Model Training and Model Evaluation
    #

    from torch.optim.lr_scheduler import ReduceLROnPlateau

    # Define the scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.3, patience=2, verbose=True)

    # Initialize early stopping
    early_stopping = EarlyStopping(patience=5, verbose=True)

    accuracy_training = []
    accuracy_validation = []

    loss_validation = []
    loss_training = []

    test_validation = []
    test_training = []

    test_loss = []
    test_accuracy = []

    time_trainings = []
    time_training = Timer()
    total_time = Timer()
    total_time.start()

    for epoch in range(num_epochs):

        time_training.start()

        # train'
        epoch_acc, epoch_loss = train(criterion, epoch, model, num_epochs, optimizer, progress_bar)
        time_training.end()
        accuracy_training.append(epoch_acc)
        loss_training.append(epoch_loss)

        time_trainings.append(time_training.elapsed_time)

        # Validation
        val_acc, val_loss = validation(criterion, model, progress_bar)

        loss_validation.append(val_loss)
        accuracy_validation.append(val_acc)

        # Step the scheduler
        scheduler.step(val_acc)

        a_t, l_t = testing(criterion, model)

        test_loss.append(l_t)
        test_accuracy.append(a_t)

        # Check early stopping
        if early_stopping(val_loss, model):
            num_epochs = epoch
            break

    total_time.end()

    from common import update_json_with_key

    tb_json = {
        "accuracy_training": accuracy_training,
        "accuracy_validation": accuracy_validation,

        "loss_training": loss_training,
        "loss_validation": loss_validation,

        "time_trainings": time_trainings,
        "total_time": total_time.elapsed_time,

        "accuracy_testing": test_accuracy,
        "loss_testing": test_loss,

        "num_epochs": num_epochs,

        "model_name": model_name,

    }

    from common import update_json_with_key

    update_json_with_key(NAME_JSON_FILE, tb_json)

    # from common import save_checkpoint

    # save_checkpoint(model, optimizer, model_name + MODEL_SAVING_POSTFIX, num_epochs)

    # plot_graph(accuracy_testing, accuracy_training, loss_training, loss_validation, model_name, num_epochs)

    return accuracy_validation, loss_validation, time_trainings, test_accuracy[-1], test_loss[-1]


def testing(criterion, model):
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    with tqdm(total=len(test_loader), desc='Testing', unit='batch') as pbar:
        with torch.no_grad():
            for inputs, labels in test_loader:

                if torch.cuda.is_available():
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

                # Update progress bar
                pbar.set_postfix(val_loss=val_loss / (val_total + 1e-8), val_accuracy=val_correct / val_total)
                pbar.update()

    # return avg loss and accuracy
    accuracy_testing = val_correct / val_total
    loss_testing = val_loss / len(test_dataset)
    return accuracy_testing, loss_testing


def validation(criterion, model, progress_bar=True):
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0
    # Validation loop with progress bar
    with tqdm(total=len(valid_loader), desc='Validation', unit='batch', disable=progress_bar) as pbar:
        with torch.no_grad():
            for inputs, labels in valid_loader:

                if torch.cuda.is_available():
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()

                # Update progress bar
                pbar.set_postfix(val_loss=val_loss / (val_total + 1e-8), val_accuracy=val_correct / val_total)
                pbar.update()
    val_loss = val_loss / len(valid_dataset)
    val_acc = val_correct / val_total
    print(f'Validation Loss: {val_loss:.4f} - Accuracy: {val_acc:.4f}')
    return val_acc, val_loss


def train(criterion, epoch, model, num_epochs, optimizer, progress_bar=True):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    epoch_acc = 0
    # Training loop with progress bar
    with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch',
              disable=progress_bar) as pbar:
        for inputs, labels in train_loader:

            if torch.cuda.is_available():
                inputs = inputs.to(device)
                labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            # Update progress bar
            pbar.set_postfix(loss=running_loss / (total + 1e-8), accuracy=correct / total)
            pbar.update()
    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = correct / total
    print(f'Epoch {epoch + 1}/{num_epochs} - Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc:.4f}')
    return epoch_acc, epoch_loss


def load_model_and_params(model, optimizer, accuracy_testing, accuracy_training, loss_training, loss_validation,
                          model_name, num_epochs):
    from common import load_checkpoint
    epochs_accuracy, accuracy_training, accuracy_testing, epochs_loss, loss_training, loss_validation, name_of_mode = get_data_from_json(
        NAME_JSON_FILE, model_name)
    model, optimizer, num_epochs = load_checkpoint(model, optimizer, model_name + MODEL_SAVING_POSTFIX, )
    return accuracy_testing, accuracy_training, loss_training, loss_validation, num_epochs


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


def load_model_poc():
    global model_name, accuracy_training, accuracy_testing, loss_validation, loss_training, num_epochs
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
