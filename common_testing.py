import torch, time
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np

from tqdm import tqdm
from torchsummary import summary
from torchvision.transforms import functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau

from data_preparation import prepare_dataset
from utils import EarlyStopping, update_json_with_key, Timer, save_checkpoint  
from evaluation.significance_testing import test_statistical_significance

NAME_JSON_FILE = 'data.json'
NAME_PRED_FILE = 'preds.json'

MODEL_SAVING_POSTFIX = "_checkpoint.pth"

batch_size = 32

# try:
#     device
# except NameError:
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Vytiskněte souhrn modelu
def print_model_summary(model, input_size=(3, 128, 128)):
    model.train()
    summary(model, input_size=input_size)


def test_model(model, model_name, num_epochs=5, progress_bar=True):

    # prepare_dataset(batch_size, '~/kan/')
    datasets, loaders = prepare_dataset(batch_size, '~/kan/')

    train_dataset, valid_dataset, test_dataset = datasets[0], datasets[1], datasets[2]
    train_loader, valid_loader, test_loader = loaders[0], loaders[1], loaders[2]

    model = model.to(device)

    print_model_summary(model)

    print(f"batch size: {batch_size}")

    # Define the optimizer
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Define the loss function
    criterion = nn.CrossEntropyLoss()

    # Define the scheduler
    scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.3, patience=2, verbose=True)

    # Initialize early stopping
    early_stopping = EarlyStopping(patience=5, verbose=True)

    accuracy_training = []
    accuracy_validation = []

    loss_validation = []
    loss_training = []

    test_loss = []
    test_accuracy = []
    predictions = []
    labels = []

    time_trainings = []
    time_training = Timer()
    total_time = Timer()
    total_time.start()

    for epoch in range(num_epochs):

        time_training.start()

        # train
        epoch_acc, epoch_loss = train(criterion, epoch, model, num_epochs, optimizer, train_loader, train_dataset, progress_bar)
        time_training.end()
        accuracy_training.append(epoch_acc)
        loss_training.append(epoch_loss)

        time_trainings.append(time_training.elapsed_time)

        # Validation
        val_acc, val_loss = validation(criterion, model, valid_loader, valid_dataset, progress_bar)

        loss_validation.append(val_loss)
        accuracy_validation.append(val_acc)

        # Step the scheduler
        scheduler.step(val_acc)

        a_t, l_t, preds, targets = testing(criterion, model, test_loader, test_dataset)

        test_loss.append(l_t)
        test_accuracy.append(a_t)

        # Check early stopping
        if early_stopping(val_loss, model):
            num_epochs = epoch
            break

    total_time.end()

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

    preds_json = {
        "preds": [el.item() for el in preds], 
        "labels": [el.item() for el in targets],
        "model_name": model_name
        }

    update_json_with_key(NAME_JSON_FILE, tb_json)
    update_json_with_key(NAME_PRED_FILE, preds_json)

    print('Saving model: {}'.format(model))
    # save_checkpoint(model, optimizer, '/home/semillerolun/kan/EfficientNetB1Kan-/models/model_checkpoints/' + model_name + MODEL_SAVING_POSTFIX, num_epochs)
    torch.save(model.state_dict(), '/home/semillerolun/kan/model_checkpoints/' + model_name + MODEL_SAVING_POSTFIX)

    return accuracy_validation, loss_validation, time_trainings, test_accuracy[-1], test_loss[-1]


def testing(criterion, model, test_loader, test_dataset):
    model.eval()
    val_loss = 0.0
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
                flattened_output = torch.flatten(outputs, 1)
                # loss = criterion(outputs, labels)
                loss = criterion(flattened_output, labels)
                val_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs, 1)
                val_total += labels.size(0)
                val_correct += (predicted == labels).sum().item()
                preds.extend(predicted)
                targets.extend(labels)
                # Update progress bar
                pbar.set_postfix(val_loss=val_loss / (val_total + 1e-8), val_accuracy=val_correct / val_total)
                pbar.update()

    # return avg loss and accuracy
    accuracy_testing = val_correct / val_total
    loss_testing = val_loss / len(test_dataset)
    return accuracy_testing, loss_testing, preds, targets       


def validation(criterion, model, valid_loader, valid_dataset, progress_bar=True):
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
                flattened_output = torch.flatten(outputs, 1)
                # loss = criterion(outputs, labels)
                loss = criterion(flattened_output, labels)
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


def train(criterion, epoch, model, num_epochs, optimizer, train_loader, train_dataset, progress_bar=True):
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
            flattened_output = torch.flatten(outputs, 1)
            # print(flattened_output.shape)
            # loss = criterion(outputs, labels)
            loss = criterion(flattened_output, labels)
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
