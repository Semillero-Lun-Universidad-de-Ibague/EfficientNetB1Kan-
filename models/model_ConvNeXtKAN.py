import torch, random, os, sys
import torch.nn as nn
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
# Hope you don't be imprisoned by legacy Python code :)
from pathlib import Path
from tqdm import tqdm
from torch.utils.data import DataLoader, random_split
from tensorflow.keras.preprocessing.image import load_img
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torchsummary import summary
from kcn import ConvNeXtKAN

sys.path.append('..')
from common import update_json_with_key, save_checkpoint, load_checkpoint
from utils import EarlyStopping

## from torch.utils.tensorboard import SummaryWriter
## from torchvision.transforms import functional as F

## import torchvision


files_path_dict = {}

# `cwd`: current directory is straightforward
cwd = Path.cwd()

train_dir = str(cwd) + "/data/Training"
test_dir = str(cwd) + "/data/Testing"

data_training = {
    'glioma_tumor': [],
    'meningioma_tumor': [],
    'no_tumor': [],
    'pituitary_tumor': []
}
for i in data_training.keys():
    for j in os.listdir(train_dir + "/" + i):
        data_training[i].append(f"{train_dir}/{i}/{j}")

data_testing = {
    'glioma_tumor': [],
    'meningioma_tumor': [],
    'no_tumor': [],
    'pituitary_tumor': []
}
for i in data_testing.keys():
    for j in os.listdir(test_dir + "/" + i):
        data_testing[i].append(f"{test_dir}/{i}/{j}")

len_train = {}
for i in data_training.keys():
    len_train[i] = len(data_training[i])
len_testing = {}
for i in data_testing.keys():
    len_testing[i] = len(data_testing[i])

print({"training": len_train, "testing": len_testing})

plt.figure(figsize=(17, 17))
index = 0
for c in data_training.keys():
    random.shuffle(data_training[c])
    path_list = data_training[c][:5]

    for i in range(1, 5):
        index += 1
        plt.subplot(4, 4, index)
        plt.imshow(load_img(path_list[i]))
        plt.title(c)

# Data Augmentation
# Define transformations
train_transform = transforms.Compose([
    transforms.RandomRotation(degrees=10),
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=0, translate=(0, 0.2)),
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

valid_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

test_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor()
])

# Load datasets
train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transform)

# Split the training data into training and validation sets
num_train = len(train_dataset)
split = int(0.8 * num_train)  # 80% training, 20% validation
train_size = split
valid_size = num_train - split

train_dataset, valid_dataset = random_split(train_dataset, [train_size, valid_size])

# Create DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=32, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
"train: ", len(train_dataset), "Valid: ", len(valid_dataset), "test: ", len(test_dataset)

# Instancujte model
# Initiate model
model = ConvNeXtKAN()

if torch.cuda.is_available():
    model.cuda()

# Vytiskněte souhrn modelu
# Print a summary of the model
def print_model_summary(model, input_size=(3, 128, 128)):
    summary(model, input_size=input_size)


print_model_summary(model)

# Define the optimizer
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Define the loss function
criterion = nn.CrossEntropyLoss()

# Define the scheduler
scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.3, patience=2, verbose=True)

# Initialize early stopping
early_stopping = EarlyStopping(patience=5, verbose=True)

accuracy_training = []
accuracy_testing = []
loss_validation = []
loss_training = []

## try:
##     num_epochs
## except NameError:
num_epochs = 5

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    epoch_acc = 0

    # Training loop with progress bar
    with tqdm(total=len(train_loader), desc=f'Epoch {epoch + 1}/{num_epochs}', unit='batch') as pbar:
        for inputs, labels in train_loader:

            if torch.cuda.is_available():
                inputs = inputs.to(torch.device('cuda'))
                labels = labels.to(torch.device('cuda'))

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

    # logging 
    accuracy_training.append(epoch_acc)
    loss_training.append(epoch_loss)

    epoch_loss = running_loss / len(train_dataset)
    epoch_acc = correct / total
    print(f'Epoch {epoch + 1}/{num_epochs} - Loss: {epoch_loss:.4f} - Accuracy: {epoch_acc:.4f}')

    accuracy_training.append(epoch_acc)
    loss_training.append(epoch_loss)

    # Validation
    model.eval()
    val_loss = 0.0
    val_correct = 0
    val_total = 0

    # Validation loop with progress bar
    with tqdm(total=len(valid_loader), desc='Validation', unit='batch') as pbar:
        with torch.no_grad():
            for inputs, labels in valid_loader:

                if torch.cuda.is_available():
                    inputs = inputs.to(torch.device('cuda'))
                    labels = labels.to(torch.device('cuda'))

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

    loss_validation.append(val_loss)
    accuracy_testing.append(val_acc)

    # Step the scheduler
    scheduler.step(val_acc)

    # Check early stopping
    if early_stopping(val_loss, model):
        num_epochs = epoch
        break

update_json_with_key('data.json', model_name, num_epochs, accuracy_training, accuracy_testing, loss_training,
                     loss_validation)

save_checkpoint(model, optimizer, model_name + "_checkpoint.pth", num_epochs)
model = ConvNeXtKAN()
optimizer = optim.Adam(model.parameters(), lr=0.001)

model, optimizer, num_epochs = load_checkpoint(model, optimizer, model_name + "_checkpoint.pth", )

fig, ax = plt.subplots(1, 2, figsize=(15, 5))

epochs = np.arange(0, num_epochs, num_epochs / len(accuracy_training))

ax[0].plot(epochs, accuracy_training, 'g', label='Training Accuracy')

epochs = np.arange(0, num_epochs, num_epochs / len(accuracy_testing))

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

