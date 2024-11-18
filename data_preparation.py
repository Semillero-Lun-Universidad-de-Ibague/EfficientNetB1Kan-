import torch, random, cv2
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms


def prepare_dataset(batch_size, path, grayscale=False, toy=False):

    train_dir = path + "/data/Training"
    test_dir = path + "/data/Testing"

    # sizeof_picture = 240
    if grayscale:
        sizeof_picture = 32
        # Define transformations
        train_transform = transforms.Compose([
            transforms.RandomRotation(degrees=10),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=0, translate=(0, 0.2)),
            transforms.Resize((sizeof_picture, sizeof_picture)),
            transforms.Grayscale(),
            transforms.ToTensor()
        ])

        valid_transform = transforms.Compose([
            transforms.Resize((sizeof_picture, sizeof_picture)),
            transforms.Grayscale(),
            transforms.ToTensor()
        ])

        test_transform = transforms.Compose([
            transforms.Resize((sizeof_picture, sizeof_picture)),
            transforms.Grayscale(),
            transforms.ToTensor()
        ])
    else:
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
    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transform)

    # Split the training data into training and validation sets
    num_train = len(train_dataset)
    split = int(0.8 * num_train)  # 80% training, 20% validation
    train_size = split
    valid_size = num_train - split

    train_dataset, valid_dataset = random_split(train_dataset, [train_size, valid_size])

    if toy:
        train_dataset = Subset(train_dataset, np.arange(80))
        valid_dataset = Subset(valid_dataset, np.arange(20))
        test_dataset = Subset(test_dataset, np.arange(20))

    # Create DataLoader
    # set manual seed to make everything more reproducible

    # Set a seed for reproducibility
    seed = 42
    torch.manual_seed(seed)  # Ensure the random behavior is consistent

    # Create a generator for the DataLoader
    g = torch.Generator()
    g.manual_seed(seed)  # Ensures DataLoader shuffling is reproducible
    train_loader = DataLoader(train_dataset, batch_size=batch_size, generator=g, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, generator=g, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, generator=g, shuffle=True)
    "train: ", len(train_dataset), "Valid: ", len(valid_dataset), "test: ", len(test_dataset)

    return (train_dataset, valid_dataset, test_dataset), (train_loader, valid_loader, test_loader)


def transform_dataset(datasets: list, device: str, ext: bool=True) -> dict:

    new_dataset = {'train_input': [], 'valid_input': [], 'test_input': [], 'train_label': [], 'valid_label': [], 'test_label': []}
    for i, dataset in enumerate(datasets):
        if ext:
            dataset = preprocess_data_for_external_kan(dataset)
        for element in dataset:
            if i == 0:
                new_dataset['train_input'].append(element[0])
                new_dataset['train_label'].append(torch.Tensor([element[1]]))
            elif i == 1:
                new_dataset['valid_input'].append(element[0])
                new_dataset['valid_label'].append(torch.Tensor([element[1]]))
            elif i == 2:
                new_dataset['test_input'].append(element[0])
                new_dataset['test_label'].append(torch.Tensor([element[1]]))

    for key, value in new_dataset.items():
        new_dataset[key] = torch.stack(value).to(device)


    return new_dataset


def preprocess_data_for_external_kan(data):

    dataset = []
    for img, label in data:
        img = cv2.resize(np.array(img), (32, 32))
        img = torch.from_numpy(img)
        img = img.flatten() / 255.0
        dataset.append((img, label))

    return dataset


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


if __name__ == "__main__":

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(0)
    d, l = prepare_dataset(8, '~/kan/', toy=True)
    transform_dataset(d, device, ext=True)