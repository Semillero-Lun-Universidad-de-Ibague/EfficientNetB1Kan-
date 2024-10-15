from pathlib import Path
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def prepare_dataset(batch_size, path):

    train_dir = path + "/data/Training"
    test_dir = path + "/data/Testing"

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

    print(f"batch size: {batch_size}")
    # Create DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    "train: ", len(train_dataset), "Valid: ", len(valid_dataset), "test: ", len(test_dataset)

    return (train_dataset, valid_dataset, test_dataset), (train_loader, valid_loader, test_loader)