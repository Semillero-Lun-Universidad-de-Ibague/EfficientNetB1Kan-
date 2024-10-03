from pathlib import Path
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms


def prepare_dataset(batch_size):

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
    # global train_dataset
    # global valid_dataset
    # global test_dataset
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
    # global train_loader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    # global valid_loader
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False)
    # global test_loader
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    "train: ", len(train_dataset), "Valid: ", len(valid_dataset), "test: ", len(test_dataset)

    return train_loader, valid_loader, test_loader