# /utils/data_loading.py
from torchvision import transforms
from torch.utils.data import DataLoader
from utils.pairedimage_dataset import PairedImageDataset

def get_transforms(config):
    """Generate torchvision transforms based on config."""
    img_height = config['training']['img_height']
    img_width = config['training']['img_width']

    transform = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor(),
    ])
    return transform

def get_dataloaders(config):
    """Create DataLoader for training, validation, and testing datasets."""
    transform = get_transforms(config)

    # Initialize datasets
    train_dataset = PairedImageDataset(config['directories']['data']['train'], transform=transform)
    val_dataset = PairedImageDataset(config['directories']['data']['val'], transform=transform)
    test_dataset = PairedImageDataset(config['directories']['data']['test'], transform=transform)

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=config['training']['batch_size'], shuffle=True, pin_memory=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=config['training']['batch_size'], shuffle=False, pin_memory=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=config['training']['batch_size'], shuffle=False, pin_memory=True, num_workers=4)

    return train_loader, val_loader, test_loader