from model_parts.unet import UNet
from data_parts.denoising_dataset import DenoisingDataset
from config_loader import load_config
import matplotlib.pyplot as plt
import torch.nn as nn
from utils.metrics import psnr
from torch.utils.data import DataLoader
from torchvision import transforms
import torch
import os

# Accessing data directories and model path from the config
config = load_config()
train_dir = os.path.join(config['directories']['data']['train'])
test_dir = os.path.join(config['directories']['data']['test'])
val_dir = os.path.join(config['directories']['data']['val'])
model_path = os.path.join(config['directories']['model']['model_path'])

# Accessing specific data paths
train_clean_path = os.path.join(config['data_paths']['train']['clean'])
train_noisy_path = os.path.join(config['data_paths']['train']['noisy'])
# Similar approach for val and test paths if needed


def main():
    # Set seed for reproducibility
    torch.manual_seed(42)

    # Set batch size, image dimensions
    batch_size = 6
    img_height = 960
    img_width = 640
    epochs = 48

    # Define the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Define data directories
    train_dir = './data/train'
    test_dir = './data/test'
    val_dir = './data/val'

    noisy_transform = transforms.Compose([
        transforms.Resize((img_height, img_width), antialias=None),            
        #RandomColorJitterWithRandomFactors(p=0.25),
        transforms.ToTensor(), 
        # Add any other noisy-specific transformationss
    ])

    clean_transform = transforms.Compose([
        transforms.Resize((img_height, img_width), antialias=None),
        transforms.ToTensor(),
        # Add any other clean-specific transformations
    ])

    train_dataset = DenoisingDataset(os.path.join(train_dir, 'noisy'), os.path.join(train_dir, 'clean'), noisy_transform=noisy_transform, clean_transform=clean_transform)
    val_dataset = DenoisingDataset(os.path.join(val_dir, 'noisy'), os.path.join(val_dir, 'clean'), noisy_transform=noisy_transform, clean_transform=clean_transform)
    test_dataset = DenoisingDataset(os.path.join(test_dir, 'noisy'), os.path.join(test_dir, 'clean'), noisy_transform=noisy_transform, clean_transform=clean_transform)

    # Load data
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)

    model = UNet().to(device)
    model_path = 'best_psnr_denocoder_pytorch.pth'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print("Pre-trained model loaded.")
    else:
        print("No pre-trained model found. Training from scratch.")

    # Initialize the model, loss, and optimizer
    l1_criterion = nn.L1Loss()
    mse_criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00004, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True)
    best_val_loss = float('inf')
    best_val_psnr = 0
    best_loss_model_state = None
    best_psnr_model_state = None
    train_losses = []
    val_losses = []
    train_psnrs = []
    val_psnrs = []

    epochs_since_best_val_psnr = 0

    early_stopping_patience = 8
    initial_accumulation_steps = 1
    step_decrease_interval = 8

    for epoch in range(epochs):
        accumulation_steps = max(1, initial_accumulation_steps - (epoch // step_decrease_interval)) 

        # Train the model
        model.train()
        running_loss = 0.0
        running_psnr = 0.0
        for i, (noisy_imgs, clean_imgs) in enumerate(train_loader):
            noisy_imgs, clean_imgs = noisy_imgs.to(device), clean_imgs.to(device)
            optimizer.zero_grad()
            outputs = model(noisy_imgs)
            l1_loss = l1_criterion(outputs, clean_imgs)
            mse_loss = mse_criterion(outputs, clean_imgs)
            loss = l1_loss + mse_loss
            loss /= accumulation_steps
            loss.backward()
            if (i + 1) % accumulation_steps == 0: 
                optimizer.step()  # update model parameters every accumulation_steps mini-batches

            # Update running_loss and running_psnr for every mini-batch
            running_loss += loss.item() * accumulation_steps
            psnr_val = psnr(outputs.detach().cpu(), clean_imgs.cpu())
            running_psnr += psnr_val

        if (i + 1) % accumulation_steps != 0 or i == len(train_loader) - 1:
            optimizer.step()  # update model parameters for remaining mini-batches

        train_losses.append(running_loss / len(train_loader))
        train_psnrs.append(running_psnr / len(train_loader))  # Add this line to store the average PSNR for the epoch

        print("Epoch [{}/{}], Loss: {:.4f}, PSNR: {:.4f}".format(epoch + 1, epochs, running_loss / len(train_loader), running_psnr / len(train_loader)))
        
        torch.cuda.empty_cache()

        # Evaluate the model on validation data
        model.eval()
        val_running_loss = 0.0
        running_psnr = 0.0

        with torch.no_grad():
            for i, (noisy_imgs, clean_imgs) in enumerate(val_loader):
                noisy_imgs, clean_imgs = noisy_imgs.to(device), clean_imgs.to(device)

                outputs = model(noisy_imgs)
                l1_loss = l1_criterion(outputs, clean_imgs)
                mse_loss = mse_criterion(outputs, clean_imgs)
                loss = l1_loss + mse_loss  # Combine L1 and L2 loss
                val_running_loss += loss.item()
                psnr_val = psnr(outputs.detach().cpu(), clean_imgs.cpu())
                running_psnr += psnr_val

        val_losses.append(val_running_loss / len(val_loader))
        val_psnrs.append(running_psnr / len(val_loader))
        print("Validation Loss: {:.4f}, PSNR: {:.4f}".format(val_running_loss / len(val_loader), running_psnr / len(val_loader)))

        # Update the scheduler after each epoch
        scheduler.step(running_psnr / len(val_loader))

        # Update best validation PSNR and reset patience counter
        if val_running_loss / len(val_loader) < best_val_loss:
            best_val_loss = val_running_loss / len(val_loader)
            best_loss_model_state = model.state_dict()
            epochs_since_best_val_loss = 0
        else:
            epochs_since_best_val_loss += 1

        if running_psnr / len(val_loader) > best_val_psnr:
            best_val_psnr = running_psnr / len(val_loader)
            best_psnr_model_state = model.state_dict()
            epochs_since_best_val_psnr = 0
        else:
            epochs_since_best_val_psnr += 1
        if epochs_since_best_val_loss >= early_stopping_patience and epochs_since_best_val_psnr >= early_stopping_patience:
            print("Early stopping triggered. Stopping training.")
            break


    # Load the best model state for testing
    model.load_state_dict(best_loss_model_state)

    # Test the model
    model.eval()
    test_running_loss = 0.0
    test_running_psnr = 0.0
    with torch.no_grad():
        for i, (noisy_imgs, clean_imgs) in enumerate(test_loader):
            noisy_imgs, clean_imgs = noisy_imgs.to(device), clean_imgs.to(device)
            outputs = model(noisy_imgs)
            l1_loss = l1_criterion(outputs, clean_imgs)
            mse_loss = mse_criterion(outputs, clean_imgs)
            loss = l1_loss + mse_loss  # Combine L1 and L2 loss
            test_running_loss += loss.item()
            psnr_val = psnr(outputs.detach().cpu(), clean_imgs.cpu())
            test_running_psnr += psnr_val
            
    print("Test Loss: {:.4f}, PSNR: {:.4f}".format(test_running_loss / len(test_loader), test_running_psnr / len(test_loader)))

    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    # Save the trained model and the best model to files
    torch.save(best_loss_model_state, 'best_loss_denocoder_pytorch.pth')
    torch.save(best_psnr_model_state, 'best_psnr_denocoder_pytorch.pth')

if __name__ == "__main__":
    main()