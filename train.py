import json
import os
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from models.unet import UNet
from utils.pairedimage_dataset import PairedImageDataset
from utils.metrics import psnr

# Load configuration from config.json
try:
    with open('config.json', 'r') as config_file:
        config = json.load(config_file)
except FileNotFoundError:
    print("Error: config.json file not found.")
    exit(1)
except json.JSONDecodeError:
    print("Error: Failed to decode config.json.")
    exit(1)

# Accessing data directories and model path from the config
train_dir = os.path.join(config['directories']['data']['train'])
test_dir = os.path.join(config['directories']['data']['test'])
val_dir = os.path.join(config['directories']['data']['val'])
model_save_path = os.path.join(config['directories']['model']['model_path']) # Updated for saving models

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

    # Transformation for before (before) images
    before_transform = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Transformation for after (after) images
    after_transform = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = PairedImageDataset(train_dir, before_transform=before_transform, after_transform=after_transform)
    val_dataset = PairedImageDataset(val_dir, before_transform=before_transform, after_transform=after_transform)
    test_dataset = PairedImageDataset(test_dir, before_transform=before_transform, after_transform=after_transform)

    # Load data
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)

    # Initialize the model
    model = UNet().to(device)
    model_path = os.path.join(model_save_path, 'best_psnr_denocoder_pytorch.pth') # Use model_save_path for loading
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
        for i, (before_imgs, after_imgs) in enumerate(train_loader):
            before_imgs, after_imgs = before_imgs.to(device), after_imgs.to(device)
            optimizer.zero_grad()
            outputs = model(before_imgs)
            l1_loss = l1_criterion(outputs, after_imgs)
            mse_loss = mse_criterion(outputs, after_imgs)
            loss = l1_loss + mse_loss
            loss /= accumulation_steps
            loss.backward()
            if (i + 1) % accumulation_steps == 0: 
                optimizer.step()  # update model parameters every accumulation_steps mini-batches

            # Update running_loss and running_psnr for every mini-batch
            running_loss += loss.item() * accumulation_steps
            psnr_val = psnr(outputs.detach().cpu(), after_imgs.cpu())
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
            for i, (before_imgs, after_imgs) in enumerate(val_loader):
                before_imgs, after_imgs = before_imgs.to(device), after_imgs.to(device)

                outputs = model(before_imgs)
                l1_loss = l1_criterion(outputs, after_imgs)
                mse_loss = mse_criterion(outputs, after_imgs)
                loss = l1_loss + mse_loss  # Combine L1 and L2 loss
                val_running_loss += loss.item()
                psnr_val = psnr(outputs.detach().cpu(), after_imgs.cpu())
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
        for i, (before_imgs, after_imgs) in enumerate(test_loader):
            before_imgs, after_imgs = before_imgs.to(device), after_imgs.to(device)
            outputs = model(before_imgs)
            l1_loss = l1_criterion(outputs, after_imgs)
            mse_loss = mse_criterion(outputs, after_imgs)
            loss = l1_loss + mse_loss  # Combine L1 and L2 loss
            test_running_loss += loss.item()
            psnr_val = psnr(outputs.detach().cpu(), after_imgs.cpu())
            test_running_psnr += psnr_val
            
    print("Test Loss: {:.4f}, PSNR: {:.4f}".format(test_running_loss / len(test_loader), test_running_psnr / len(test_loader)))

    plt.plot(train_losses, label="Training Loss")
    plt.plot(val_losses, label="Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()

    # When saving models, use the model_save_path
    torch.save(best_loss_model_state, os.path.join(model_save_path, 'best_loss_denocoder_pytorch.pth'))
    torch.save(best_psnr_model_state, os.path.join(model_save_path, 'best_psnr_denocoder_pytorch.pth'))

if __name__ == "__main__":
    main()