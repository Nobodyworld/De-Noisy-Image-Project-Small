# train.py
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

############################################################################################################################################################################

def main():
    # Set seed for reproducibility
    torch.manual_seed(42)

    # Set batch size, image dimensions
    batch_size = 12
    img_height = 960
    img_width = 640
    epochs = 48

    # Define the device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

############################################################################################################################################################################

    # Transformation for before (before) images
    before_transform = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor()
    ])

    # Transformation for after (after) images
    after_transform = transforms.Compose([
        transforms.Resize((img_height, img_width)),
        transforms.ToTensor()
    ])

    train_dataset = PairedImageDataset(train_dir, before_transform=before_transform, after_transform=after_transform)
    val_dataset = PairedImageDataset(val_dir, before_transform=before_transform, after_transform=after_transform)
    test_dataset = PairedImageDataset(test_dir, before_transform=before_transform, after_transform=after_transform)

    # Load data
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)

############################################################################################################################################################################

    # Initialize the model
    model = UNet().to(device)
    model_path = 'best_psnr_denocoder_pytorch.pth'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path))
        print("Pre-trained model loaded.")
    else:
        print("No pre-trained model found. Training from scratch.")

############################################################################################################################################################################

    l1_criterion = nn.L1Loss() # L1 loss (mean absolute error) for penalizing the absolute difference between target and predicted images.
    mse_criterion = nn.MSELoss() # Mean squared error loss for penalizing the squared difference between target and predicted images.
    optimizer = torch.optim.Adam(model.parameters(), lr=0.00004, weight_decay=0.001) # Initialize the Adam optimizer with model parameters, learning rate, and weight decay for regularization.
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5, verbose=True) # Learning rate scheduler to reduce the learning rate when a metric has stopped improving, configured to monitor a metric for improvement, reducing LR by a factor of 0.5 after a patience of 5 epochs without improvement.
    best_val_loss = float('inf') # Initialize the best validation loss to infinity for comparison.
    best_val_psnr = 0 # Initialize the best validation PSNR to zero for comparison.
    best_loss_model_state = None # Placeholder for storing the model state with the best loss.
    best_psnr_model_state = None # Placeholder for storing the model state with the best PSNR.
    train_losses = [] # List to store training losses per epoch.
    val_losses = [] # List to store validation losses per epoch.
    train_psnrs = [] # List to store training PSNR values per epoch.
    val_psnrs = [] # List to store validation PSNR values per epoch.
    epochs_since_best_val_psnr = 0  # Counter to track epochs since last improvement in validation PSNR.
    early_stopping_patience = 8  # Number of epochs to wait for an improvement before stopping the training.
    initial_accumulation_steps = 1  # Initial number of gradient accumulation steps.
    step_decrease_interval = 8  # Interval in epochs to decrease accumulation steps.

############################################################################################################################################################################

    for epoch in range(epochs):  # Loop over the dataset multiple times, according to the number of epochs.
        # Calculate the current number of accumulation steps, decreasing it over time based on the epoch.
        accumulation_steps = max(1, initial_accumulation_steps - (epoch // step_decrease_interval))

        model.train()  # Set the model to training mode (enables dropout, batch normalization layers to behave accordingly).
        running_loss = 0.0  # Variable to accumulate losses over the epoch.
        running_psnr = 0.0  # Variable to accumulate PSNR values over the epoch.
        for i, (before_imgs, after_imgs) in enumerate(train_loader):  # Iterate over the training dataset.
            before_imgs, after_imgs = before_imgs.to(device), after_imgs.to(device)  # Move data to the appropriate device (GPU or CPU).
            optimizer.zero_grad()  # Clear the gradients of all optimized tensors.
            outputs = model(before_imgs)  # Forward pass: compute predicted outputs by passing inputs to the model.
            l1_loss = l1_criterion(outputs, after_imgs)  # Compute the L1 loss between predicted and target images.
            mse_loss = mse_criterion(outputs, after_imgs)  # Compute the MSE loss between predicted and target images.
            loss = l1_loss + mse_loss  # Combine L1 and MSE losses.
            loss /= accumulation_steps  # Scale loss down by the number of accumulation steps (for consistent gradient magnitude).
            loss.backward()  # Backward pass: compute gradient of the loss with respect to model parameters.
            if (i + 1) % accumulation_steps == 0: 
                optimizer.step()  # Perform a single optimization step (parameter update).

            running_loss += loss.item() * accumulation_steps  # Accumulate the scaled loss.
            psnr_val = psnr(outputs.detach().cpu(), after_imgs.cpu())  # Compute the PSNR between predicted and target images.
            running_psnr += psnr_val  # Accumulate PSNR values.

        if (i + 1) % accumulation_steps != 0 or i == len(train_loader) - 1:
            optimizer.step()  # Ensure optimizer step is performed if the last batch doesn't align with accumulation steps.

        train_losses.append(running_loss / len(train_loader))  # Calculate and store the average training loss for the epoch.
        train_psnrs.append(running_psnr / len(train_loader))  # Calculate and store the average training PSNR for the epoch.

        print("Epoch [{}/{}], Loss: {:.4f}, PSNR: {:.4f}".format(epoch + 1, epochs, running_loss / len(train_loader), running_psnr / len(train_loader)))
        
        torch.cuda.empty_cache()  # Clear unused memory from GPU to avoid memory leaks.

############################################################################################################################################################################

        model.eval()  # Set the model to evaluation mode (disables dropout and batch normalization layers).
        val_running_loss = 0.0  # Variable to accumulate validation losses over the epoch.
        val_running_psnr = 0.0  # Variable to accumulate PSNR values over the validation dataset.

        with torch.no_grad():  # Disable gradient calculation to reduce memory consumption and speed up computations.
            for i, (before_imgs, after_imgs) in enumerate(val_loader):  # Iterate over the validation dataset.
                before_imgs, after_imgs = before_imgs.to(device), after_imgs.to(device)  # Move data to the appropriate device.

                outputs = model(before_imgs)  # Forward pass: compute predicted outputs by passing inputs to the model.
                l1_loss = l1_criterion(outputs, after_imgs)  # Compute the L1 loss between predicted and target images.
                mse_loss = mse_criterion(outputs, after_imgs)  # Compute the MSE loss between predicted and target images.
                loss = l1_loss + mse_loss  # Combine L1 and MSE losses for the overall loss.
                val_running_loss += loss.item()  # Accumulate the validation loss.
                psnr_val = psnr(outputs.detach().cpu(), after_imgs.cpu())  # Compute the PSNR between predicted and target images.
                val_running_psnr += psnr_val  # Accumulate PSNR values.

        val_losses.append(val_running_loss / len(val_loader))  # Calculate and store the average validation loss for the epoch.
        val_psnrs.append(val_running_psnr / len(val_loader))  # Calculate and store the average validation PSNR for the epoch.
        print("Validation Loss: {:.4f}, PSNR: {:.4f}".format(val_running_loss / len(val_loader), val_running_psnr / len(val_loader)))  # Print the average validation loss and PSNR.

        scheduler.step(val_running_psnr / len(val_loader))  # Adjust the learning rate based on the validation PSNR.

############################################################################################################################################################################

        # Define a flag to indicate improvement in either metric.
        improvement = False

        # Update the best validation loss and corresponding model state if the current validation loss is lower than the best recorded loss.
        if val_running_loss / len(val_loader) < best_val_loss:
            best_val_loss = val_running_loss / len(val_loader)
            best_loss_model_state = model.state_dict()
            epochs_since_best_val_loss = 0  # Reset the counter for epochs since last improvement in validation loss.
            improvement = True
        else:
            epochs_since_best_val_loss += 1  # Increment the counter if no improvement in validation loss.

        # Update the best validation PSNR and corresponding model state if the current validation PSNR is higher than the best recorded PSNR.
        if running_psnr / len(val_loader) > best_val_psnr:
            best_val_psnr = running_psnr / len(val_loader)
            best_psnr_model_state = model.state_dict()
            epochs_since_best_val_psnr = 0  # Reset the counter for epochs since last improvement in validation PSNR.
            improvement = True
        else:
            epochs_since_best_val_psnr += 1  # Increment the counter if no improvement in validation PSNR.

        # If there's no improvement in either metric, consider early stopping.
        if not improvement:
            if epochs_since_best_val_loss >= early_stopping_patience or epochs_since_best_val_psnr >= early_stopping_patience:
                print("Early stopping triggered. Stopping training.")  # Notify that training is being stopped early.
                break  # Exit the training loop.

############################################################################################################################################################################

    model.load_state_dict(best_psnr_model_state)  # Load the model state with the best validation loss for testing.

    model.eval()  # Set the model to evaluation mode for testing.
    test_running_loss = 0.0  # Initialize variable to accumulate test losses.
    test_running_psnr = 0.0  # Initialize variable to accumulate PSNR values on the test set.

    with torch.no_grad():  # Disable gradient computation to save memory and computations during testing.
        for i, (before_imgs, after_imgs) in enumerate(test_loader):  # Iterate over the test dataset.
            before_imgs, after_imgs = before_imgs.to(device), after_imgs.to(device)  # Move data to the device.
            outputs = model(before_imgs)  # Forward pass: compute predictions for the test dataset.
            l1_loss = l1_criterion(outputs, after_imgs)  # Compute L1 loss between predictions and true values.
            mse_loss = mse_criterion(outputs, after_imgs)  # Compute MSE loss between predictions and true values.
            loss = l1_loss + mse_loss  # Aggregate losses for a comprehensive error measure.
            test_running_loss += loss.item()  # Accumulate the test loss.
            psnr_val = psnr(outputs.detach().cpu(), after_imgs.cpu())  # Calculate PSNR between predictions and true values.
            test_running_psnr += psnr_val  # Accumulate PSNR values for the test set.

    # Calculate and print average loss and PSNR for the test dataset.
    print("Test Loss: {:.4f}, PSNR: {:.4f}".format(test_running_loss / len(test_loader), test_running_psnr / len(test_loader)))

    # Save the model states that achieved the best validation loss and PSNR to disk.
    torch.save(best_loss_model_state, 'best_loss_denocoder_pytorch.pth')  # Save model state with the best validation loss.
    torch.save(best_psnr_model_state, 'best_psnr_denocoder_pytorch.pth')  # Save model state with the best validation PSNR.

############################################################################################################################################################################

    # Plot the training and validation loss per epoch to visualize the learning process.
    plt.plot(train_losses, label="Training Loss")  # Plot training loss over epochs.
    plt.plot(val_losses, label="Validation Loss")  # Plot validation loss over epochs.
    plt.plot(train_psnrs, label="Training PSNR")  # Plot training PSNR over epochs.
    plt.plot(val_psnrs, label="Validation PSNR")  # Plot validation PSNR over epochs.
    plt.title("Training and Validation Loss and PSNR")  # Set the title of the plot.
    plt.xlabel("Epoch")  # Label the x-axis as 'Epoch'.
    plt.ylabel("Loss")  # Label the y-axis as 'Loss'.
    plt.legend()  # Show the legend to distinguish between training and validation loss.
    plt.show()

if __name__ == "__main__":
    main()