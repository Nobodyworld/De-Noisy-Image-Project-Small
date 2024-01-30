import os
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from PIL import Image
import matplotlib.pyplot as plt
import random


class UNet(nn.Module):
    def __init__(self):
        super(UNet, self).__init__()
        # Encoder
        self.enc_conv1 = nn.Sequential(nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True), nn.Conv2d(16, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True))
        self.pool1 = nn.MaxPool2d(2, 2)
        self.res_enc1 = nn.Sequential(nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(inplace=True), nn.BatchNorm2d(16), nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(inplace=True), nn.BatchNorm2d(16))
        self.enc_conv2 = nn.Sequential(nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.pool2 = nn.MaxPool2d(2, 2)
        self.res_enc2 = nn.Sequential(nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(inplace=True), nn.BatchNorm2d(32), nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(inplace=True), nn.BatchNorm2d(32))
        self.enc_conv3 = nn.Sequential(nn.Conv2d(32, 48, 3, padding=1), nn.BatchNorm2d(48), nn.ReLU(inplace=True), nn.Conv2d(48, 48, 3, padding=1), nn.BatchNorm2d(48), nn.ReLU(inplace=True))
        self.pool3 = nn.MaxPool2d(2, 2)
        self.res_enc3 = nn.Sequential(nn.Conv2d(48, 48, 3, padding=1), nn.ReLU(inplace=True), nn.BatchNorm2d(48), nn.Conv2d(48, 48, 3, padding=1), nn.ReLU(inplace=True), nn.BatchNorm2d(48))
        self.enc_conv4 = nn.Sequential(nn.Conv2d(48, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))        
        self.pool4 = nn.MaxPool2d(2, 2)    

        # Mid
        self.mid_conv = nn.Sequential(nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        # Decoder
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) # Upsampling to 120x80
        self.dec_conv4 = nn.Sequential(nn.Conv2d(128, 48, 3, padding=1), nn.BatchNorm2d(48), nn.ReLU(inplace=True), nn.Conv2d(48, 48, 3, padding=1), nn.BatchNorm2d(48), nn.ReLU(inplace=True))
        self.up3 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) # Upsampling to 240x160
        self.dec_conv3 = nn.Sequential(nn.Conv2d(96, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(inplace=True))
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) # Upsampling to 480x320
        self.dec_conv2 = nn.Sequential(nn.Conv2d(64, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True), nn.Conv2d(16, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True))
        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True) # Upsampling to 960x640
        self.dec_conv1 = nn.Sequential(nn.Conv2d(32, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True), nn.Conv2d(16, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(inplace=True))

        # Output layer
        self.out_conv = nn.Sequential(nn.Conv2d(16, 3, 1), nn.Sigmoid())


    def forward(self, x):
        # Encoder
        x1 = self.enc_conv1(x)
        x2 = self.pool1(x1)
        x2 = x2 + self.res_enc1(x2)
        x3 = self.enc_conv2(x2)
        x4 = self.pool2(x3)
        x4 = x4 + self.res_enc2(x4)
        x5 = self.enc_conv3(x4)
        x6 = self.pool3(x5)
        x6 = x6 + self.res_enc3(x6)
        x7 = self.enc_conv4(x6)
        x8 = self.pool4(x7)

        # Mid
        x_mid = self.mid_conv(x8)

        # Decoder
        x_up4 = self.up4(x_mid)
        x_dec4 = self.dec_conv4(torch.cat([x_up4, x7], dim=1))
        x_up3 = self.up3(x_dec4)
        print(x_up3.shape, x6.shape)
        x_dec3 = self.dec_conv3(torch.cat([x_up3, x6], dim=1))
        x_up2 = self.up2(x_dec3)
        x_dec2 = self.dec_conv2(torch.cat([x_up2, x5], dim=1))
        x_up1 = self.up1(x_dec2)
        x_dec1 = self.dec_conv1(torch.cat([x_up1, x4], dim=1))

        # Output layer
        x_out = self.out_conv(x_dec1)
        return x_out
        
class DenoisingDataset(Dataset):
    def __init__(self, noisy_dir, clean_dir, noisy_transform=None, clean_transform=None):
        self.noisy_dir = noisy_dir
        self.clean_dir = clean_dir
        self.noisy_transform = noisy_transform
        self.clean_transform = clean_transform

        self.noisy_filenames = os.listdir(noisy_dir)
        self.clean_filenames = os.listdir(clean_dir)

    def __len__(self):
        return len(self.noisy_filenames)

    def __getitem__(self, idx):
        noisy_img = Image.open(os.path.join(self.noisy_dir, self.noisy_filenames[idx])).convert("RGB")
        clean_img = Image.open(os.path.join(self.clean_dir, self.clean_filenames[idx])).convert("RGB")

        if self.noisy_transform:
            noisy_img = self.noisy_transform(noisy_img)
        if self.clean_transform:
            clean_img = self.clean_transform(clean_img)

        return noisy_img, clean_img
    
def psnr(pred, target, max_pixel=1.0, eps=1e-10, reduction='mean'):
    mse = torch.mean((pred - target) ** 2, dim=(1, 2, 3))
    psnr_val = 20 * torch.log10(max_pixel / torch.sqrt(mse + eps))

    if reduction == 'mean':
        return psnr_val.mean().cpu().item()
    elif reduction == 'none':
        return psnr_val.cpu().numpy()
    else:
        raise ValueError("Invalid reduction mode. Supported modes are 'mean' and 'none'.")
    
class RandomColorJitterWithRandomFactors(torchvision.transforms.ColorJitter):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, p=0.25):
        super().__init__(brightness, contrast, saturation, hue)
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            brightness_factor = random.uniform(0.90, 1.10)
            contrast_factor = random.uniform(0.90, 1.10)
            saturation_factor = random.uniform(0.90, 1.10)
            hue_factor = random.uniform(-0.1, 0.1)

            jitter = torchvision.transforms.ColorJitter(
                brightness=brightness_factor,
                contrast=contrast_factor,
                saturation=saturation_factor,
                hue=(hue_factor, hue_factor)
            )
            return jitter(img)
        else:
            return img
   
        
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
    train_dir = './train'
    test_dir = './test'
    val_dir = './val'

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