from utils.config_manager import load_config
from utils.data_loading import get_dataloaders
from utils.training import train_one_epoch, validate
from utils.testing import test
from utils.checkpointing import save_checkpoint, load_checkpoint
from utils.optim_scheduler import setup_optimizer_scheduler
from utils.plotting import plot_metrics
from models.unet import UNet
import torch
import torch.nn as nn
import os

def main():
    config = load_config('config/config.json')
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_loader, val_loader, test_loader = get_dataloaders(config['directories']['data'])
    model = UNet(input_channels=config['model']['input_shape'][-1], 
                 output_channels=config['model']['output_channels']).to(device)
    optimizer, scheduler = setup_optimizer_scheduler(model, config)
    
    l1_criterion = nn.L1Loss().to(device)
    mse_criterion = nn.MSELoss().to(device)
    
    model_path = config['directories']['models'] + '/best_psnr_model.pth'
    if os.path.exists(model_path):
        load_checkpoint(model, model_path)
    
    # Initialize best metrics and early stopping counters
    best_val_loss = float('inf')
    best_val_psnr = 0
    epochs_since_improvement = 0  # Unified counter for simplicity
    
    train_losses, val_losses, train_psnrs, val_psnrs = [], [], [], []
    
    for epoch in range(config['training']['epochs']):
        print(f"Epoch {epoch+1}/{config['training']['epochs']}")
        
        train_loss, train_psnr = train_one_epoch(model, device, train_loader, optimizer, l1_criterion, mse_criterion, config['training'])
        val_loss, val_psnr = validate(model, device, val_loader, l1_criterion, mse_criterion, config['training'])
        
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_psnrs.append(train_psnr)
        val_psnrs.append(val_psnr)
        
        scheduler.step(val_loss)
        
        # Check for improvement in val_loss for checkpointing and early stopping
        if val_loss < best_val_loss or val_psnr > best_val_psnr:
            best_val_loss = min(val_loss, best_val_loss)
            best_val_psnr = max(val_psnr, best_val_psnr)
            epochs_since_improvement = 0
            save_checkpoint({'state_dict': model.state_dict()}, model_path)
            print("Saved improved model checkpoint.")
        else:
            epochs_since_improvement += 1
        
        if epochs_since_improvement >= config['training']['early_stopping_patience']:
            print("Early stopping triggered.")
            break
        
    test_loss, test_psnr = test(model, device, test_loader, l1_criterion, mse_criterion)
    plot_metrics(train_losses, val_losses, train_psnrs, val_psnrs)
    
if __name__ == "__main__":
    main()
