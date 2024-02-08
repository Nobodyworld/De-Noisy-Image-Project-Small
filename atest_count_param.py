from models.unet import UNet
import torch

# Initialize the model
model = UNet()

# Load the state dictionary from the saved model
state_dict = torch.load("best_psnr_denocoder_pytorch.pth")

# Load the state dictionary into the model
model.load_state_dict(state_dict)

# Now you can count the trainable parameters
total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of trainable parameters: {total_params}")
