from workbench.model import UNet
import torch
import json

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

# Extract directories and paths from the config
model_save_path = config['paths']['model']['save_path']

# Load the trained model
model = UNet()
# Ensure the model is loaded to CPU if CUDA is not available
try:
    model.load_state_dict(torch.load(model_save_path))
except FileNotFoundError:
    print(f"Error: Model file {model_save_path} not found.")
    exit(1)
except RuntimeError as e:
    print(f"Error loading the model: {e}")
    exit(1)

total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Number of trainable parameters: {total_params}")