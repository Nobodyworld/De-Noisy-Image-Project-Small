# /utils/chekpointing.py
import torch

def save_checkpoint(state_dict, filename='best_psnr_denocoder_pytorch.pth'):
    """
    Saves the model's state dictionary to a file.

    Parameters:
    - state_dict: The model's state dictionary.
    - filename: Path to the file where the state dictionary should be saved.
    """
    torch.save(state_dict, filename)

def load_checkpoint(model, filename, device):
    """
    Loads the model's state dictionary from a file.

    Parameters:
    - model: The model instance to load the state dictionary into.
    - filename: Path to the file from which the state dictionary should be loaded.
    - device: The device to map the loaded state dictionary to.
    """
    checkpoint = torch.load(filename, map_location=device)
    if "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)
