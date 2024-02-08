# /utils/chekpointing.py
import torch

def save_checkpoint(state, filename='best_psnr_denocoder_pytorch.pth'):
    torch.save(state, filename)

def load_checkpoint(model, filename='best_psnr_denocoder_pytorch.pth'):
    model.load_state_dict(torch.load(filename))