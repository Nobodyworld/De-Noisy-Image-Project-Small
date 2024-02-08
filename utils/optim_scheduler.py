# /utils/optim_scheduler.py
import torch.optim as optim

def setup_optimizer_scheduler(model, config):
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'], weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', factor=0.5, patience=5, verbose=True)
    return optimizer, scheduler