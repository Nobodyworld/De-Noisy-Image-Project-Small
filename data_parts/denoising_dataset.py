import os
from PIL import Image
from torch.utils.data import Dataset

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