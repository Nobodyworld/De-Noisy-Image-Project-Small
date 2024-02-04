# Description: Custom transforms for data augmentation

import random
import torchvision

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