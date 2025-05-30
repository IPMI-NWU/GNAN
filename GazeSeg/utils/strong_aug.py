import random

import torch
import torch.nn.functional as F
from torchvision import transforms
import numpy as np


def RandomBrightnessContrast(img, brightness_limit=0.5, contrast_limit=0.5, p=1.0):
    output = torch.zeros_like(img)
    threshold = 0.5
    brightness_min = 0.2
    contrast_min = 0.1

    for i in range(output.shape[0]):
        img_min, img_max = torch.min(img[i]), torch.max(img[i])

        output[i] = (img[i] - img_min) / (img_max - img_min) * 255.0
        if random.random() < p:
            brightness = 1.0 + random.uniform(-brightness_limit, brightness_limit)
            if brightness > 1.0:
                brightness = max(1.0 + brightness_min, brightness)
            else:
                brightness = min(1.0 - brightness_min, brightness)
            output[i] = torch.clamp(output[i] * brightness, 0., 255.)

            contrast = 0.0 + random.uniform(-contrast_limit, contrast_limit)
            if contrast > 0.0:
                contrast = max(0.0 + contrast_min, contrast)
            else:
                contrast = min(0.0 - contrast_min, contrast)
            output[i] = torch.clamp(output[i] + (output[i] - threshold * 255.0) * contrast, 0., 255.)

        output[i] = output[i] / 255.0 * (img_max - img_min) + img_min
    return output


def add_gaussian_noise(tensor, device, mean=0.0, std=0.1):
    noise = torch.normal(mean=mean, std=std, size=tensor.size()).to(device)
    noisy_tensor = tensor + noise
    return noisy_tensor


def apply_strong_augmentations(tensor, device):
    tensor = RandomBrightnessContrast(tensor)
    tensor = add_gaussian_noise(tensor, device)
    return tensor
