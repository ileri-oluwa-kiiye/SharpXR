import glob
import numpy as np
import random
import os
import torchvision.transforms.functional as TF
from torch.utils.data import Dataset
from PIL import Image


class XrayDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform

        # Gather all valid image paths
        self.image_paths = sorted([
            p for p in glob.glob(os.path.join(root_dir, "**", "*.*"), recursive=True)
            if p.lower().endswith(('.png', '.jpg', '.jpeg'))
        ])

        if len(self.image_paths) == 0:
            raise ValueError(f"No valid image found in {root_dir}")
    
    def __len__(self):
        return len(self.image_paths)
    
    def add_gaussian_poisson_noise(self, image):
        """Adds both noise to a greyscale image"""
        image = image.astype(np.float32)

        # Apply poisson noise
        poisson_scale = random.uniform(50, 300)  # Vary the poisson noise per image
        photon_image = image/255.0 * poisson_scale
        noisy_poisson = np.random.poisson(photon_image)
        poisson_noise = (noisy_poisson/poisson_scale * 255.0)

        # Apply Gaussian noise
        gaussian_std = random.uniform(5, 25)
        gaussian_noise = np.random.normal(0, gaussian_std, image.shape)

        # Combine both noises
        noisy_combined = poisson_noise + gaussian_noise
        noisy_combined = np.clip(noisy_combined, 0, 255)

        return noisy_combined.astype(np.uint8)
    
    def __getitem__(self, idx):
        img_path = self.image_paths[idx]

        try:
            image = Image.open(img_path).convert('L')
        except Exception as e:
            print(f"Error loading image {img_path}: {str(e)}")
            image = Image.new('L', (256, 256), 0)  # Fallback on a black image
        
        # Resize all images to 256x256 before processing
        image = image.resize((256, 256))

        # Apply augmentation before adding noise
        if random.random() > 0.5:
            image = TF.hflip(image)
        if random.random() > 0.5:
            angle = random.uniform(-15, 15)
            image = TF.rotate(image, angle)
        brightness_factor = random.uniform(0.9, 1.1)
        contrast_factor = random.uniform(0.9, 1.1)
        image = TF.adjust_brightness(image, brightness_factor)
        image = TF.adjust_contrast(image, contrast_factor)

        # Convert pil image to numpy for noise application
        image_np = np.array(image, dtype=np.float32)
        noisy_np = self.add_gaussian_poisson_noise(image_np)

        # Convert both images to pytorch tensors
        clean_tensor = TF.to_tensor(image)  # [0, 1]
        noisy_tensor = TF.to_tensor(Image.fromarray(noisy_np))  # [0, 1]

        return noisy_tensor, clean_tensor