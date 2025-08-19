import numpy as np
import torch
import torch.nn as nn
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.metrics import structural_similarity as ssim


def calculate_snr(pred, target):
    """Calculate Signal-to-Noise Ratio"""
    signal_power = np.mean(target ** 2)
    noise_power = np.mean((target - pred) ** 2)
    if noise_power == 0:
        return float('inf')
    return 10 * np.log10(signal_power / noise_power)


def rmse_loss(pred, target):
    """Root Mean Square Error loss function"""
    return torch.sqrt(nn.functional.mse_loss(pred, target))