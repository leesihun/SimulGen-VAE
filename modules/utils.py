import os
import glob
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random
import torch

def get_latest_file(directory):
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory {directory} does not exist")
    
    files = glob.glob(os.path.join(directory, '*'))
    if not files:
        raise FileNotFoundError(f"No files found in {directory}")

    latest_file = max(files, key=os.path.getmtime)
    return latest_file

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
import time
class MyBaseDataset(Dataset):
    def __init__(self, x_data, load_all):
        print('Loading data...')
        if load_all:
            print('🚀 SPEED OPTIMIZED: Preparing data for GPU loading')
            # Keep data on CPU initially - will be moved to GPU in training loop for multi-worker compatibility
            self.x_data = torch.tensor(x_data, dtype=torch.float16)  # Remove .to(device)
            print(f'   ✓ Data prepared in FP16: {self.x_data.shape}')
            print(f'   ✓ Memory used: {self.x_data.element_size() * self.x_data.nelement() / 1024**3:.2f} GB (50% reduction)')
            self.data_on_gpu = False  # Will be moved to GPU after DataLoader
            self.load_all = True
        else:
            print('Streaming mode (slower but low memory)')
            self.x_data = x_data
            self.data_on_gpu = False
            self.load_all = False

    def __getitem__(self, index):
        output = self.x_data[index]
        if not self.data_on_gpu:
            # Return CPU tensor - let training loop handle GPU transfer
            if isinstance(output, np.ndarray):
                output = torch.from_numpy(output).float()  # Keep on CPU
            # For load_all=True: data is already FP16 tensor on CPU
        return output

    def __len__(self):
        return len(self.x_data)

from skimage.util import random_noise
from torchvision.transforms import v2

class PINNDataset(Dataset):
    def __init__(self, x_data, y1_data, y2_data):
        # Keep data on CPU to avoid CUDA context sharing issues with multi-threaded DataLoaders
        self.x_data = torch.tensor(x_data, dtype=torch.float32)  # Remove .to(device)
        self.y1_data = torch.tensor(y1_data, dtype=torch.float32)  # Remove .to(device)
        self.y2_data = torch.tensor(y2_data, dtype=torch.float32)  # Remove .to(device)

    def __getitem__(self, index):
        x = self.x_data[index]
        # Return CPU tensors - training loop will handle GPU transfer
        return x, self.y1_data[index], self.y2_data[index]
    
    def __len__(self):
        return self.x_data.shape[0]

def check_model_for_nan(model, description="Model"):
    """Check if model parameters contain NaN values"""
    nan_found = False
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            print(f"NaN found in {description} parameter: {name}")
            nan_found = True
        if param.grad is not None and torch.isnan(param.grad).any():
            print(f"NaN found in {description} gradient: {name}")
            nan_found = True
    return nan_found

def check_tensor_stats(tensor, name="Tensor"):
    """Print statistics about a tensor to help debug NaN issues"""
    if tensor is None:
        print(f"{name}: None")
        return
    
    print(f"{name} stats:")
    print(f"  Shape: {tensor.shape}")
    print(f"  Min: {tensor.min().item():.6f}")
    print(f"  Max: {tensor.max().item():.6f}")
    print(f"  Mean: {tensor.mean().item():.6f}")
    print(f"  Std: {tensor.std().item():.6f}")
    print(f"  Has NaN: {torch.isnan(tensor).any().item()}")
    print(f"  Has Inf: {torch.isinf(tensor).any().item()}")

def safe_log(x, eps=1e-8):
    """Safe logarithm that avoids log(0)"""
    return torch.log(torch.clamp(x, min=eps))

def safe_exp(x, max_val=10):
    """Safe exponential that avoids overflow"""
    return torch.exp(torch.clamp(x, max=max_val))