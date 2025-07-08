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
            print('Loading all data...')
            self.x_data = torch.tensor(x_data).to(device)
        else:
            self.x_data = x_data

    def __getitem__(self, index):
        output= self.x_data[index]
        return output

    def __len__(self):
        return len(self.x_data)

from skimage.util import random_noise
from torchvision.transforms import v2

class PINNDataset(Dataset):
    def __init__(self, x_data, y1_data, y2_data):
        self.x_data = torch.tensor(x_data).to(device)
        self.y1_data = torch.tensor(y1_data).to(device)
        self.y2_data = torch.tensor(y2_data).to(device)

    def __getitem__(self, index):
        x = self.x_data[index]

        return x, self.y1_data[index], self.y2_data[index]
    
    def __len__(self):
        return self.x_data.shape[0]

def check_model_for_nan(model, name="Model"):
    """Check if model parameters contain NaN or Inf values"""
    has_nan = False
    for param_name, param in model.named_parameters():
        if torch.isnan(param).any() or torch.isinf(param).any():
            print(f"Warning: {name} parameter {param_name} contains NaN or Inf")
            has_nan = True
    return has_nan

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