import os
import glob
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random
import torch
import multiprocessing

def get_optimal_workers(dataset_size, load_all, batch_size):
    """
    Intelligently determine optimal number of DataLoader workers based on:
    - Dataset size
    - Whether data is preloaded (load_all)
    - System capabilities
    - Batch size
    """
    if load_all:
        # Data already on GPU, no need for workers
        return 0
    
    # Get CPU count, but cap it reasonably
    cpu_count = multiprocessing.cpu_count()
    
    # For small datasets, use fewer workers to avoid overhead
    if dataset_size < 500:
        return 0
    elif dataset_size < 2000:
        return min(2, cpu_count)
    elif dataset_size < 10000:
        return min(4, cpu_count)
    else:
        # For large datasets, use more workers but not too many
        # Rule of thumb: 1 worker per 2-4 CPU cores, capped at 8
        optimal = min(max(cpu_count // 2, 2), 8)
        return optimal

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
            self.x_data = torch.tensor(x_data).to(device)
            self.load_all = True
        else:
            # Keep as numpy array for multi-threaded access
            self.x_data = x_data
            self.load_all = False

    def __getitem__(self, index):
        if self.load_all:
            # Data already on GPU as tensor
            output = self.x_data[index]
        else:
            # Convert numpy slice to tensor efficiently
            # Use torch.from_numpy for zero-copy conversion, then ensure contiguous
            output = torch.from_numpy(self.x_data[index].copy()).float()
        
        return output

    def __len__(self):
        return len(self.x_data)

from skimage.util import random_noise
from torchvision.transforms import v2

class PINNDataset(Dataset):
    def __init__(self, x_data, y1_data, y2_data):
        # Keep data on CPU to avoid CUDA context sharing issues with multi-threaded DataLoaders
        self.x_data = torch.tensor(x_data).to(device)
        self.y1_data = torch.tensor(y1_data).to(device)
        self.y2_data = torch.tensor(y2_data).to(device)

    def __getitem__(self, index):
        x = self.x_data[index]
        # Return CPU tensors - training loop will handle GPU transfer
        return x, self.y1_data[index], self.y2_data[index]
    
    def __len__(self):
        return self.x_data.shape[0]

def get_latest_file(path, pattern):
    """Get the most recently modified file matching pattern in path"""
    import glob
    import os
    
    files = glob.glob(os.path.join(path, pattern))
    if not files:
        return None
    return max(files, key=os.path.getctime)