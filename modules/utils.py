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
    Determines optimal number of DataLoader workers based on dataset characteristics.
    For small datasets with large batches, fewer workers are often better.
    """
    import torch
    
    if load_all:
        # Data already on GPU, no workers needed
        return 0
    
    # For small datasets with large batches, minimize overhead
    if dataset_size < 500:  # Small variety case
        return 0  # Single-threaded is often faster
    elif dataset_size < 2000:
        return min(2, torch.multiprocessing.cpu_count())
    else:
        # For larger datasets, use more workers
        optimal = min(4, torch.multiprocessing.cpu_count())
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

class CachedDataset(Dataset):
    """
    A dataset that caches frequently accessed samples in memory.
    Ideal for small datasets with large batches.
    """
    def __init__(self, x_data, load_all=False, cache_size=None):
        print('Loading data with smart caching...')
        self.load_all = load_all
        
        if load_all:
            # Full GPU preloading for small datasets
            self.x_data = torch.tensor(x_data).to(device)
            self.cache = None
        else:
            # Keep original data as numpy for memory efficiency
            self.x_data = x_data
            # Cache for frequently accessed samples
            self.cache_size = cache_size or min(len(x_data), 1000)
            self.cache = {}
            self.access_count = {}
            
    def __getitem__(self, index):
        if self.load_all:
            return self.x_data[index]
        
        # Check cache first
        if index in self.cache:
            self.access_count[index] = self.access_count.get(index, 0) + 1
            return self.cache[index]
        
        # Convert and potentially cache
        output = torch.from_numpy(self.x_data[index].copy()).float()
        
        # Cache if we have space or if this is frequently accessed
        if len(self.cache) < self.cache_size:
            self.cache[index] = output
            self.access_count[index] = 1
        
        return output
    
    def __len__(self):
        return len(self.x_data)