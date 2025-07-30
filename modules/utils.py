import os
import glob
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random
import torch
import multiprocessing
import gc


def get_latest_file(directory, pattern='*'):
    """Get the most recently modified file in directory matching pattern.
    
    Args:
        directory (str): Directory path to search
        pattern (str): File pattern to match (default: '*' for all files)
    
    Returns:
        str: Path to the most recently modified file
    
    Raises:
        FileNotFoundError: If directory doesn't exist or no matching files found
    """
    if not os.path.exists(directory):
        raise FileNotFoundError(f"Directory {directory} does not exist")
    
    files = glob.glob(os.path.join(directory, pattern))
    if not files:
        raise FileNotFoundError(f"No files matching '{pattern}' found in {directory}")

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
            # Keep as numpy array but ensure it's C-contiguous for faster access
            if not x_data.flags['C_CONTIGUOUS']:
                print("Converting data to C-contiguous format for faster access...")
                x_data = np.ascontiguousarray(x_data)
            self.x_data = x_data
            self.load_all = False
            
            # Pre-allocate pinned memory for faster CPU-GPU transfers
            if torch.cuda.is_available():
                print("Pre-allocating pinned memory for faster transfers...")
                sample_shape = x_data[0].shape
                self.pinned_buffer = torch.empty(sample_shape, dtype=torch.float32, pin_memory=True)
            else:
                self.pinned_buffer = None

    def __getitem__(self, index):
        if self.load_all:
            # Data already on GPU as tensor
            output = self.x_data[index]
        else:
            # Optimized CPU-to-GPU transfer
            if self.pinned_buffer is not None:
                # Use pinned memory buffer for faster transfer
                self.pinned_buffer.copy_(torch.from_numpy(self.x_data[index]))
                output = self.pinned_buffer.clone()
            else:
                # Fallback for CPU-only systems
                output = torch.from_numpy(self.x_data[index].copy()).float()
        
        return output

    def __len__(self):
        return len(self.x_data)

    def prefetch(self):
        """Prefetch data to GPU"""
        if hasattr(self, 'x_data') and torch.cuda.is_available():
            self.x_data = self.x_data.cuda(non_blocking=True)
            return True
        return False

from skimage.util import random_noise
from torchvision.transforms import v2

def cuda_memory_cleanup():
    """Attempt to clear CUDA memory and cache."""
    try:
        # Empty CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
            # Force garbage collection
            gc.collect()
            
            # Print current memory state
            allocated = torch.cuda.memory_allocated() / 1024**2
            reserved = torch.cuda.memory_reserved() / 1024**2
            max_allocated = torch.cuda.max_memory_allocated() / 1024**2
            
            print(f"CUDA Memory - Allocated: {allocated:.2f}MB, Reserved: {reserved:.2f}MB, Max: {max_allocated:.2f}MB")
            
            return True
    except Exception as e:
        print(f"Error during CUDA memory cleanup: {e}")
        return False

def safe_to_device(tensor, device):
    """Safely move tensor to specified device with error handling."""
    try:
        return tensor.to(device)
    except RuntimeError as e:
        if "CUDA" in str(e):
            print(f"CUDA error moving tensor to device: {e}")
            print("Attempting memory cleanup...")
            
            # Try to free memory
            cuda_memory_cleanup()
            
            try:
                # Try again with smaller chunks if it's a large tensor
                if tensor.numel() > 1e6:  # If tensor has more than 1M elements
                    print("Large tensor detected, trying alternative approach...")
                    # Move to CPU first if not already there
                    if tensor.device.type != 'cpu':
                        tensor = tensor.cpu()
                    return tensor.to(device)
                else:
                    return tensor.to(device)
            except RuntimeError:
                print("Still cannot move to CUDA, falling back to CPU")
                return tensor.to('cpu')
        else:
            # Re-raise if not CUDA related
            raise e

class LatentConditionerDataset(Dataset):
    """GPU-optimized dataset for LatentConditioner training - eliminates CPU bottleneck."""
    def __init__(self, input_data, output_data1, output_data2, preload_gpu=True):
        # Clean NaN values
        if np.isnan(input_data).any():
            print("Warning: NaN values detected in input_data, replacing with zeros")
            input_data = np.nan_to_num(input_data, nan=0.0)
        
        if np.isnan(output_data1).any():
            print("Warning: NaN values detected in output_data1, replacing with zeros")
            output_data1 = np.nan_to_num(output_data1, nan=0.0)
        
        if np.isnan(output_data2).any():
            print("Warning: NaN values detected in output_data2, replacing with zeros")
            output_data2 = np.nan_to_num(output_data2, nan=0.0)
        
        # CRITICAL OPTIMIZATION: Pre-convert to tensors and preload to GPU if possible
        if preload_gpu and torch.cuda.is_available():
            try:
                print("🚀 Preloading LatentConditioner data to GPU for maximum speed...")
                self.input_data = torch.from_numpy(input_data).float().cuda()
                self.output_data1 = torch.from_numpy(output_data1).float().cuda() 
                self.output_data2 = torch.from_numpy(output_data2).float().cuda()
                self.on_gpu = True
                print(f"✅ Successfully preloaded {len(input_data)} samples to GPU")
                print(f"   GPU memory used: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
            except RuntimeError as e:
                print(f"⚠️ Failed to preload to GPU ({e}), using CPU tensors with pinned memory")
                self.input_data = torch.from_numpy(input_data).float().pin_memory()
                self.output_data1 = torch.from_numpy(output_data1).float().pin_memory()
                self.output_data2 = torch.from_numpy(output_data2).float().pin_memory()
                self.on_gpu = False
        else:
            # Fallback: Use pinned memory for faster CPU->GPU transfer
            self.input_data = torch.from_numpy(input_data).float().pin_memory()
            self.output_data1 = torch.from_numpy(output_data1).float().pin_memory()
            self.output_data2 = torch.from_numpy(output_data2).float().pin_memory()
            self.on_gpu = False
    
    def __len__(self):
        return len(self.input_data)
    
    def __getitem__(self, idx):
        try:
            # Data is already tensors - just index them (extremely fast)
            x = self.input_data[idx]
            y1 = self.output_data1[idx] 
            y2 = self.output_data2[idx]
            return x, y1, y2
        except Exception as e:
            print(f"Error getting LatentConditioner dataset item {idx}: {e}")
            # Return zeros as fallback
            return torch.zeros_like(self.input_data[0]), \
                   torch.zeros_like(self.output_data1[0]), \
                   torch.zeros_like(self.output_data2[0])

def get_optimal_workers(dataset_size, is_load_all=False, batch_size=32):
    """Intelligently determine optimal number of DataLoader workers.
    
    Args:
        dataset_size (int): Number of samples in the dataset
        is_load_all (bool): Whether data is preloaded to GPU
        batch_size (int): Batch size for training
    
    Returns:
        int: Optimal number of DataLoader workers
    
    Note:
        Returns 0 workers if data is preloaded to GPU for maximum efficiency.
    """
    if is_load_all:
        # When data is preloaded to GPU, best to use 0 workers
        return 0
    
    # Get CPU count but cap at 8 for reasonable performance
    cpu_count = min(8, torch.multiprocessing.cpu_count())
    
    # For very small datasets, fewer workers are better
    if dataset_size < 1000:
        return 0  # Single-threaded is best for tiny datasets
    elif dataset_size < 10000:
        return min(2, cpu_count)  # Small datasets need fewer workers
    else:
        # Larger datasets - scale workers based on batch size
        if batch_size < 16:
            return min(4, cpu_count)
        else:
            return min(cpu_count, 6)  # Cap at 6 workers

