import os
import glob
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader
import random
import torch
import multiprocessing
import gc
import torch.distributed as dist


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
                # Convert to C-contiguous without debug print for speed
                x_data = np.ascontiguousarray(x_data)
            self.x_data = x_data
            self.load_all = False
            
            # Pre-allocate pinned memory for faster CPU-GPU transfers
            if torch.cuda.is_available():
                # Allocate pinned memory without debug print for speed
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

def cuda_memory_cleanup(debug_mode=0):
    """Attempt to clear CUDA memory and cache."""
    try:
        # Empty CUDA cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            
            # Force garbage collection
            gc.collect()
            
            # Print current memory state only in debug mode
            if debug_mode == 1:
                allocated = torch.cuda.memory_allocated() / 1024**2
                reserved = torch.cuda.memory_reserved() / 1024**2
                max_allocated = torch.cuda.max_memory_allocated() / 1024**2
                
                print(f"CUDA Memory - Allocated: {allocated:.2f}MB, Reserved: {reserved:.2f}MB, Max: {max_allocated:.2f}MB")
            
            return True
    except Exception as e:
        if debug_mode == 1:
            print(f"Error during CUDA memory cleanup: {e}")
        return False

def safe_to_device(tensor, device, debug_mode=0):
    """Safely move tensor to specified device with error handling."""
    try:
        return tensor.to(device)
    except RuntimeError as e:
        if "CUDA" in str(e):
            if debug_mode == 1:
                print(f"CUDA error moving tensor to device: {e}")
                print("Attempting memory cleanup...")
            
            # Try to free memory
            cuda_memory_cleanup(debug_mode)
            
            try:
                # Try again with smaller chunks if it's a large tensor
                if tensor.numel() > 1e6:  # If tensor has more than 1M elements
                    if debug_mode == 1:
                        print("Large tensor detected, trying alternative approach...")
                    # Move to CPU first if not already there
                    if tensor.device.type != 'cpu':
                        tensor = tensor.cpu()
                    return tensor.to(device)
                else:
                    return tensor.to(device)
            except RuntimeError:
                if debug_mode == 1:
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
                # GPU preloading without debug print for speed
                self.input_data = torch.from_numpy(input_data).float().cuda()
                self.output_data1 = torch.from_numpy(output_data1).float().cuda() 
                self.output_data2 = torch.from_numpy(output_data2).float().cuda()
                self.on_gpu = True
            except RuntimeError as e:
                # Still print this as it's an important fallback notification
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


def setup_distributed_training(args):
    """Setup distributed training if requested.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        bool: True if distributed training is enabled, False otherwise
    """
    if args.use_ddp:
        try:
            # torchrun automatically sets environment variables
            local_rank = int(os.environ.get("LOCAL_RANK", -1))
            if local_rank == -1:
                print("For DDP training, please use: torchrun --nproc_per_node=NUM_GPUS SimulGen-VAE.py --use_ddp [other args]")
                is_distributed = False
            else:
                # Initialize the process group (torchrun handles most setup)
                torch.cuda.set_device(local_rank)
                dist.init_process_group(backend="nccl")
                is_distributed = True
                print(f"Initialized DDP process group. Rank {dist.get_rank()} of {dist.get_world_size()}")
        except Exception as e:
            print(f"Failed to initialize DDP: {e}")
            print("Falling back to single GPU training")
            is_distributed = False
    else:
        is_distributed = False
    
    return is_distributed


def print_gpu_mem_checkpoint(msg, debug_mode=0):
    """Print GPU memory usage statistics for debugging.
    
    Args:
        msg (str): Descriptive message for the checkpoint
        debug_mode (int): Debug verbosity level (0=off, 1=on)
    """
    if debug_mode == 1 and torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        max_allocated = torch.cuda.max_memory_allocated() / 1024**2
        print(f"[GPU MEM] {msg}: Allocated={allocated:.2f}MB, Max Allocated={max_allocated:.2f}MB")
        torch.cuda.reset_peak_memory_stats()


def parse_condition_file(filepath):
    """Parse configuration file to extract training parameters.
    
    Reads the condition.txt file and extracts key-value pairs for
    training configuration, skipping comments and section markers.
    
    Args:
        filepath (str): Path to the condition.txt configuration file
        
    Returns:
        dict: Dictionary mapping parameter names to their values
        
    Note:
        Ignores lines starting with %, ', or # (comments and sections)
    """
    params = {}
    with open(filepath, encoding='utf-8') as f:
        for line in f:
            # Remove comments and whitespace
            line = line.split('#')[0].strip()
            if not line or line.startswith('%') or line.startswith("'"):
                continue  # skip empty lines and section markers
            parts = line.split()
            if len(parts) >= 2:
                key = parts[0]
                value = parts[1]
                params[key] = value
    return params


def parse_training_parameters(params):
    """Parse raw parameter dictionary into structured training configuration.
    
    Args:
        params (dict): Raw parameters from condition.txt file
        
    Returns:
        dict: Structured configuration dictionary with typed values
    """
    config = {}
    
    # Basic dimensions
    config['num_param'] = int(params['Dim1'])
    config['num_time'] = int(params['Dim2'])
    config['num_time_to'] = int(params['Dim2_red'])
    config['num_node'] = int(params['Dim3'])
    config['num_node_to'] = int(params['Dim3_red'])
    config['num_var'] = int(params['num_var'])
    
    # Training parameters
    config['n_epochs'] = int(params['Training_epochs'])
    config['batch_size'] = int(params['Batch_size'])
    config['LR'] = float(params['LearningR'])
    config['latent_dim'] = int(params['Latent_dim'])
    config['latent_dim_end'] = int(params['Latent_dim_end'])
    config['loss_type'] = int(params['Loss_type'])
    config['stretch'] = int(params['Stretch'])
    config['alpha'] = int(params['alpha'])
    
    # Augmentation parameters
    config['num_samples_f'] = int(params.get('num_aug_f', 0))
    config['num_samples_a'] = int(params.get('num_aug_a', 0))
    config['print_graph_recon'] = int(params.get('rec_graph', 0))
    config['recon_iter'] = int(params.get('Recon_iter', 1))
    
    # Physical parameters
    config['num_physical_param'] = int(params['num_param'])
    config['n_sample'] = int(params['n_sample'])
    config['param_dir'] = params['param_dir']
    
    # Latent conditioner parameters
    config['latent_conditioner_epoch'] = int(params['n_epoch'])
    config['latent_conditioner_lr'] = float(params['latent_conditioner_lr'])
    config['latent_conditioner_batch_size'] = int(params['latent_conditioner_batch'])
    config['latent_conditioner_data_type'] = params['input_type']
    config['param_data_type'] = params['param_data_type']
    config['latent_conditioner_weight_decay'] = float(params.get('latent_conditioner_weight_decay', 1e-4))
    config['latent_conditioner_dropout_rate'] = float(params.get('latent_conditioner_dropout_rate', 0.3))
    config['use_spatial_attention'] = int(params.get('use_spatial_attention', 1))
    
    # PCA parameters
    config['use_pca'] = int(params['use_pca'])
    config['pca_components'] = int(params['pca_components'])
    config['pca_patch_size'] = int(params['pca_patch_size'])
    
    return config


def evaluate_vae_reconstruction(VAE, dataloader, device, num_param, num_filter_enc, latent_dim, 
                               latent_dim_end, recon_iter=1, dataset_name="Dataset"):
    """Evaluate VAE reconstruction performance on a dataset.
    
    Args:
        VAE: Trained VAE model
        dataloader: DataLoader for evaluation
        device: torch device for computations
        num_param: Number of parameters/samples
        num_filter_enc: Encoder filter configuration
        latent_dim: Hierarchical latent dimension
        latent_dim_end: Main latent dimension  
        recon_iter: Number of reconstruction iterations per sample
        dataset_name: Name for logging purposes
        
    Returns:
        tuple: (latent_vectors, hierarchical_latent_vectors, reconstruction_loss, reconstructed, total_loss)
    """
    import torch.nn as nn
    import numpy as np
    from modules.decoder import reparameterize
    
    # Initialize storage arrays
    latent_vectors = np.zeros([num_param, latent_dim_end])
    hierarchical_latent_vectors = np.zeros([num_param, len(num_filter_enc)-1, latent_dim])
    reconstruction_loss = np.zeros([num_param])
    
    # Try to get data shape from different possible dataset structures
    try:
        if hasattr(dataloader.dataset, 'dataset'):
            # For split datasets
            data_shape = dataloader.dataset.dataset.x_data.shape
        else:
            # For regular datasets
            data_shape = dataloader.dataset.x_data.shape
    except:
        # Fallback - get from first batch
        sample_batch = next(iter(dataloader))
        data_shape = sample_batch.shape
        # Reset dataloader
        dataloader = dataloader.__class__(dataloader.dataset, **dataloader.__dict__)
    
    reconstructed = np.empty([num_param, data_shape[1], data_shape[2]])
    
    loss_total = 0
    num_var = 1  # For loss_save array
    loss_save = np.zeros([num_var])
    
    print(f"Evaluating {dataset_name}...")
    
    for j, image in enumerate(dataloader):
        loss_save[:] = 100
        x = image.to(device)
        del image

        mu, log_var, xs = VAE.encoder(x)
        best_loss = float('inf')
        
        for i in range(recon_iter):
            std = torch.exp(0.5*log_var)
            latent_vector = reparameterize(mu, std)

            gen_x, _ = VAE.decoder(latent_vector, xs, mode='fix')
            gen_x_np = gen_x.cpu().detach().numpy()

            loss = nn.MSELoss()(gen_x, x)

            if loss < loss_save[0]:
                loss_save[0] = loss
                latent_vector_save = latent_vector
                latent_vectors[j, :] = latent_vector_save[0, :].cpu().detach().numpy()

                for k in range(len(xs)):
                    hierarchical_latent_vectors[j, k, :] = xs[k].cpu().detach().numpy()[0]

                reconstruction_loss[j] = loss
                reconstructed[j, :, :] = gen_x_np[0, :, :]

                del latent_vector, gen_x, gen_x_np, latent_vector_save

        print(f'Parameter {j+1} finished - MSE: {loss:.4E}')
        loss_total = loss_total + loss.cpu().detach().numpy()
        
        # Clean up
        del loss, x, mu, log_var, xs, std

    print('')
    average_loss = loss_total / (j + 1) if j >= 0 else 0
    print(f'Total {dataset_name} MSE loss: {average_loss:.3e}')
    print('--------------------------------')
    print('')
    
    return latent_vectors, hierarchical_latent_vectors, reconstruction_loss, reconstructed, loss_total


def evaluate_vae_simple(VAE, dataloader, device, dataset_name="Dataset"):
    """Simple VAE evaluation without storing latent vectors.
    
    Args:
        VAE: Trained VAE model
        dataloader: DataLoader for evaluation
        device: torch device for computations
        dataset_name: Name for logging purposes
        
    Returns:
        float: Total reconstruction loss
    """
    import torch.nn as nn
    from modules.decoder import reparameterize
    
    loss_total = 0
    print(f"Evaluating {dataset_name}...")
    
    with torch.no_grad():
        for j, image in enumerate(dataloader):
            x = image.to(device)
            mu, log_var, xs = VAE.encoder(x)
            std = torch.exp(0.5*log_var)
            latent_vector = reparameterize(mu, std)
            gen_x, _ = VAE.decoder(latent_vector, xs, mode='fix')
            
            loss = nn.MSELoss()(gen_x, x)
            print(f'Parameter {j+1} finished - MSE: {loss:.4E}')
            loss_total = loss_total + loss.cpu().detach().numpy()
    
    print('')
    average_loss = loss_total / (j + 1) if j >= 0 else 0
    print(f'Total {dataset_name} MSE loss: {average_loss:.3e}')
    print('--------------------------------')
    print('')
    
    return loss_total

