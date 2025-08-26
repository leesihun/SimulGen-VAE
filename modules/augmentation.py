import torch
import numpy as np
from torch.utils.data import Dataset
import random
from modules.utils import MyBaseDataset

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class AugmentedDataset(MyBaseDataset):
    """
    Dataset class that applies on-the-fly data augmentation to time series data
    Extends the MyBaseDataset class with various augmentation techniques
    """
    def __init__(self, x_data, load_all, augmentation_config=None):
        """
        Initialize the dataset with augmentation options
        
        Args:
            x_data: Input data in shape [samples, channels, time]
            load_all: Whether to load all data to GPU
            augmentation_config: Dictionary of augmentation parameters
        """
        super().__init__(x_data, load_all)
        
        # Default augmentation configuration - optimized for SimulGenVAE
        self.augmentation_config = {
            'noise_prob': 0.5,        # Probability of adding noise
            'noise_level': 0.1,      # Noise intensity (10%)
            'scaling_prob': 0.5,      # Probability of scaling
            'scaling_range': (0.8, 1.2), # Scaling factor range
            'shift_prob': 0.0,        # Probability of time shifting
            'shift_max': 0.0,         # Maximum shift fraction
            'mixup_prob': 0.5,        # Probability of applying mixup
            'mixup_alpha': 0.3,       # Mixup interpolation strength
            'cutout_prob': 0.0,       # Probability of applying cutout
            'cutout_max': 0.1,        # Maximum cutout fraction
            'enabled': True           # Master switch for augmentation
        }
        
        # Update with user config if provided
        if augmentation_config is not None:
            self.augmentation_config.update(augmentation_config)
        
        # Initialize training mode
        self.training = True
    
    def __getitem__(self, index):
        """Get a data sample with augmentation applied"""
        # Get original sample
        sample = super().__getitem__(index)
        
        # Skip augmentation if disabled or during evaluation
        if not self.augmentation_config['enabled'] or not self.training:
            return sample
        
        # Apply augmentations with their respective probabilities
        sample = self._apply_augmentations(sample, index)
        
        return sample
    
    def _apply_augmentations(self, sample, index):
        """Apply multiple augmentations based on probabilities"""
        # Add Gaussian noise
        if random.random() < self.augmentation_config['noise_prob']:
            sample = self._add_noise(sample)
        
        # Apply random scaling
        if random.random() < self.augmentation_config['scaling_prob']:
            sample = self._apply_scaling(sample)
        
        # Apply time shift
        if random.random() < self.augmentation_config['shift_prob']:
            sample = self._apply_shift(sample)
        
        # Apply mixup (need another sample)
        if random.random() < self.augmentation_config['mixup_prob'] and len(self) > 1:
            # Get another random sample for mixup
            other_idx = random.randint(0, len(self) - 1)
            while other_idx == index:  # Ensure different sample
                other_idx = random.randint(0, len(self) - 1)
            other_sample = super().__getitem__(other_idx)
            sample = self._apply_mixup(sample, other_sample)
        
        # Apply cutout (set random time segments to zero)
        if random.random() < self.augmentation_config['cutout_prob']:
            sample = self._apply_cutout(sample)
            
        return sample
    
    def _add_noise(self, sample):
        """Add Gaussian noise to the sample"""
        noise = torch.randn_like(sample) * self.augmentation_config['noise_level']
        return sample + noise
    
    def _apply_scaling(self, sample):
        """Apply random amplitude scaling"""
        scale_min, scale_max = self.augmentation_config['scaling_range']
        scale_factor = scale_min + random.random() * (scale_max - scale_min)
        return sample * scale_factor
    
    def _apply_shift(self, sample):
        """Apply random time shift"""
        max_shift = int(sample.shape[-1] * self.augmentation_config['shift_max'])
        if max_shift > 0:
            shift = random.randint(-max_shift, max_shift)
            if shift > 0:
                # Shift right
                result = torch.zeros_like(sample)
                result[..., shift:] = sample[..., :-shift]
                return result
            elif shift < 0:
                # Shift left
                shift = abs(shift)
                result = torch.zeros_like(sample)
                result[..., :-shift] = sample[..., shift:]
                return result
        return sample
    
    def _apply_mixup(self, sample1, sample2):
        """Apply mixup between two samples"""
        alpha = self.augmentation_config['mixup_alpha']
        # Sample lambda from beta distribution
        lam = np.random.beta(alpha, alpha)
        # Ensure lambda is not too extreme
        lam = max(0.1, min(lam, 0.9))
        # Mix samples
        mixed_sample = lam * sample1 + (1 - lam) * sample2
        return mixed_sample
    
    def _apply_cutout(self, sample):
        """Apply cutout - set random time segments to zero"""
        max_length = int(sample.shape[-1] * self.augmentation_config['cutout_max'])
        if max_length > 0:
            length = random.randint(1, max_length)
            start = random.randint(0, sample.shape[-1] - length)
            
            # Create mask
            mask = torch.ones_like(sample)
            mask[..., start:start+length] = 0
            
            # Apply mask
            return sample * mask
        return sample
    
    def set_training(self, training=True):
        """Set dataset to training or evaluation mode"""
        self.training = training
    
    def set_augmentation_enabled(self, enabled=True):
        """Enable or disable augmentation"""
        self.augmentation_config['enabled'] = enabled


# Function to create augmented dataloaders
def create_augmented_dataloaders(x_data, batch_size, load_all=False, augmentation_config=None, 
                                val_split=0.2, num_workers=None):
    """
    Create training and validation dataloaders with augmentation
    
    Args:
        x_data: Input data in shape [samples, channels, time]
        batch_size: Batch size for training
        load_all: Whether to load all data to GPU
        augmentation_config: Dictionary of augmentation parameters (optional)
                           If None, uses sensible defaults
        val_split: Fraction of data to use for validation
        num_workers: Number of workers for dataloader
        
    Returns:
        train_dataloader, val_dataloader
    """
    from torch.utils.data import DataLoader
    import torch
    
    # Create train config (use provided config or class defaults)
    train_config = augmentation_config
    
    # Create val config (disable augmentation for validation)
    if augmentation_config is not None:
        val_config = augmentation_config.copy()
        val_config['enabled'] = False
    else:
        val_config = {'enabled': False}
    
    # Split indices first
    dataset_size = len(x_data)
    val_size = int(dataset_size * val_split)
    train_size = dataset_size - val_size
    indices = torch.randperm(dataset_size)
    train_indices = indices[:train_size]
    val_indices = indices[train_size:]
    
    # Create single dataset and handle augmentation via wrapper
    full_dataset = AugmentedDataset(x_data, load_all, train_config)
    
    # Create wrapper class for validation that disables augmentation
    class ValidationWrapper(torch.utils.data.Dataset):
        def __init__(self, dataset, indices):
            self.dataset = dataset
            self.indices = indices
            # Temporarily disable augmentation
            self.original_enabled = dataset.augmentation_config['enabled']
            
        def __len__(self):
            return len(self.indices)
            
        def __getitem__(self, idx):
            # Disable augmentation for validation
            self.dataset.augmentation_config['enabled'] = False
            sample = self.dataset[self.indices[idx]]
            # Restore original setting
            self.dataset.augmentation_config['enabled'] = self.original_enabled
            return sample
    
    # Create train subset (uses augmentation)
    train_subset = torch.utils.data.Subset(full_dataset, train_indices)
    
    # Create validation subset (disables augmentation)
    val_subset = ValidationWrapper(full_dataset, val_indices)
    
    # Determine optimal number of workers if not specified
    if num_workers is None:
        if load_all:
            num_workers = 0  # No workers needed when data is on GPU
        else:
            import multiprocessing
            num_workers = min(4, multiprocessing.cpu_count())
    
    # Create dataloaders
    train_dataloader = DataLoader(
        train_subset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=not load_all
    )
    
    val_dataloader = DataLoader(
        val_subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=not load_all
    )
    
    return train_dataloader, val_dataloader 