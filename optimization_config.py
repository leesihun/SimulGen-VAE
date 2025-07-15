"""
Optimization Configuration for SimulGen-VAE

This file contains optimization settings for different training scenarios.
Choose the configuration that best matches your dataset characteristics.
"""

# =============================================================================
# SCENARIO 1: Small Variety, Large Batches (Your Case)
# =============================================================================
SMALL_VARIETY_LARGE_BATCH = {
    'description': 'Optimized for datasets with limited variety but large batch sizes',
    'load_all': True,          # Preload all data to GPU
    'use_cached_dataset': True,
    'compile_model': True,     # Use torch.compile for consistent shapes
    'batch_size_multiplier': 1.5,  # Increase batch size by 50%
    'num_workers': 0,          # Single-threaded for small datasets
    'pin_memory': False,       # Not needed when load_all=True
    'prefetch_factor': None,   # Not applicable for single-threaded
    'persistent_workers': False,
    'gradient_accumulation_steps': 1,
    'memory_fraction': 0.95,   # Use most GPU memory
    'enable_tf32': True,
    'cudnn_benchmark': True,
    'cudnn_deterministic': False,
    'mixed_precision': True,
    'optimizer_settings': {
        'zero_grad_set_to_none': True,
        'weight_decay': 0,  # Disable for small datasets
        'gradient_clipping': 5.0,
    },
    'scheduler_settings': {
        'type': 'cosine',
        'eta_min_ratio': 0.01,
    }
}

# =============================================================================
# SCENARIO 2: Large Variety, Small Batches
# =============================================================================
LARGE_VARIETY_SMALL_BATCH = {
    'description': 'Optimized for datasets with high variety but small batch sizes',
    'load_all': False,         # Keep data on CPU
    'use_cached_dataset': True,
    'compile_model': False,    # Variable shapes prevent compilation
    'batch_size_multiplier': 1.0,
    'num_workers': 4,          # Multi-threaded loading
    'pin_memory': True,
    'prefetch_factor': 2,
    'persistent_workers': True,
    'gradient_accumulation_steps': 4,  # Simulate larger batches
    'memory_fraction': 0.8,
    'enable_tf32': True,
    'cudnn_benchmark': False,  # Variable shapes
    'cudnn_deterministic': False,
    'mixed_precision': True,
    'optimizer_settings': {
        'zero_grad_set_to_none': True,
        'weight_decay': 1e-4,
        'gradient_clipping': 1.0,
    },
    'scheduler_settings': {
        'type': 'cosine',
        'eta_min_ratio': 0.01,
    }
}

# =============================================================================
# SCENARIO 3: Memory Constrained
# =============================================================================
MEMORY_CONSTRAINED = {
    'description': 'Optimized for limited GPU memory',
    'load_all': False,
    'use_cached_dataset': True,
    'compile_model': False,
    'batch_size_multiplier': 0.5,  # Reduce batch size
    'num_workers': 2,
    'pin_memory': True,
    'prefetch_factor': 1,
    'persistent_workers': True,
    'gradient_accumulation_steps': 8,  # Compensate for small batches
    'memory_fraction': 0.7,
    'enable_tf32': True,
    'cudnn_benchmark': True,
    'cudnn_deterministic': False,
    'mixed_precision': True,
    'optimizer_settings': {
        'zero_grad_set_to_none': True,
        'weight_decay': 1e-4,
        'gradient_clipping': 1.0,
    },
    'scheduler_settings': {
        'type': 'cosine',
        'eta_min_ratio': 0.01,
    }
}

# =============================================================================
# SCENARIO 4: Maximum Speed (Your Recommended Setting)
# =============================================================================
MAXIMUM_SPEED = {
    'description': 'Aggressive optimizations for maximum training speed',
    'load_all': True,
    'use_cached_dataset': True,
    'compile_model': True,
    'batch_size_multiplier': 2.0,  # Double the batch size if memory allows
    'num_workers': 0,
    'pin_memory': False,
    'prefetch_factor': None,
    'persistent_workers': False,
    'gradient_accumulation_steps': 1,
    'memory_fraction': 0.98,   # Use almost all GPU memory
    'enable_tf32': True,
    'cudnn_benchmark': True,
    'cudnn_deterministic': False,
    'mixed_precision': True,
    'optimizer_settings': {
        'zero_grad_set_to_none': True,
        'weight_decay': 0,
        'gradient_clipping': 10.0,  # Higher for stability with large batches
    },
    'scheduler_settings': {
        'type': 'cosine',
        'eta_min_ratio': 0.001,
    }
}

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def get_config(scenario='small_variety_large_batch'):
    """
    Get optimization configuration for a specific scenario.
    
    Args:
        scenario: One of 'small_variety_large_batch', 'large_variety_small_batch', 
                 'memory_constrained', 'maximum_speed'
    
    Returns:
        Dictionary with optimization settings
    """
    configs = {
        'small_variety_large_batch': SMALL_VARIETY_LARGE_BATCH,
        'large_variety_small_batch': LARGE_VARIETY_SMALL_BATCH,
        'memory_constrained': MEMORY_CONSTRAINED,
        'maximum_speed': MAXIMUM_SPEED,
    }
    
    if scenario not in configs:
        print(f"Warning: Unknown scenario '{scenario}'. Using 'small_variety_large_batch'.")
        scenario = 'small_variety_large_batch'
    
    config = configs[scenario].copy()
    print(f"Using optimization config: {config['description']}")
    return config

def apply_config(config, args):
    """
    Apply optimization configuration to training arguments.
    
    Args:
        config: Configuration dictionary from get_config()
        args: Training arguments object
    
    Returns:
        Modified args object
    """
    # Update load_all setting
    if hasattr(args, 'load_all'):
        args.load_all = config['load_all']
    
    # Apply batch size multiplier
    if hasattr(args, 'batch_size') and config['batch_size_multiplier'] != 1.0:
        original_batch_size = args.batch_size
        args.batch_size = int(args.batch_size * config['batch_size_multiplier'])
        print(f"Adjusted batch size: {original_batch_size} -> {args.batch_size}")
    
    return args

def print_optimization_summary(config):
    """Print a summary of active optimizations."""
    print("\n" + "="*60)
    print("ACTIVE OPTIMIZATIONS")
    print("="*60)
    print(f"Configuration: {config['description']}")
    print(f"Data Loading: {'GPU preload' if config['load_all'] else 'CPU with caching'}")
    print(f"Model Compilation: {'Enabled' if config['compile_model'] else 'Disabled'}")
    print(f"Mixed Precision: {'Enabled' if config['mixed_precision'] else 'Disabled'}")
    print(f"Batch Size Multiplier: {config['batch_size_multiplier']}x")
    print(f"Workers: {config['num_workers']}")
    print(f"Memory Fraction: {config['memory_fraction']}")
    print(f"Gradient Accumulation: {config['gradient_accumulation_steps']} steps")
    print("="*60)
    print() 