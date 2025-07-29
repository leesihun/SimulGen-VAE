# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Core Architecture

SimulGenVAE is a hierarchical Variational Autoencoder designed for simulation data generation with three key components:

1. **VAE Network** (`modules/VAE_network.py`): Main encoder-decoder architecture
   - Hierarchical latent space: `latent_dim` (8) → `latent_dim_end` (32)
   - Supports multiple loss functions: MSE, MAE, SmoothL1, Huber
   - Uses GroupNorm for batch-size independent normalization

2. **Latent Conditioner Models**: Three specialized architectures for different input types
   - **MLP** (`latent_conditioner_model_parametric.py`): For tabular/parametric data (`input_type=csvs`)
   - **CNN** (`latent_conditioner_model_cnn.py`): For image data (`input_type=image`) 
   - **Vision Transformer** (`latent_conditioner_model_vit.py`): For image data with attention (`input_type=image_vit`)

3. **Training Infrastructure** (`modules/train.py`): 
   - Distributed Data Parallel (DDP) support
   - Advanced optimizations: SAM optimizer, mixed precision, CUDA graphs
   - Anti-overfitting: progressive dropout, label smoothing, EMA weights

## Configuration System

All training parameters are controlled via `input_data/condition.txt`:

```
# Key VAE parameters
Training_epochs    10002
Batch_size         16  
LearningR          0.0005
Latent_dim         8      # Hierarchical dimension
Latent_dim_end     32     # Main latent dimension

# LatentConditioner parameters  
input_type         image  # Options: image, csvs, image_vit
latent_conditioner_lr           0.00001
latent_conditioner_batch        64
latent_conditioner_weight_decay 1e-4
```

## Development Commands

### Single GPU Training
```bash
python SimulGen-VAE.py --preset=1 --plot=2 --lc_only=0 --size=small --load_all=1
```

### Multi-GPU Training (Recommended)
```bash
torchrun --nproc_per_node=4 SimulGen-VAE.py --use_ddp --preset=1 --plot=2 --lc_only=0 --size=small --load_all=1
```

### Train Only LatentConditioner
```bash
python SimulGen-VAE.py --lc_only=1 --preset=1
```

### Key Arguments
- `--preset`: Dataset configuration (1-5)
- `--lc_only`: Train only LatentConditioner (0=full VAE, 1=LC only)
- `--size`: Model size preset (small/big)
- `--load_all`: Preload dataset to GPU memory (0/1)
- `--plot`: Visualization mode (0=interactive, 1=save, 2=off)
- `--use_ddp`: Enable distributed training

### Dependencies Installation
```bash
pip install -r requirements.txt
```

### Monitoring
```bash
tensorboard --logdir=runs --port=6006
watch -n 0.5 nvidia-smi  # GPU monitoring
```

## Data Pipeline

### Input Data Structure
- **VAE Dataset**: 3D pickle arrays `[num_parameters, num_timesteps, num_nodes]`
- **LatentConditioner**: Images (256×256, resized to 128×128) or CSV files
- **Configuration**: Managed via `condition.txt` and `preset.txt`

### Model Selection Logic
The `input_type` parameter in `condition.txt` determines which latent conditioner model is used:
- `csvs` → MLP model for parametric data
- `image` → CNN model for traditional image processing
- `image_vit` → Vision Transformer for modern attention-based processing

## Performance Features

### Multi-GPU Scaling
- Automatic batch size scaling with GPU count
- NCCL backend for efficient communication
- Expected speedup: 2 GPUs (~1.8x), 4 GPUs (~3.5x), 8 GPUs (~6.5x)

### Memory Optimizations
- Mixed precision training (FP16/BF16)
- CUDA graphs for consistent operations
- GPU data prefetching with pinned memory
- `--load_all=1` for small datasets that fit in GPU memory

### Anti-Overfitting Arsenal
- SAM (Sharpness-Aware Minimization) optimizer
- Progressive dropout scheduling (30% → 10%)
- Label smoothing and EMA weights
- Five data augmentation techniques: noise, scaling, shifting, mixup, cutout

## Code Structure Notes

- **Modular Design**: All components in `modules/` directory for clean separation
- **GroupNorm Usage**: Batch-size independent normalization for stable DDP training
- **Loss Flexibility**: Multiple loss functions pre-compiled for efficiency
- **Error Handling**: Graceful DDP fallback and comprehensive validation
- **Modern PyTorch**: Uses `torchrun` for DDP, `torch.compile` for optimization