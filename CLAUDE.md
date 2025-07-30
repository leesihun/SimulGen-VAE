# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Overview

SimulGenVAE is a high-performance Variational Autoencoder system designed for fast generation and inference of transient/static simulation data with Physics-Aware Neural Network (PANN) integration. The system supports both single-GPU and multi-GPU distributed training with three different latent conditioning architectures.

## Core Architecture

### Main Components
- **VAE Network** (`modules/VAE_network.py`): Main Variational Autoencoder with hierarchical encoder-decoder architecture
- **Latent Conditioner** (`modules/latent_conditioner.py`): Conditions latent space on external parameters (images/parametric data)
- **Training System** (`modules/train.py`): Advanced training pipeline with mixed precision, gradient checkpointing
- **Data Processing** (`modules/data_preprocess.py`): Handles simulation dataset loading and preprocessing

### Latent Conditioning Architectures
1. **MLP-based** (`latent_conditioner_model_parametric.py`): For parametric data input
2. **CNN-based** (`latent_conditioner_model_cnn.py`): For image/outline detection with spatial attention
3. **Vision Transformer** (`latent_conditioner_model_vit.py`): For complex image analysis

### Key Features
- Hierarchical latent space (main: 32D, hierarchical: 8D)
- Comprehensive overfitting prevention (dropout, augmentation, early stopping)
- Mixed precision training with GradScaler
- Distributed Data Parallel (DDP) support
- Advanced learning rate scheduling (warmup + cosine annealing)
- Multiple loss functions (MSE, MAE, smoothL1, Huber)

## Common Commands

### Training Commands

#### Single GPU Training
```bash
python SimulGen-VAE.py --preset=1 --plot=2 --lc_only=0 --size=small --load_all=1
```

#### Multi-GPU Training
```bash
torchrun --nproc_per_node=4 SimulGen-VAE.py --use_ddp --preset=1 --plot=2 --lc_only=0 --size=small --load_all=1
```

#### DDP Launcher (Simplified)
```bash
python launch_ddp.py --gpus=2 --preset=1 --plot=2 --size=small
```

#### Train Only Latent Conditioner
```bash
python SimulGen-VAE.py --preset=1 --lc_only=1 --plot=2
```

### Key Arguments
- `--preset`: Dataset configuration (1-5, defined in preset.txt)
- `--plot`: Plotting mode (0=off, 1=basic, 2=comprehensive)
- `--lc_only`: Train only latent conditioner (0=full training, 1=LC only)
- `--size`: Model size (small/big)
- `--load_all`: Load all data to memory (0=lazy loading, 1=preload)
- `--use_ddp`: Enable distributed training

### No Standard Test/Lint Commands
This repository does not include standard testing or linting commands. The system relies on internal validation loops and checkpointing for quality assurance.

## Configuration System

### Primary Configuration Files
1. **`preset.txt`**: Dataset presets with encoder/decoder filter configurations
   - Line 1: Headers (data_No, init_beta_divisor, num_filter_enc, latent_conditioner_filter)
   - Line 2: Preset number selection
   - Line 3: Beta initialization divisor
   - Line 4: Encoder filter sizes (e.g., "1024 512 256 128")
   - Line 5: Latent conditioner filter sizes (e.g., "16 32 64")

2. **`input_data/condition.txt`**: Comprehensive training parameters
   - VAE parameters: epochs, batch size, learning rate, latent dimensions
   - Latent Conditioner parameters: architecture type, dropout rates, attention settings
   - Data parameters: dimensions, augmentation settings
   - Input type specification: "image", "csvs", or "image_vit"

### Configuration Structure
```
Common params
Dim1: 484 (number of parameters)
Dim2: 200 (timesteps)  
Dim3: 95008 (nodes)

VAE Training
Training_epochs: 10002
Batch_size: 16
LearningR: 0.0005
Latent_dim: 8 (hierarchical)
Latent_dim_end: 32 (main)

Latent Conditioner
n_epoch: 20000
latent_conditioner_lr: 0.001
latent_conditioner_dropout_rate: 0.3
use_spatial_attention: 1
input_type: image
param_data_type: .png
```

## Data Organization

### Expected Directory Structure
```
SimulGenVAE/
├── input_data/
│   ├── condition.txt          # Main configuration
│   └── dataset#X.pickle       # VAE training data
├── images/                    # Latent conditioner input images
├── output/                    # Training outputs and plots
├── model_save/               # Saved model checkpoints
├── checkpoints/              # Training state checkpoints
└── modules/                  # Core implementation
```

### Dataset Format
- **VAE Data**: 3D pickled arrays `[num_param, num_time, num_node]`
- **Image Data**: PNG/JPG files in `/images` directory (resized to 256x256)
- **Parametric Data**: CSV files for MLP-based conditioning

## Development Workflow

### Model Development Pattern
1. Configure parameters in `input_data/condition.txt`
2. Set dataset preset in `preset.txt`
3. Start with small model and single GPU for testing
4. Scale to multi-GPU for full training
5. Monitor via TensorBoard logs in output directory

### Key Implementation Notes
- Uses mixed precision training (autocast + GradScaler)
- Implements aggressive overfitting prevention (see OVERFITTING_PREVENTION_MEASURES.md)
- Supports gradient checkpointing for memory efficiency
- Automatic NaN detection and training recovery
- Advanced learning rate scheduling with warmup phases

### Memory Management
- `--load_all=1`: Preload all data (faster but memory intensive)
- `--load_all=0`: Lazy loading (memory efficient but slower)
- Mixed precision reduces memory usage by ~40%
- Gradient checkpointing available for extreme memory constraints

## Debugging and Monitoring

### Built-in Diagnostics
- Real-time loss monitoring with outlier detection
- GPU memory usage tracking
- NaN detection with automatic recovery
- Overfitting ratio monitoring (val_loss/train_loss)
- Comprehensive validation statistics every 10 epochs

### Output Interpretation
- Training outputs saved to `output/` directory
- TensorBoard logs for loss curves and metrics
- Model checkpoints automatically saved for best validation loss
- Plotting system generates reconstruction comparisons

## Important Implementation Details

### Latent Space Architecture
- **Hierarchical Design**: Two-level latent representation
- **Main Latent**: 32D for primary data representation
- **Hierarchical Latent**: 8D for multi-scale features
- **Conditioning**: External parameters condition the latent space

### Anti-Overfitting Strategy
The system implements comprehensive overfitting prevention:
- Progressive dropout rates (0.1-0.6 depending on architecture)
- Outline-preserving data augmentation for images
- Early stopping with validation monitoring
- Weight decay and gradient clipping
- Label smoothing and mixup augmentation

### Distributed Training
- Uses modern torchrun instead of deprecated torch.distributed.launch
- Automatic local_rank detection
- NCCL backend for multi-GPU communication
- Graceful fallback to single-GPU if DDP fails