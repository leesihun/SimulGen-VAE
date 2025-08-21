# CLAUDE.md

This file provides comprehensive guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Important Precautions

⚠️ **CRITICAL**: The SimulGenVAE code is designed to run on specialized hardware with GPU acceleration. Claude Code should NEVER attempt to:
- Run the training scripts locally
- Execute any GPU-dependent code
- Try to retrieve or access training data from runs/datasets
- Attempt to load large pickle files or datasets

Always ask the user for any information needed about training runs, datasets, or system configurations.

## Project Overview

SimulGenVAE is a high-performance Variational Autoencoder system designed for fast generation and inference of transient/static simulation data with Physics-Aware Neural Network (PANN) integration. The system supports both single-GPU and multi-GPU distributed training with three different latent conditioning architectures.

### Version Information
- **Version**: 2.0.0 (Refactored & Documented)
- **Author**: SiHun Lee, Ph.D.
- **Contact**: kevin1007kr@gmail.com

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

## Common Training Commands

### Single GPU Training
```bash
python SimulGen-VAE.py --preset=1 --plot=2 --lc_only=0 --size=small --load_all=1
```

### Multi-GPU Training
```bash
torchrun --nproc_per_node=4 SimulGen-VAE.py --use_ddp --preset=1 --plot=2 --lc_only=0 --size=small --load_all=1
```

### DDP Launcher (Simplified)
```bash
python launch_ddp.py --gpus=2 --preset=1 --plot=2 --size=small
```

### Train Only Latent Conditioner
```bash
python SimulGen-VAE.py --preset=1 --lc_only=1 --plot=2
```

### Key Command Arguments
- `--preset`: Dataset configuration (1-5, defined in preset.txt)
- `--plot`: Plotting mode (0=interactive, 1=save, 2=off)
- `--lc_only`: Train mode (0=full training, 1=LC only)
- `--size`: Model size (small/big)
- `--load_all`: Load all data to memory (0=lazy loading, 1=preload)
- `--use_ddp`: Enable distributed training

### No Standard Test/Lint Commands
This repository does not include standard testing or linting commands. The system relies on internal validation loops and checkpointing for quality assurance.

## Configuration System

### Primary Configuration Files

#### 1. `preset.txt` - Dataset Presets
- **Line 1**: Headers (data_No, init_beta_divisor, num_filter_enc, latent_conditioner_filter)
- **Line 2**: Preset number selection
- **Line 3**: Beta initialization divisor
- **Line 4**: Encoder filter sizes (e.g., "1024 512 256 128")
- **Line 5**: Latent conditioner filter sizes (e.g., "32 64 128 256")

#### 2. `input_data/condition.txt` - Comprehensive Training Parameters

**Common Parameters:**
```
Dim1: 484     # number of parameters
Dim2: 200     # timesteps
Dim3: 95008   # nodes
```

**VAE Training Parameters:**
```
Training_epochs: 10002
Batch_size: 16
LearningR: 0.001
Latent_dim: 8         # hierarchical latent dimension
Latent_dim_end: 32    # main latent dimension
Loss_type: 1          # 1: MSE, 2: MAE
```

**Latent Conditioner Parameters:**
```
n_epoch: 1000
latent_conditioner_lr: 0.0001
latent_conditioner_batch: 32
latent_conditioner_dropout_rate: 0.2
use_spatial_attention: 0           # 0=off, 1=on
input_type: image                  # image, csvs, image_vit
param_data_type: .png
```

**End-to-End Training Configuration:**
```
use_e2e_training: 1                # 0=disabled, 1=enabled
e2e_loss_function: MSE             # MSE, MAE, Huber, SmoothL1
e2e_vae_model_path: model_save/SimulGen-VAE
use_latent_regularization: 1       # 0=disabled, 1=enabled
latent_reg_weight: 1
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

## Training Modes

### 1. Full VAE Training (`lc_only=0`)
Trains both VAE encoder/decoder and latent conditioner simultaneously.

### 2. Latent Conditioner Only (`lc_only=1`)
Trains only the latent conditioner using a pre-trained VAE model.

### 3. End-to-End Training
Set `use_e2e_training=1` in `condition.txt` for direct data reconstruction optimization.

## Development Workflow

### Model Development Pattern
1. Configure parameters in `input_data/condition.txt`
2. Set dataset preset in `preset.txt`
3. Start with small model and single GPU for testing
4. Scale to multi-GPU for full training
5. Monitor via TensorBoard logs in output directory

### Key Implementation Notes
- Uses mixed precision training (autocast + GradScaler)
- Implements aggressive overfitting prevention
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
- Uses modern `torchrun` instead of deprecated `torch.distributed.launch`
- Automatic local_rank detection
- NCCL backend for multi-GPU communication
- Graceful fallback to single-GPU if DDP fails

## File Organization and Key Modules

### Entry Points
- `SimulGen-VAE.py`: Main training script with comprehensive argument parsing
- `launch_ddp.py`: Simplified DDP launcher with preset configurations

### Core Modules (`modules/` directory)
- `VAE_network.py`: Main VAE implementation with hierarchical encoder-decoder
- `encoder.py`: Hierarchical encoder with spectral normalization
- `decoder.py`: Progressive decoder with skip connections
- `train.py`: Advanced training pipeline with mixed precision
- `latent_conditioner.py`: Latent conditioning training logic
- `latent_conditioner_model_*.py`: Different conditioning architectures
- `data_preprocess.py`: Efficient data loading and preprocessing
- `augmentation.py`: Data augmentation for images
- `losses.py`: Custom loss functions
- `utils.py`: Utility functions and helpers
- `plotter.py`: Visualization and plotting utilities

### Configuration and Data
- `preset.txt`: Dataset and architecture presets
- `input_data/condition.txt`: Comprehensive training configuration
- `requirements.txt`: Python dependencies

## Development Guidelines for Claude

### What Claude Can Do
- Read and analyze code structure
- Explain implementation details
- Suggest improvements to code organization
- Help with documentation and comments
- Assist with configuration file modifications
- Debug logical issues in code
- Provide architectural advice

### What Claude Should NOT Do
- Execute training scripts
- Run GPU-dependent code
- Load large datasets or pickle files
- Attempt to train models locally
- Access training outputs or logs
- Try to visualize training results

### Working with Configuration Files
- Always ask user for current configuration before suggesting changes
- Understand the relationship between `preset.txt` and `condition.txt`
- Be aware of the different input types (image, csvs, image_vit)
- Consider memory constraints when suggesting batch sizes or model sizes

### Code Analysis Guidelines
- Focus on code structure and logic rather than execution
- Pay attention to distributed training setup
- Understand the hierarchical latent space design
- Be aware of the three different conditioning architectures
- Consider the anti-overfitting measures implemented

## Troubleshooting Common Issues

### Configuration Problems
- Check consistency between `preset.txt` and `condition.txt`
- Verify input_type matches available data
- Ensure dimensions match dataset specifications

### Memory Issues
- Suggest smaller batch sizes
- Recommend `--load_all=0` for memory efficiency
- Consider gradient checkpointing options

### Training Instability
- Check learning rates in configuration
- Verify dropout rates are appropriate
- Suggest early stopping parameters

### Distributed Training Issues
- Ensure proper GPU visibility
- Check NCCL backend configuration
- Verify torchrun usage instead of deprecated methods

This documentation provides comprehensive guidance for working with the SimulGenVAE codebase while respecting the hardware and execution limitations.