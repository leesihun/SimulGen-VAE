# CLAUDE.md

This file provides comprehensive guidance to Claude Code (claude.ai/code) when working with the SimulGenVAE codebase.

## Project Overview

SimulGenVAE v2.0.0 is a high-performance Physics-Aware Variational Autoencoder system designed for fast generation and inference of transient/static simulation data. The system supports both single-GPU and multi-GPU distributed training with comprehensive data augmentation, validation, and three distinct latent conditioning architectures.

### Core Architecture

The system implements a hierarchical VAE with:
- **Main latent space**: 32-dimensional primary representation
- **Hierarchical latent space**: 8-dimensional auxiliary representation with configurable layers
- **Three conditioning architectures**: MLP (parametric), CNN (image), and Vision Transformer (image)
- **Physics-Aware Neural Networks (PANN)** integration for domain-specific constraints

## Key Training Commands

### Single GPU Training
```bash
# Full VAE training (encoder + decoder + latent conditioner)
python SimulGen-VAE.py --preset=1 --plot=2 --lc_only=0 --size=small

# Latent conditioner only training (using pre-trained VAE)
python SimulGen-VAE.py --preset=1 --lc_only=1 --size=small
```

### Multi-GPU Distributed Training
```bash
# Direct torchrun approach
torchrun --nproc_per_node=4 SimulGen-VAE.py --use_ddp --preset=1

# Using convenience launcher script
python launch_ddp.py --gpus=2 --preset=1 --plot=2 --size=small
```

## Project Structure and Key Files

### Entry Points
- **`SimulGen-VAE.py`**: Main training script with comprehensive CLI
- **`launch_ddp.py`**: Distributed training launcher with simplified configuration

### Core Architecture Modules
- **`modules/VAE_network.py`**: Core VAE implementation with hierarchical latent spaces
- **`modules/encoder.py`**: Hierarchical encoder networks with progressive feature extraction
- **`modules/decoder.py`**: Hierarchical decoder networks with upsampling and reconstruction
- **`modules/train.py`**: Main training loop with warmup, loss scheduling, and validation

### Latent Conditioning Models
- **`modules/latent_conditioner_model_parametric.py`**: MLP-based conditioning for tabular/CSV data
- **`modules/latent_conditioner_model_cnn.py`**: CNN-based conditioning for images with ResNet backbone
- **`modules/latent_conditioner.py`**: Training orchestrator for all conditioning approaches
- **`modules/latent_conditioner_e2e.py`**: End-to-end training pipeline
- **`modules/enhanced_latent_conditioner_training.py`**: Advanced training features and optimizations

### Data Processing and Utilities
- **`modules/data_preprocess.py`**: Dataset reduction, normalization, and scaling
- **`modules/augmentation.py`**: Comprehensive data augmentation pipelines
- **`modules/pca_preprocessor.py`**: PCA dimensionality reduction for efficient processing
- **`modules/utils.py`**: Core utilities including dataset loading and distributed training setup
- **`modules/input_variables.py`**: Configuration parsing and validation

### Loss Functions and Optimization
- **`modules/losses.py`**: KL divergence and reconstruction loss implementations
- **`modules/enhanced_loss_functions.py`**: Advanced loss functions including Huber and SmoothL1
- **`modules/common.py`**: Shared utilities including weight initialization and spectral normalization

### Visualization and Evaluation
- **`modules/plotter.py`**: Comprehensive visualization tools for training monitoring
- **`modules/reconstruction_evaluator.py`**: Model evaluation metrics and validation tools

## Configuration System

### Preset Configuration (`preset.txt`)
Format: `data_No, init_beta_divisor, num_filter_enc, latent_conditioner_filter`
- Line 1: Header description
- Line 2: Dataset number (1-5)
- Line 3: Beta divisor for KL loss warmup (0 = no warmup)
- Line 4: VAE encoder filter progression (space-separated, e.g., "1024 512 256 128")
- Line 5: Latent conditioner filter progression (space-separated, e.g., "8 16 32 64 128 256 128")

### Primary Configuration (`input_data/condition.txt`)

#### Data Dimensions
- **Dim1**: Number of parameters (e.g., 484)
- **Dim2**: Number of timesteps (e.g., 200)  
- **Dim3**: Number of nodes (e.g., 95008)
- **num_var**: Number of variables (typically 1)

#### VAE Training Parameters
- **Training_epochs**: Total training epochs (e.g., 10002)
- **Batch_size**: Training batch size (e.g., 16)
- **LearningR**: Learning rate (e.g., 0.001)
- **Latent_dim**: Hierarchical latent dimension (e.g., 8)
- **Latent_dim_end**: Main latent dimension (e.g., 32)
- **Loss_type**: Reconstruction loss (1=MSE, 2=MAE, 3=SmoothL1, 4=Huber)
- **alpha**: KL loss scaling factor (e.g., 1000000)

#### Latent Conditioner Parameters
- **num_param**: Number of conditioning parameters
- **param_dir**: Data directory (e.g., "/images")
- **input_type**: Data type ("image" or "csvs")
- **param_data_type**: File extension (e.g., ".png")
- **n_epoch**: Conditioner training epochs (e.g., 10000)
- **latent_conditioner_lr**: Conditioner learning rate (e.g., 0.001)
- **latent_conditioner_batch**: Conditioner batch size (e.g., 16)
- **use_spatial_attention**: Enable spatial attention (1=enabled)

#### End-to-End Training
- **use_e2e_training**: Enable end-to-end training (1=enabled)
- **e2e_loss_function**: E2E loss type (MSE, MAE, Huber, SmoothL1)
- **e2e_vae_model_path**: Path to pre-trained VAE model
- **use_latent_regularization**: Enable latent regularization (1=enabled)
- **latent_reg_weight**: Regularization weight (e.g., 0.001)

## Command Line Arguments

### Core Arguments
- **`--preset`**: Dataset preset selection (1-5, reads from preset.txt)
- **`--plot`**: Visualization mode (0=interactive, 1=save plots, 2=disable)
- **`--lc_only`**: Training mode (0=full VAE, 1=latent conditioner only)
- **`--size`**: Model architecture (small=memory efficient, large=full performance)
- **`--load_all`**: Memory mode (0=lazy loading, 1=preload all data)
- **`--use_ddp`**: Enable distributed data parallel training

## Data Input Types and Structure

### Simulation Data (Primary)
- **Format**: Pickled 3D numpy arrays (dataset*.pickle)
- **Structure**: [parameters, timesteps, nodes]
- **Location**: Root directory alongside scripts

### Image Data (Conditioning)
- **Format**: PNG/JPG files
- **Location**: `/images` directory
- **Processing**: Automatic resizing to 256×256 for CNN conditioning

### Parametric Data (Conditioning)
- **Format**: CSV files
- **Processing**: PCA reduction for efficient MLP conditioning
- **Features**: Numerical parameters for direct MLP input

## Training Workflows

### Full VAE Training (`lc_only=0`)
1. Initialize VAE encoder and decoder networks
2. Load simulation data with optional augmentation
3. Train VAE with KL warmup and reconstruction loss
4. Monitor training through TensorBoard logs
5. Save checkpoints and final model

### Latent Conditioner Only Training (`lc_only=1`)
1. Load pre-trained VAE model (frozen)
2. Initialize appropriate conditioning model (MLP/CNN/ViT)
3. Train conditioner to predict latent representations
4. Validate against VAE reconstructions
5. Save trained conditioner model

### End-to-End Training
1. Load pre-trained VAE and conditioner models
2. Fine-tune both networks jointly
3. Apply latent regularization for consistency
4. Optimize for downstream task performance

## Development Best Practices

### Model Checkpointing
- Checkpoints saved to `checkpoints/` directory during training
- Final models saved to `model_save/` directory
- Models include optimizer state for resuming training

### Logging and Monitoring
- TensorBoard logs written to `output/` directory
- Comprehensive loss tracking and visualization
- GPU memory and training speed monitoring

### Memory Management
- Lazy loading option for large datasets (`load_all=0`)
- Gradient checkpointing for memory efficiency
- Mixed precision support in newer versions

### Distributed Training
- Automatic GPU detection and utilization
- Synchronized batch normalization across GPUs
- Gradient synchronization and scaling

## File Organization

### Input Data Structure
```
input_data/
├── condition.txt          # Primary configuration
└── [additional configs]   # Dataset-specific settings
```

### Output Structure  
```
output/                    # Training logs and outputs
checkpoints/              # Model checkpoints during training
model_save/               # Final trained models
images/                   # Image data for conditioning (optional)
```

### Development Utilities
- **`git-auto-push.sh`**: Automated git commits with timestamps
- **`requirements.txt`**: Python dependencies specification

## Important Notes for Development

### Memory Considerations
- Large simulation datasets require careful memory management
- Use `size=small` for development and limited memory systems
- Monitor GPU memory usage during distributed training

### Configuration Validation
- Always validate preset.txt format before training
- Ensure data dimensions match between condition.txt and actual data
- Verify file paths exist before starting training

### Model Compatibility
- Latent conditioner models are tied to specific VAE architectures
- Ensure consistent latent dimensions across components
- Save and load complete model states for reproducibility

### Performance Optimization
- Use distributed training for large datasets and models
- Enable mixed precision when available
- Consider data augmentation impact on training time

## Troubleshooting Common Issues

### Memory Issues
- Reduce batch size in condition.txt
- Use `size=small` architecture
- Enable lazy loading (`load_all=0`)

### Training Instability
- Adjust learning rates in condition.txt
- Check KL warmup settings (beta divisor)
- Verify data normalization and scaling

### Distributed Training Problems  
- Ensure consistent CUDA versions across nodes
- Check network connectivity for multi-node setups
- Verify proper port configuration in launch scripts