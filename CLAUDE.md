# CLAUDE.md

This file provides comprehensive guidance to Claude Code (claude.ai/code) when working with the SimulGenVAE codebase.

## ⚠️ Critical Safety Precautions

**NEVER attempt to:**
- Execute training scripts or GPU-dependent code
- Run model training, evaluation, or inference
- Load large datasets, pickle files, or model checkpoints
- Access training outputs, logs, or generated data
- Attempt CUDA operations or GPU memory operations

**Always ask the user** for information about:
- Training results, model performance, or validation metrics
- Dataset contents, structure, or statistics  
- Hardware specifications or system configurations
- Running training sessions or experiment results

## 📋 Project Overview

SimulGenVAE v2.0.0 is a high-performance Variational Autoencoder system designed for physics simulation data processing. The system features:

- **Hierarchical VAE Architecture**: Dual-level latent representation (32D main + 8D hierarchical)
- **Three Conditioning Architectures**: MLP, CNN, and Vision Transformer for different input types
- **Advanced Training Pipeline**: DDP support, mixed precision, gradient checkpointing
- **Physics-Aware Design**: Optimized for temporal and spatial simulation data
- **Production-Ready**: Comprehensive error handling, monitoring, and optimization

### Version Information
- **Version**: 2.0.0 (Refactored & Documented)
- **Author**: SiHun Lee, Ph.D.
- **Contact**: kevin1007kr@gmail.com
- **Architecture**: Hierarchical encoder-decoder with external parameter conditioning

## 🏗 Core Architecture Components

### Main Components
1. **VAE Network** (`modules/VAE_network.py`): Main Variational Autoencoder class
2. **Hierarchical Encoder** (`modules/encoder.py`): Multi-scale temporal-spatial compression
3. **Hierarchical Decoder** (`modules/decoder.py`): Progressive reconstruction with skip connections
4. **Latent Conditioner System**: Three architectural variants for parameter conditioning
5. **Training Pipeline** (`modules/train.py`): Advanced training with validation and scheduling
6. **Data Processing** (`modules/data_preprocess.py`): Loading, scaling, and augmentation

### Latent Conditioning Architectures

#### 1. MLP-based (`latent_conditioner_model_parametric.py`)
- **Purpose**: Numerical/tabular parametric data processing
- **Input**: CSV files with simulation parameters
- **Architecture**: Enhanced multi-layer perceptron with residual connections
- **Features**: Progressive dropout, layer normalization, GELU activations
- **Use Case**: Parameter studies, optimization results

#### 2. CNN-based (`latent_conditioner_model_cnn.py`)
- **Purpose**: 2D image data processing (geometry, boundary conditions)
- **Input**: PNG/JPG images automatically resized to 256×256
- **Architecture**: ResNet-style blocks with GroupNorm and spectral normalization
- **Features**: Squeeze-and-Excitation attention, adaptive pooling, SiLU activations
- **Use Case**: Geometric shapes, boundary condition visualization

#### 3. Vision Transformer (`latent_conditioner_model_vit.py`)
- **Purpose**: Complex image analysis with transformer architecture
- **Input**: High-resolution images requiring attention mechanisms
- **Architecture**: Patch-based embedding (16×16) with multi-head self-attention
- **Features**: Position embeddings, stochastic depth, global average pooling
- **Use Case**: Complex spatial patterns, multi-scale geometric features

## ⚙️ Configuration System

### Primary Configuration Files

#### `preset.txt` - Architecture and Dataset Presets
```
Line 1: Headers (data_No, init_beta_divisor, num_filter_enc, latent_conditioner_filter)
Line 2: Preset selection number (1-5)
Line 3: Beta initialization divisor for KL annealing
Line 4: Encoder filter progression (e.g., "1024 512 256 128")
Line 5: Latent conditioner filter progression (e.g., "4 8 16 32 64")
```

#### `input_data/condition.txt` - Comprehensive Training Parameters

**Data Dimensions:**
```
Dim1: 484      # Number of simulation parameters
Dim2: 200      # Number of timesteps
Dim3: 95008    # Number of nodes/spatial points
num_var: 1     # Number of variables per node
```

**VAE Training Configuration:**
```
Training_epochs: 10002        # Total training epochs
Batch_size: 16               # Batch size for VAE training
LearningR: 0.001             # VAE learning rate
Latent_dim: 8                # Hierarchical latent dimension
Latent_dim_end: 32           # Main latent dimension  
Loss_type: 1                 # 1=MSE, 2=MAE, 3=SmoothL1, 4=Huber
alpha: 1000000               # KL divergence weight scaling
```

**Latent Conditioner Configuration:**
```
n_epoch: 10000                        # Conditioning training epochs
latent_conditioner_lr: 0.001          # Conditioner learning rate
latent_conditioner_batch: 16          # Conditioner batch size
latent_conditioner_weight_decay: 1e-4 # L2 regularization
latent_conditioner_dropout_rate: 0.3  # Dropout probability
use_spatial_attention: 0              # CNN spatial attention (0=off, 1=on)
input_type: image                     # image, csv, image_vit
param_data_type: .png                 # File extension for input data
param_dir: /images                    # Directory containing input data
```

**End-to-End Training Configuration:**
```
use_e2e_training: 1                   # Enable end-to-end training (0=off, 1=on)
e2e_loss_function: Huber              # E2E loss: MSE, MAE, Huber, SmoothL1
e2e_vae_model_path: model_save/SimulGen-VAE  # Path to pre-trained VAE
use_latent_regularization: 1          # Enable latent regularization (0=off, 1=on)
latent_reg_weight: 0.001              # Weight for latent regularization term
```

## 🚀 Training Modes and Execution

### Command-Line Interface

**Main Entry Point**: `SimulGen-VAE.py`
```bash
python SimulGen-VAE.py [OPTIONS]
```

**Key Arguments:**
- `--preset`: Dataset preset (1-5, default: 1)
- `--plot`: Visualization (0=interactive, 1=save, 2=off, default: 2)  
- `--lc_only`: Training mode (0=full VAE, 1=conditioner only, default: 0)
- `--size`: Model size (small/large, default: small)
- `--load_all`: Memory strategy (0=lazy loading, 1=preload all, default: 0)
- `--use_ddp`: Enable distributed training (flag)

### Training Modes

#### 1. Full VAE Training (`lc_only=0`)
**Complete end-to-end training pipeline:**
- Trains VAE encoder-decoder with hierarchical latent space
- Simultaneous latent conditioner training
- KL annealing schedule with warmup and cosine scheduling
- Validation monitoring with early stopping
- Model checkpointing and best model saving

**Example Command:**
```bash
python SimulGen-VAE.py --preset=1 --lc_only=0 --size=small --load_all=1
```

**Generated Outputs:**
- `model_save/SimulGen-VAE`: Complete trained VAE model
- `model_save/latent_vectors.npy`: Main latent representations [N, 32]
- `model_save/xs.npy`: Hierarchical latent representations [N, layers, 8]
- Various scaler files for data normalization

#### 2. Latent Conditioner Only (`lc_only=1`) 
**Conditioner-only training using pre-trained VAE:**
- Loads existing VAE from `model_save/SimulGen-VAE`
- Trains only the conditioning network (MLP/CNN/ViT)
- Uses pre-computed latent vectors as targets
- Faster iteration for parameter studies

**Example Command:**
```bash
python SimulGen-VAE.py --preset=1 --lc_only=1 --plot=2 --size=small
```

**Prerequisites:**
- Pre-trained VAE model must exist in `model_save/`
- Latent vectors must be pre-computed (`latent_vectors.npy`, `xs.npy`)

#### 3. End-to-End Training (`use_e2e_training=1`)
**Direct reconstruction optimization:**
- Condition Input → Latent Conditioner → VAE Decoder → Reconstruction
- Unified loss combining reconstruction error and latent regularization
- Bypasses intermediate latent space supervision
- Optimal for deployment scenarios

**Configuration in `condition.txt`:**
```
use_e2e_training: 1
e2e_loss_function: Huber
latent_reg_weight: 0.001
```

### Distributed Training

#### Using torchrun (Recommended)
```bash
# Single node, multiple GPUs
torchrun --nproc_per_node=4 SimulGen-VAE.py --use_ddp --preset=1

# Multi-node distributed training
torchrun --nnodes=2 --node_rank=0 --master_addr="192.168.1.1" \
         --nproc_per_node=4 SimulGen-VAE.py --use_ddp --preset=1
```

#### Using DDP Launcher
```bash
python launch_ddp.py --gpus=4 --preset=1 --size=large --master_port=29500
```

**DDP Features:**
- Automatic batch size scaling across GPUs
- NCCL backend for efficient communication
- Gradient synchronization and averaging
- Rank-aware logging and checkpointing

## 📊 Data Organization and Processing

### Expected Directory Structure
```
SimulGenVAE/
├── input_data/
│   ├── condition.txt          # Main configuration file
│   └── dataset*.pickle        # Pickled simulation data [N, T, M]
├── images/                    # Conditioning images (256×256 PNG/JPG)
├── model_save/               # Trained models and latent vectors
├── checkpoints/              # Training state checkpoints
├── output/                   # Training plots and logs
└── modules/                  # Core implementation modules
```

### Data Format Specifications

#### Simulation Data Format
**File**: `input_data/dataset*.pickle`
**Shape**: `[num_parameters, num_timesteps, num_nodes]`
**Type**: Float32 NumPy arrays
**Processing**: Automatic scaling to [-0.7, 0.7] range for VAE optimization

#### Image Data Format  
**Directory**: `images/`
**Format**: PNG/JPG files, automatically resized to 256×256
**Naming**: Sequential (param_001.png, param_002.png, etc.)
**Processing**: Normalized to [0, 1] range, optional augmentation

#### CSV Data Format
**Directory**: Specified by `param_dir` in condition.txt
**Format**: Standard CSV with headers
**Processing**: MinMaxScaler normalization, saved as pickle for reuse

### Data Processing Pipeline

1. **Loading** (`input_variables.py`): Configuration parsing and dataset loading
2. **Reduction** (`data_preprocess.py`): Temporal/spatial dimension reduction  
3. **Augmentation**: On-the-fly augmentation preserving physics constraints
4. **Scaling**: MinMaxScaler with consistent ranges across train/validation
5. **DataLoader Creation**: Optimized loaders with memory management

### Memory Management Strategies

#### Lazy Loading (`load_all=0`)
- **Benefits**: Reduced GPU memory usage, supports larger datasets
- **Trade-offs**: Slower data loading, I/O bottlenecks
- **Use Case**: Limited GPU memory, very large datasets

#### GPU Preloading (`load_all=1`) 
- **Benefits**: Faster training, reduced I/O overhead
- **Trade-offs**: High GPU memory usage, limited dataset size
- **Use Case**: Sufficient GPU memory, performance-critical training

## 🔧 Development and Debugging Guidelines

### Code Analysis Approach

#### What Claude Can Help With
1. **Code Structure Analysis**: Understanding module organization and dependencies
2. **Configuration Guidance**: Explaining parameter relationships and effects
3. **Architecture Review**: Analyzing model designs and data flow
4. **Bug Investigation**: Reviewing code logic and identifying potential issues
5. **Documentation**: Creating/updating comments and documentation
6. **Optimization Suggestions**: Identifying performance improvement opportunities

#### Configuration File Relationships
- `preset.txt` controls architecture dimensions (filters, layers)
- `condition.txt` controls training parameters and data paths
- Both must be consistent for successful training
- Dimension mismatches between config and data cause runtime errors

### Common Configuration Patterns

#### Small vs Large Model Selection
```python
# Small model (default): Faster training, less memory
--size=small
# Encoder: [1024, 512, 256, 128] filters
# Memory: ~8-12GB GPU memory

# Large model: Better capacity, more memory
--size=large  
# Encoder: Larger filter counts and additional layers
# Memory: ~16-32GB GPU memory
```

#### Input Type Configuration
```python
# For image inputs (geometry, boundaries)
input_type: image
param_data_type: .png
param_dir: /images

# For parametric inputs (simulation parameters)
input_type: csv  
param_data_type: .csv
param_dir: /parametric_data

# For Vision Transformer
input_type: image_vit
param_data_type: .png  
param_dir: /images
```

### Debugging Common Issues

#### Configuration Inconsistencies
```python
# Check dimension consistency
Dim1 (parameters) == len(simulation_data)
Dim2 (timesteps) == simulation_data.shape[1] 
Dim3 (nodes) == simulation_data.shape[2]

# Check filter progression validity
num_filter_enc: Must be decreasing sequence
latent_conditioner_filter: Must match conditioning architecture
```

#### Memory-Related Problems
```python
# Batch size too large
Batch_size: 16 → 8 → 4  # Progressive reduction

# Enable lazy loading  
--load_all=0

# Use smaller model
--size=small
```

#### Training Instability
```python
# Learning rate too high
LearningR: 0.001 → 0.0001

# KL annealing too aggressive  
init_beta_divisor: 0 → 2 → 4  # Slower KL annealing

# Insufficient regularization
latent_conditioner_dropout_rate: 0.3 → 0.5
```

### File Organization Understanding

#### Entry Points
- `SimulGen-VAE.py`: Main training script with comprehensive CLI
- `launch_ddp.py`: Simplified distributed training launcher

#### Core Modules (`modules/` directory)
- `VAE_network.py`: Main VAE class with forward/backward passes
- `encoder.py`/`decoder.py`: Hierarchical encoder-decoder architectures  
- `train.py`: Training loop with validation, checkpointing, scheduling
- `latent_conditioner*.py`: Three conditioning architecture variants
- `data_preprocess.py`: Data loading, scaling, augmentation pipeline
- `utils.py`: Utility functions, datasets, distributed training setup
- `losses.py`: KL divergence computation with numerical stability
- `common.py`: Weight initialization, spectral normalization utilities

#### Support Files
- `augmentation.py`: Physics-preserving data augmentation
- `plotter.py`: Visualization utilities for training monitoring
- `reconstruction_evaluator.py`: Model evaluation and comparison tools

## 🛠 Advanced Features and Optimizations

### Training Pipeline Features

#### KL Annealing Schedule
```python
# Progressive KL weight increase during training
# Epochs 0 - 0.3*total: beta = init_beta (very small)
# Epochs 0.3*total - 0.8*total: beta increases to 1.0
# Epochs 0.8*total - total: beta = 1.0 (full KL weight)
```

#### Learning Rate Scheduling
```python
# CosineAnnealingWarmRestarts with warmup
# Initial warmup phase for stable convergence
# Cosine annealing with restarts for better optimization
```

#### Validation and Early Stopping
```python
# Validation loss monitoring every 10 epochs
# Best model checkpointing based on validation loss
# Overfitting detection via train/validation ratio
# Early stopping available (configurable patience)
```

### Performance Optimization Features

#### Memory Optimizations
- **Mixed Precision**: Available but disabled by user preference
- **Gradient Checkpointing**: Available for memory-constrained scenarios
- **Channels-Last Memory Format**: Better performance on modern hardware
- **Memory Pinning**: Optimized GPU-CPU data transfers

#### Computational Optimizations
- **Model Compilation**: `torch.compile` support for PyTorch 2.0+
- **Optimal Worker Selection**: Automatic DataLoader worker optimization
- **CUDA Graphs**: Available for ultimate performance (experimental)
- **Non-blocking Transfers**: Asynchronous GPU operations

### Distributed Training Features

#### DDP Configuration
- **Automatic Rank Detection**: Uses `torchrun` local rank assignment
- **Gradient Synchronization**: Automatic all-reduce across GPUs
- **Model Replication**: Consistent model states across processes
- **Batch Size Scaling**: Automatic per-GPU batch size adjustment

#### Communication Backend
- **NCCL Backend**: Optimized GPU-GPU communication
- **TCP Fallback**: CPU-based communication when NCCL unavailable
- **Port Management**: Configurable master port for multi-job scenarios

## 🔍 Monitoring and Evaluation

### Training Monitoring
```python
# TensorBoard integration for real-time monitoring
# Metrics logged: losses, KL components, learning rates
# Memory usage tracking and reporting
# Validation metrics and overfitting detection
```

### Model Evaluation Tools
```python
# ReconstructionEvaluator: Quantitative reconstruction assessment
# Latent space analysis: PCA, t-SNE visualization
# Conditioning effectiveness: Input-output correlation analysis
# Physics constraint validation: Conservation law checking
```

### Output Interpretation
```python
# Training logs: Loss curves, convergence indicators
# Model checkpoints: Best validation loss models
# Latent representations: Saved for downstream analysis
# Reconstruction comparisons: Visual validation plots
```

## 🚨 Common Pitfalls and Solutions

### Configuration Errors
1. **Dimension Mismatches**: Ensure config dimensions match actual data
2. **Path Issues**: Verify all data paths exist and are accessible
3. **Memory Overcommitment**: Balance batch_size and load_all settings
4. **Architecture Conflicts**: Check filter progressions are valid

### Training Issues  
1. **NaN Losses**: Usually indicates learning rate too high or data issues
2. **No Convergence**: Check KL annealing schedule and learning rates
3. **Overfitting**: Increase dropout, reduce model size, add regularization
4. **Memory Errors**: Reduce batch size, use lazy loading, smaller model

### Distributed Training Issues
1. **Communication Timeouts**: Check network connectivity, firewall settings
2. **Rank Synchronization**: Ensure consistent configuration across nodes  
3. **Memory Imbalance**: Verify consistent GPU memory across devices
4. **Port Conflicts**: Use different master ports for concurrent jobs

## 📋 Development Workflow

### Typical Development Pattern
1. **Configuration Setup**: Modify `condition.txt` and `preset.txt` 
2. **Data Preparation**: Organize datasets in expected directory structure
3. **Small-Scale Testing**: Start with small model, single GPU
4. **Hyperparameter Tuning**: Adjust learning rates, batch sizes, architecture
5. **Full-Scale Training**: Scale to multi-GPU, large model as needed
6. **Validation and Analysis**: Evaluate results, compare configurations

### Best Practices for Configuration Changes
- **Always backup** working configurations before modifications
- **Test incrementally** with single GPU before distributed training
- **Monitor early epochs** for convergence indicators and stability
- **Document changes** in comments or separate notes
- **Version control** configuration files alongside code

### Debugging Workflow
1. **Check logs** for specific error messages and stack traces
2. **Verify data** shapes and ranges match configuration expectations
3. **Test components** individually (VAE only, conditioner only)
4. **Reduce complexity** temporarily (smaller batch, single GPU)
5. **Compare** with known working configurations

This documentation provides comprehensive guidance for working with SimulGenVAE while respecting the critical safety constraints around execution and hardware requirements.