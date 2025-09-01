# SimulGenVAE v2.0.0

A high-performance Physics-Aware Variational Autoencoder system designed for fast generation and inference of transient/static simulation data with Physics-Aware Neural Network (PANN) integration.

## Features

- **Multi-GPU Support**: Single-GPU and distributed training (DDP) with automatic scaling
- **Multiple Architectures**: Three latent conditioning architectures (MLP, CNN, Vision Transformer)
- **Hierarchical Latent Space**: Main latent space (32D) with hierarchical dimensions (8D × layers)
- **Mixed Precision Training**: Optimized memory usage with gradient checkpointing support
- **Comprehensive Data Processing**: Advanced augmentation and validation pipelines
- **Flexible Input Types**: Support for simulation arrays, parametric CSV, and image data

## Supported Input Types

| Input Type | Format | Description |
|------------|--------|-------------|
| **Simulation Data** | 3D arrays [parameters, timesteps, nodes] | Primary physics simulation data |
| **Parametric Data** | CSV files | For MLP-based latent conditioning |
| **Image Data** | PNG/JPG files | For CNN/Vision Transformer conditioning |
| **PCA-processed** | Reduced dimensionality | Efficient MLP conditioning from images |

## Installation

### Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended)
- PyTorch with CUDA support

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Key Dependencies

- PyTorch + torchvision
- NumPy, pandas, scikit-learn
- matplotlib, OpenCV
- librosa, audiomentations
- TensorBoard for monitoring

## Quick Start

### Basic Training

```bash
# Single GPU training
python SimulGen-VAE.py --preset=1 --plot=2 --lc_only=0 --size=small

# Multi-GPU training
torchrun --nproc_per_node=4 SimulGen-VAE.py --use_ddp --preset=1

# Latent Conditioner only
python SimulGen-VAE.py --preset=1 --lc_only=1 --size=small
```

### Using the DDP Launcher

```bash
# Simplified multi-GPU launch
python launch_ddp.py --gpus=2 --preset=1 --plot=2 --size=small
```

## Configuration

### Preset System

Configure training parameters in `preset.txt`:
```
data_No, init_beta_divisior, num_filter_enc, latent_conditioner_filter
1
0
1024 512 256 128
8 16 32 64 96 128 160 192 160 128 96 64 48 32
```

### Detailed Configuration

Edit `input_data/condition.txt` for comprehensive settings:

```ini
# Dataset dimensions
Dim1        484     # number of parameters
Dim2        200     # number of timesteps  
Dim3        95008   # number of nodes

# VAE Training Parameters
Training_epochs     10002
Batch_size         8
LearningR          0.001
Latent_dim         8      # Hierarchical latent dimension
Latent_dim_end     32     # Main latent dimension
Loss_type          1      # 1:MSE, 2:MAE, 3:SmoothL1, 4:Huber

# Latent Conditioner Settings
input_type         image  # image, csvs
param_dir          /images
n_epoch            3000
latent_conditioner_lr    0.003
use_spatial_attention    1
```

## Command Line Arguments

| Argument | Description | Options |
|----------|-------------|---------|
| `--preset` | Dataset preset selection | 1-5 (reads from preset.txt) |
| `--plot` | Visualization mode | 0=interactive, 1=save, 2=off |
| `--lc_only` | Training mode | 0=full VAE, 1=LatentConditioner only |
| `--size` | Model architecture | small, large |
| `--load_all` | Memory strategy | 0=lazy loading, 1=preload all |
| `--use_ddp` | Enable distributed training | flag |

## Architecture Overview

### Core Components

```
SimulGen-VAE.py                 # Main training script
├── modules/
│   ├── VAE_network.py         # Core VAE with hierarchical latent space
│   ├── encoder.py             # Encoder architecture
│   ├── decoder.py             # Decoder architecture
│   ├── latent_conditioner_model_parametric.py  # MLP conditioning
│   ├── latent_conditioner_model_cnn.py         # CNN/ViT conditioning
│   ├── train.py               # VAE training loops
│   ├── latent_conditioner.py  # LC training
│   └── utils.py               # Core utilities & data loading
```

### Training Modes

1. **Full VAE Training** (`--lc_only=0`)
   - Trains complete VAE encoder/decoder + LatentConditioner
   - Best for new datasets or architecture changes

2. **Latent Conditioner Only** (`--lc_only=1`) 
   - Trains only LatentConditioner using pre-trained VAE
   - Efficient for parameter space exploration

3. **End-to-End Training**
   - Joint optimization of VAE and conditioning network
   - Configured via `use_e2e_training=1` in condition.txt

## Data Preparation

### Dataset Structure

```
input_data/
├── condition.txt              # Main configuration
├── dataset1.pickle           # Simulation data (3D arrays)
├── dataset2.pickle           # Additional datasets
└── images/                   # Image data for conditioning
    ├── param_001.png
    ├── param_002.png
    └── ...
```

### Data Format

- **Simulation Data**: Pickled NumPy arrays with shape `[parameters, timesteps, nodes]`
- **Image Data**: PNG/JPG files in `/images` directory 
- **CSV Data**: Parametric data for MLP conditioning

## Monitoring and Output

### Directory Structure

```
├── checkpoints/              # Model checkpoints during training
├── model_save/              # Final saved models
├── output/                  # Training logs and results
└── images/                  # Generated/reconstructed samples
```

### TensorBoard Integration

Monitor training progress:
```bash
tensorboard --logdir=output/
```

## Performance Optimization

### Memory Management
- Use `--load_all=0` for large datasets (lazy loading)
- Adjust batch sizes based on GPU memory
- Enable gradient checkpointing for memory efficiency

### Multi-GPU Training
- Automatic GPU detection and scaling
- Optimized data distribution across devices
- Synchronized batch normalization for stable training

## Examples

### Physics Simulation Training

```bash
# Train on simulation data with small model
python SimulGen-VAE.py --preset=1 --size=small --plot=1

# Large-scale distributed training
torchrun --nproc_per_node=8 SimulGen-VAE.py --use_ddp --preset=2 --size=large
```

### Image-Conditioned Generation

```bash
# Train with image conditioning
python SimulGen-VAE.py --preset=3 --lc_only=0 --size=small

# Focus on latent conditioner with pre-trained VAE  
python SimulGen-VAE.py --preset=3 --lc_only=1 --size=small
```

## Contributing

This project was developed for physics-aware simulation data processing. For questions or collaboration:

**Author**: SiHun Lee, Ph.D.  
**Contact**: kevin1007kr@gmail.com  
**Version**: 2.0.0 (Refactored & Documented)

## License

This project is available for research and educational purposes. Please cite appropriately if used in academic work.