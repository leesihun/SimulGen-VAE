# SimulGenVAE

A high-performance Variational Autoencoder (VAE) designed for simulation data generation and inference, featuring Physics-Aware Neural Network (PANN) integration and advanced latent conditioning capabilities.

## Overview

SimulGenVAE is a sophisticated deep learning framework for processing transient and static simulation data. It combines a hierarchical VAE architecture with multiple latent conditioning models to support three core tasks:

- **Parametric Estimation**: Multi-parameter regression using tabular data
- **Non-parametric Estimation**: Image and CAD-based inference
- **Probabilistic Estimation**: Uncertainty quantification and scattering analysis

## Key Features

### Core Architecture
- **Hierarchical VAE**: Multi-scale latent representations with encoder-decoder architecture
- **Latent Conditioning**: Three specialized models (MLP, CNN, Vision Transformer)
- **Physics Integration**: PANN support for physics-informed constraints
- **Multi-GPU Training**: Distributed Data Parallel (DDP) with automatic scaling

### Performance Optimizations
- **Mixed Precision Training**: FP16/BF16 support with gradient scaling
- **CUDA Graphs**: Accelerated computation for consistent operations
- **Memory Optimizations**: GPU data prefetching and pinned memory
- **Advanced Augmentation**: On-the-fly data augmentation with 5 techniques

### Anti-Overfitting Arsenal
- **SAM Optimizer**: Sharpness-Aware Minimization for flat minima
- **Progressive Dropout**: Adaptive regularization scheduling
- **Label Smoothing**: Regularized training targets
- **Ensemble Methods**: Snapshot ensembling and test-time augmentation

## Quick Start

### Installation

```bash
# Clone repository
git clone <repository-url>
cd SimulGenVAE

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```bash
# Single GPU training
python SimulGen-VAE.py --preset=1 --plot=2 --lc_only=0 --size=small --load_all=1

# Multi-GPU training (4 GPUs)
torchrun --nproc_per_node=4 SimulGen-VAE.py --use_ddp --preset=1 --plot=2 --lc_only=0 --size=small --load_all=1

# Train only LatentConditioner
python SimulGen-VAE.py --lc_only=1 --preset=1
```

## Architecture

### VAE Network (`modules/VAE_network.py`)
- **Encoder**: Hierarchical compression with GroupNorm layers
- **Decoder**: Progressive reconstruction with skip connections
- **Loss Functions**: MSE, MAE, SmoothL1, Huber loss support

### Latent Conditioning Models

#### 1. MLP Model (`latent_conditioner_model_parametric.py`)
- **Purpose**: Parametric/tabular data processing
- **Architecture**: Multi-layer perceptron with GELU activations
- **Features**: Progressive dropout, extreme bottleneck compression

#### 2. CNN Model (`latent_conditioner_model_cnn.py`)
- **Purpose**: Image data processing (traditional approach)
- **Architecture**: ResNet-style blocks with GroupNorm
- **Features**: Spatial attention, adaptive pooling

#### 3. Vision Transformer (`latent_conditioner_model_vit.py`)
- **Purpose**: Image data processing (modern transformer approach)
- **Architecture**: Patch-based attention mechanism
- **Specifications**:
  - Input: 128×128 images → 64 patches (16×16 each)
  - 2 transformer layers, 4 attention heads
  - 64 embedding dimensions
  - Global average pooling (no CLS token)

## Configuration

### Command Line Arguments

| Flag | Description | Example |
|------|-------------|---------|
| `--preset` | Dataset preset selection | `--preset=1` |
| `--plot` | Visualization mode (1=show, 2=off) | `--plot=2` |
| `--lc_only` | Train only LatentConditioner (1=yes, 0=no) | `--lc_only=0` |
| `--size` | Network size preset | `--size=small` |
| `--load_all` | Preload dataset to GPU | `--load_all=1` |
| `--use_ddp` | Enable distributed training | `--use_ddp` |

### Data Configuration (`input_data/condition.txt`)

Key parameters:
```
# VAE Parameters
Training_epochs    10002
Batch_size        16
LearningR         0.0005
Latent_dim        8      # Hierarchical latent dimension
Latent_dim_end    32     # Main latent dimension

# LatentConditioner Parameters
latent_conditioner_lr           0.00001
latent_conditioner_batch        64
latent_conditioner_weight_decay 1e-4
input_type                      image    # Options: image, csvs, image_vit
```

### Model Selection
Choose latent conditioning architecture by setting `input_type`:
- `csvs`: MLP for parametric data
- `image`: CNN for image data
- `image_vit`: Vision Transformer for image data

## Data Format

### VAE Dataset
- **Format**: Pickle file containing 3D NumPy array
- **Shape**: `[num_parameters, num_timesteps, num_nodes]`
- **Example**: `dataset1.pickle` with shape `[484, 200, 95008]`

### LatentConditioner Dataset
- **Images**: JPG/PNG files in `/images` directory
- **Parametric**: CSV files with feature vectors
- **Resolution**: 256×256 for image inputs (resized to 128×128 internally)

## Advanced Features

### Data Augmentation
Five on-the-fly augmentation techniques:
1. **Noise Injection**: Gaussian noise (configurable intensity)
2. **Amplitude Scaling**: Random scaling (±10% default)
3. **Time Shifting**: Temporal shifts (10% max default)
4. **Mixup**: Linear interpolation between samples
5. **Cutout**: Random masking of segments

### Normalization Strategy
- **GroupNorm**: Batch-size independent normalization
- **Adaptive Groups**: `min(8, max(1, channels//4))`
- **Benefits**: Better DDP compatibility, stable small-batch training

### Anti-Overfitting Measures
- **SAM Optimizer**: Sharpness-Aware Minimization (rho=0.05)
- **EMA Weights**: Exponential moving average (decay=0.999)
- **Progressive Dropout**: 30% → 10% over training
- **Information Bottleneck**: SVD regularization on weights
- **Gradient Penalty**: L2 penalty on gradients (after epoch 10)

## Multi-GPU Training

### Setup
```bash
# Set environment variables
export NCCL_DEBUG=INFO
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Launch training
torchrun --nproc_per_node=4 SimulGen-VAE.py --use_ddp [other_args]
```

### Performance Scaling
- **2 GPUs**: ~1.8x speedup
- **4 GPUs**: ~3.5x speedup
- **8 GPUs**: ~6.5x speedup

### Batch Size Scaling
Scale batch size linearly with GPU count:
- Small model: 16 → 32 (2 GPUs) → 64 (4 GPUs)
- Large model: 8 → 16 (2 GPUs) → 32 (4 GPUs)

## Monitoring and Debugging

### TensorBoard
```bash
tensorboard --logdir=runs --port=6006
```

### GPU Monitoring
```bash
watch -n 0.5 nvidia-smi
```

### Common Issues

| Issue | Solution |
|-------|----------|
| **CUDA OOM** | Reduce `Batch_size` in `condition.txt` |
| **NaN losses** | Lower gradient clipping (5.0 → 1.0) |
| **DDP hangs** | Set `export NCCL_TIMEOUT=1800` |
| **Slow first epoch** | Expected with model compilation |
| **Import errors** | Check all module dependencies |

## File Structure

```
SimulGenVAE/
├── SimulGen-VAE.py              # Main training script
├── requirements.txt             # Dependencies
├── input_data/
│   └── condition.txt           # Configuration parameters
├── modules/
│   ├── VAE_network.py          # Core VAE architecture
│   ├── encoder.py              # VAE encoder
│   ├── decoder.py              # VAE decoder
│   ├── latent_conditioner.py   # Training utilities
│   ├── latent_conditioner_model_parametric.py  # MLP model
│   ├── latent_conditioner_model_cnn.py         # CNN model
│   ├── latent_conditioner_model_vit.py         # ViT model
│   ├── train.py                # Training loop
│   ├── losses.py               # Loss functions
│   ├── augmentation.py         # Data augmentation
│   └── common.py               # Shared utilities
└── checkpoints/                # Model checkpoints
└── output/                     # Generated results
```

## Requirements

- **Python**: ≥3.9 (tested on 3.10)
- **PyTorch**: ≥2.0 with CUDA support
- **GPU**: CUDA-capable with ≥12GB memory
- **Multi-GPU**: NCCL backend support

### Dependencies
```
torch
numpy
pandas
matplotlib
torchinfo
scikit-learn
opencv-python
tensorboard
torchvision
pytorch_warmup
scikit-image
```

## Performance Tips

1. **Memory Optimization**: Use `--load_all=1` for small datasets
2. **Compilation**: Enable `torch.compile` for repeated operations
3. **Mixed Precision**: Automatic FP16 training on compatible GPUs
4. **CUDA Graphs**: Enabled for consistent operation shapes
5. **Data Loading**: Pinned memory with async transfers

## Version History

- **v1.5.0**: Modular architecture with multi-model support
- **v1.4.4**: Comprehensive anti-overfitting arsenal
- **v1.4.3**: LatentConditioner architecture simplification
- **v1.4.0**: ResNet-style blocks with SE attention
- **v1.3.0**: On-the-fly data augmentation
- **v1.2.0**: GroupNorm normalization strategy

## Author

**SiHun Lee, Ph.D.**
- Email: kevin1007kr@gmail.com
- LinkedIn: [Profile](https://www.linkedin.com/in/시훈-이-13009a172/?originalSubdomain=kr)

## License

This project is provided as-is for research and educational purposes.