# SimulGenVAE v1.4.3

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-1.9%2B-orange)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-Academic-green)](LICENSE)

**SimulGenVAE** is a high-performance Variational Autoencoder system designed for fast generation and inference of transient/static simulation data with Physics-Aware Neural Network (PANN) integration. The system supports both single-GPU and multi-GPU distributed training with three different latent conditioning architectures.

## 🚀 Key Features

- **Hierarchical VAE Architecture**: Two-level latent representation (32D main + 8D hierarchical)
- **Multi-Modal Conditioning**: Support for parametric data, images, and Vision Transformer inputs
- **Distributed Training**: Modern PyTorch DDP with automatic scaling
- **Advanced Anti-Overfitting**: Comprehensive regularization suite (dropout, augmentation, early stopping)
- **Mixed Precision Training**: 40% memory reduction with autocast + GradScaler
- **Physics-Aware Design**: Specialized for simulation data with temporal dynamics

## 📋 Table of Contents

- [Installation](#-installation)
- [Quick Start](#-quick-start)
- [Usage](#-usage)
- [Configuration](#-configuration)
- [Architecture](#-architecture)
- [Training](#-training)
- [Results](#-results)
- [Contributing](#-contributing)
- [Citation](#-citation)

## 🛠 Installation

### Prerequisites

- Python 3.8+
- CUDA 11.0+ (for GPU training)
- 8GB+ GPU memory (16GB+ recommended for large datasets)

### Dependencies

Install required packages:

```bash
pip install -r requirements.txt
```

**Core Dependencies:**
- PyTorch with CUDA support
- NumPy, Pandas, Matplotlib
- OpenCV, scikit-image, scikit-learn
- TensorBoard for monitoring
- torchinfo, torchsummaryX for model analysis

## 🚀 Quick Start

### 1. Prepare Your Data

**For VAE Training:**
- Place simulation datasets as `input_data/dataset#X.pickle`
- Format: 3D arrays `[num_parameters, num_timesteps, num_nodes]`

**For Latent Conditioning:**
- **Images**: Place PNG/JPG files in `/images` directory
- **Parametric**: Use CSV files for MLP-based conditioning

### 2. Configure Training

Edit `input_data/condition.txt`:
```txt
# VAE Parameters
Training_epochs    10002
Batch_size        16
LearningR         0.0005
Latent_dim        8     # Hierarchical dimension
Latent_dim_end    32    # Main latent dimension

# Latent Conditioner
input_type        image  # image, csvs, image_vit
n_epoch          20000
latent_conditioner_lr    0.001
```

Set dataset preset in `preset.txt`:
```txt
data_No, init_beta_divisor, num_filter_enc, latent_conditioner_filter
1
0
1024 512 256 128
16 32 64
```

### 3. Start Training

**Single GPU:**
```bash
python SimulGen-VAE.py --preset=1 --plot=2 --lc_only=0 --size=small --load_all=1
```

**Multi-GPU (Recommended):**
```bash
torchrun --nproc_per_node=4 SimulGen-VAE.py --use_ddp --preset=1 --plot=2 --lc_only=0 --size=small --load_all=1
```

**Easy DDP Launcher:**
```bash
python launch_ddp.py --gpus=2 --preset=1 --plot=2 --size=small
```

## 📖 Usage

### Command Line Arguments

| Argument | Description | Options | Default |
|----------|-------------|---------|---------|
| `--preset` | Dataset configuration preset | 1-5 | 1 |
| `--plot` | Plotting mode | 0=off, 1=basic, 2=comprehensive | 2 |
| `--lc_only` | Train only latent conditioner | 0=full, 1=LC only | 0 |
| `--size` | Model size variant | small, big | small |
| `--load_all` | Data loading strategy | 0=lazy, 1=preload | 1 |
| `--use_ddp` | Enable distributed training | flag | false |

### Training Modes

**Full Training (VAE + Latent Conditioner):**
```bash
python SimulGen-VAE.py --preset=1 --lc_only=0
```

**Latent Conditioner Only:**
```bash
python SimulGen-VAE.py --preset=1 --lc_only=1
```

**Memory-Efficient Training:**
```bash
python SimulGen-VAE.py --preset=1 --load_all=0 --size=small
```

## ⚙️ Configuration

### System Architecture

The system uses a two-file configuration approach:

1. **`preset.txt`**: Model architecture presets
2. **`input_data/condition.txt`**: Training hyperparameters

### Key Parameters

**VAE Configuration:**
- `Latent_dim`: Hierarchical latent dimension (typically 8)
- `Latent_dim_end`: Main latent dimension (typically 32)
- `Loss_type`: 1=MSE, 2=MAE
- `Training_epochs`: Number of VAE training epochs

**Latent Conditioner Configuration:**
- `input_type`: "image", "csvs", or "image_vit"
- `latent_conditioner_dropout_rate`: Dropout strength (0.1-0.6)
- `use_spatial_attention`: Enable CNN attention mechanisms
- `param_data_type`: File extension for image inputs

### Anti-Overfitting Settings

SimulGenVAE implements comprehensive overfitting prevention:

```txt
latent_conditioner_dropout_rate     0.3      # Main dropout control
latent_conditioner_weight_decay     1e-5     # L2 regularization
use_spatial_attention              1         # Attention mechanisms
n_epoch                           20000      # Early stopping patience
```

See `OVERFITTING_PREVENTION_MEASURES.md` for detailed analysis.

## 🏗 Architecture

### Core Components

**1. VAE Network (`modules/VAE_network.py`)**
- Hierarchical encoder-decoder with skip connections
- Multi-scale latent representation
- Support for multiple loss functions (MSE, MAE, Huber, SmoothL1)

**2. Latent Conditioner (`modules/latent_conditioner.py`)**
- **MLP**: For parametric data conditioning
- **CNN**: Image-based conditioning with spatial attention
- **ViT**: Vision Transformer for complex image analysis

**3. Training System (`modules/train.py`)**
- Mixed precision training with GradScaler
- Advanced learning rate scheduling (warmup + cosine annealing)
- Distributed Data Parallel (DDP) support

### Latent Space Design

```
Input Data → Encoder → [Main Latent: 32D] + [Hierarchical: 8D] → Decoder → Output
                              ↑
                    Latent Conditioner
                    (Images/Parameters)
```

## 🎯 Training

### Training Pipeline

1. **Initialization**: Load datasets and configure models
2. **Warmup Phase**: Gradual learning rate increase (10 epochs)
3. **Main Training**: Cosine annealing with early stopping
4. **Validation**: Continuous monitoring with best model saving

### Monitoring

**TensorBoard Integration:**
```bash
tensorboard --logdir=output/
```

**Real-time Metrics:**
- Training/validation loss curves
- Overfitting ratio monitoring
- GPU memory usage tracking
- NaN detection and recovery

### Performance Optimization

**Memory Management:**
- Mixed precision: ~40% memory reduction
- Gradient checkpointing: Additional memory savings
- Lazy loading: Reduced RAM usage for large datasets

**Training Speed:**
- Multi-GPU scaling with DDP
- Optimized data loading with multiple workers
- CUDA kernel optimization for outline-preserving augmentations

## 📊 Results

### Supported Tasks

**1. Parametric Estimations**
- Multi-parametric regression from simulation data
- Physics-informed parameter inference

**2. Non-parametric Estimations**
- Image-to-simulation mapping
- CAD geometry conditioning

**3. Probabilistic Estimations**
- Uncertainty quantification in simulations
- Scattering analysis and sampling

### Performance Targets

- **Training Loss**: < 1e-2 (MSE)
- **Validation Loss**: < 5e-2 (MSE)
- **Convergence**: 50% improvement over baseline VAE
- **Memory Efficiency**: 40% reduction with mixed precision

## 🤝 Contributing

### Development Workflow

1. Fork the repository
2. Create a feature branch: `git checkout -b feature-name`
3. Configure parameters in `input_data/condition.txt`
4. Test with small model: `--size=small --load_all=0`
5. Scale to full training for validation
6. Submit pull request with performance metrics

### Code Style

- Follow PEP 8 conventions
- Document new parameters in configuration files
- Include performance benchmarks for major changes
- Test both single-GPU and multi-GPU configurations

## 📚 Citation

If you use SimulGenVAE in your research, please cite:

```bibtex
@software{simulgen_vae_2024,
  title={SimulGenVAE: High-Performance Variational Autoencoder for Simulation Data},
  author={Lee, SiHun},
  version={1.4.3},
  year={2024},
  email={kevin1007kr@gmail.com},
  url={https://www.linkedin.com/in/시훈-이-13009a172/?originalSubdomain=kr}
}
```

## 📞 Contact

**Author**: SiHun Lee, Ph.D.  
**Email**: kevin1007kr@gmail.com  
**LinkedIn**: [SiHun Lee](https://www.linkedin.com/in/시훈-이-13009a172/?originalSubdomain=kr)

## 📝 License

This project is licensed for academic use. Please contact the author for commercial licensing.

## 🔄 Version History

### v1.4.3 (Current)
- Streamlined argument system (`--lc_only` vs `--train_latent_conditioner_only`)
- Modern DDP with torchrun support
- Enhanced error handling and graceful fallbacks
- Updated documentation and user experience improvements

### v1.4.2
- Advanced learning rate scheduling (warmup + deep cosine annealing)
- Residual connections in LatentConditioner output heads
- Comprehensive data analysis during training initialization
- Enhanced loss monitoring with outlier detection

### v1.4.0
- Redesigned LatentConditioner with ResNet-style blocks
- SE (Squeeze-and-Excitation) attention mechanisms
- Shared backbone architecture eliminating network duplication
- Early stopping with separate y1/y2 loss tracking
- Fixed batch size issues with LayerNorm optimization

---

⭐ **Star this repository if SimulGenVAE helps your research!**