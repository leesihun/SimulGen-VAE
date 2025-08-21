# SimulGenVAE

A high-performance Variational Autoencoder system designed for fast generation and inference of transient/static simulation data with Physics-Aware Neural Network (PANN) integration.

## Overview

SimulGenVAE is a comprehensive deep learning framework that combines Variational Autoencoders (VAE) with advanced latent conditioning mechanisms for simulation data processing. The system supports both single-GPU and multi-GPU distributed training with three different latent conditioning architectures.

### Key Features

- **Hierarchical Latent Space**: Two-level representation (main: 32D, hierarchical: 8D)
- **Multiple Conditioning Architectures**: MLP, CNN, and Vision Transformer based
- **Distributed Training**: Single-GPU and multi-GPU support with DDP
- **Advanced Anti-Overfitting**: Comprehensive prevention measures with early stopping
- **Mixed Precision Training**: Memory-efficient training with gradient checkpointing
- **Flexible Input Types**: Supports simulation data, images, and parametric data

## Architecture

### Core Components

- **VAE Network** (`modules/VAE_network.py`): Main Variational Autoencoder with hierarchical encoder-decoder
- **Latent Conditioner** (`modules/latent_conditioner.py`): Conditions latent space on external parameters
- **Training System** (`modules/train.py`): Advanced training pipeline with mixed precision
- **Data Processing** (`modules/data_preprocess.py`): Handles simulation dataset loading and preprocessing

### Latent Conditioning Models

1. **MLP-based** (`latent_conditioner_model_parametric.py`): For parametric data input
2. **CNN-based** (`latent_conditioner_model_cnn.py`): For image/outline detection with spatial attention
3. **Vision Transformer** (`latent_conditioner_model_vit.py`): For complex image analysis

## Installation

### Requirements

```bash
pip install -r requirements.txt
```

Core dependencies:
- PyTorch
- NumPy
- Pandas
- Matplotlib
- TensorBoard
- OpenCV
- scikit-learn
- torchvision

### Hardware Requirements

- **GPU**: CUDA-compatible GPU recommended (multi-GPU supported)
- **Memory**: Varies based on dataset size and model configuration
- **Storage**: Sufficient space for datasets, models, and outputs

## Quick Start

### Basic Training Commands

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

### Command Line Arguments

- `--preset`: Dataset configuration (1-5, defined in preset.txt)
- `--plot`: Plotting mode (0=interactive, 1=save, 2=off)
- `--lc_only`: Train mode (0=full VAE training, 1=latent conditioner only)
- `--size`: Model size (small/big)
- `--load_all`: Memory mode (0=lazy loading, 1=preload all data)
- `--use_ddp`: Enable distributed training

## Configuration

### Primary Configuration Files

#### 1. `preset.txt` - Dataset Presets
```
data_No, init_beta_divisior, num_filter_enc, latent_conditioner_filter
1
0
1024 512 256 128
32 64 128 256
```

#### 2. `input_data/condition.txt` - Training Parameters

Key sections:
- **Common Parameters**: Data dimensions (Dim1: 484, Dim2: 200, Dim3: 95008)
- **VAE Parameters**: Epochs, batch size, learning rate, latent dimensions
- **Latent Conditioner**: Architecture type, dropout rates, attention settings
- **End-to-End Training**: Direct reconstruction optimization settings

### Input Types Configuration

Set `input_type` in `condition.txt`:
- `image`: PNG/JPG files for CNN/ViT conditioning
- `csvs`: CSV files for MLP conditioning
- `image_vit`: Images processed with Vision Transformer

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

### Dataset Formats

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

## Advanced Features

### Anti-Overfitting Strategy
- Progressive dropout rates (0.1-0.6)
- Outline-preserving data augmentation for images
- Early stopping with validation monitoring
- Weight decay and gradient clipping
- Label smoothing and mixup augmentation

### Memory Management
- `--load_all=1`: Preload all data (faster but memory intensive)
- `--load_all=0`: Lazy loading (memory efficient but slower)
- Mixed precision reduces memory usage by ~40%
- Gradient checkpointing for extreme memory constraints

### Distributed Training
- Uses modern `torchrun` instead of deprecated `torch.distributed.launch`
- Automatic local_rank detection
- NCCL backend for multi-GPU communication
- Graceful fallback to single-GPU if DDP fails

## Monitoring and Debugging

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

## Development Workflow

### Typical Development Pattern
1. Configure parameters in `input_data/condition.txt`
2. Set dataset preset in `preset.txt`
3. Start with small model and single GPU for testing
4. Scale to multi-GPU for full training
5. Monitor via TensorBoard logs in output directory

### Performance Optimization
- Use mixed precision training for memory efficiency
- Enable gradient checkpointing for large models
- Optimize batch sizes based on GPU memory
- Use distributed training for faster convergence

## Troubleshooting

### Common Issues

#### Memory Problems
- Reduce batch size in `condition.txt`
- Use `--load_all=0` for lazy loading
- Enable gradient checkpointing
- Use smaller model size (`--size=small`)

#### Training Instability
- Check for NaN values in loss
- Reduce learning rate
- Increase dropout rates
- Enable early stopping

#### Multi-GPU Issues
- Ensure NCCL backend is properly configured
- Check GPU visibility with `nvidia-smi`
- Use `torchrun` instead of deprecated launchers

## Contributing

### Code Structure
- Follow existing naming conventions
- Add comprehensive docstrings
- Implement proper error handling
- Include validation and testing

### Performance Considerations
- Profile memory usage for new features
- Test both single and multi-GPU configurations
- Validate numerical stability
- Document computational complexity

## Citation

If you use SimulGenVAE in your research, please cite:

```bibtex
@software{simulgen_vae,
  author = {SiHun Lee},
  title = {SimulGenVAE: High-Performance Physics-Aware Variational Autoencoder},
  version = {2.0.0},
  email = {kevin1007kr@gmail.com},
  year = {2024}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Contact

- **Author**: SiHun Lee, Ph.D.
- **Email**: kevin1007kr@gmail.com
- **Version**: 2.0.0 (Refactored & Documented)

## Acknowledgments

- PyTorch team for the deep learning framework
- Contributors to the physics-aware neural network community
- Open source libraries that make this project possible