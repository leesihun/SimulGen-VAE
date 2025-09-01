# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

SimulGenVAE is a high-performance Physics-Aware Variational Autoencoder system designed for fast generation and inference of transient/static simulation data. It supports both single-GPU and multi-GPU distributed training with comprehensive data augmentation and validation.

## Key Training Commands

### Single GPU Training
```bash
python SimulGen-VAE.py --preset=1 --plot=2 --lc_only=0 --size=small
```

### Multi-GPU Distributed Training
```bash
torchrun --nproc_per_node=4 SimulGen-VAE.py --use_ddp --preset=1
```

### Latent Conditioner Only Training
```bash
python SimulGen-VAE.py --preset=1 --lc_only=1 --size=small
```

### DDP Launcher (Simplified)
```bash
python launch_ddp.py --gpus=2 --preset=1 --plot=2 --size=small
```

## Dependencies and Setup

Install dependencies with:
```bash
pip install -r requirements.txt
```

Key dependencies include PyTorch, NumPy, pandas, matplotlib, torchinfo, audiomentations, scikit-learn, librosa, OpenCV, TensorBoard, and torchvision.

## Architecture Overview

### Core Components

1. **Main Script**: `SimulGen-VAE.py` - Entry point with comprehensive command-line interface
2. **VAE Network**: `modules/VAE_network.py` - Core variational autoencoder with hierarchical latent spaces
3. **Encoder/Decoder**: `modules/encoder.py` and `modules/decoder.py` - Network architectures
4. **Latent Conditioner Models**: 
   - `modules/latent_conditioner_model_parametric.py` - MLP-based conditioning
   - `modules/latent_conditioner_model_cnn.py` - CNN/Vision Transformer conditioning

### Training Modes

- **Full VAE** (`lc_only=0`): Train complete VAE encoder/decoder + LatentConditioner
- **LC Only** (`lc_only=1`): Train only LatentConditioner using pre-trained VAE

### Model Sizes

- **small**: Memory-efficient variant for smaller systems
- **large**: Full-scale model for maximum performance

## Data Configuration

### Preset System
Configuration presets are managed through `preset.txt`. The file format:
- Line 1: `data_No, init_beta_divisior, num_filter_enc, latent_conditioner_filter`
- Line 2: Dataset number
- Line 3: Beta divisor
- Line 4: VAE encoder filters (space-separated)
- Line 5: Latent conditioner filters (space-separated)

### Input Data Structure
Primary configuration in `input_data/condition.txt`:
- Simulation data: 3D arrays [parameters, timesteps, nodes]
- Image data: PNG/JPG files in `/images` directory
- CSV data: Parametric data for MLP conditioning

### Supported Input Types
- **Simulation Data**: Pickled 3D arrays (dataset*.pickle format)
- **Images**: PNG/JPG for CNN/Vision Transformer conditioning
- **CSV**: Parametric data for MLP conditioning
- **PCA-processed**: Efficient MLP conditioning from image data

## Key Module Functions

### Data Processing
- `modules/data_preprocess.py` - Dataset reduction and scaling
- `modules/augmentation.py` - Data augmentation pipelines
- `modules/pca_preprocessor.py` - PCA dimensionality reduction

### Training Components
- `modules/train.py` - Main VAE training loop
- `modules/latent_conditioner.py` - Latent conditioner training
- `modules/enhanced_latent_conditioner_training.py` - Advanced training features
- `modules/latent_conditioner_e2e.py` - End-to-end training pipeline

### Utilities
- `modules/utils.py` - Core utilities, data loading, distributed training setup
- `modules/plotter.py` - Visualization tools
- `modules/reconstruction_evaluator.py` - Model evaluation metrics

## Command Line Arguments

- `--preset`: Dataset preset (1-5, reads from preset.txt)
- `--plot`: Visualization (0=interactive, 1=save, 2=off)
- `--lc_only`: Training mode (0=full VAE, 1=LatentConditioner only)
- `--size`: Model architecture (small/large)
- `--load_all`: Memory mode (0=lazy loading, 1=preload all)
- `--use_ddp`: Enable distributed data parallel training

## Development Workflow

1. Configure training parameters in `input_data/condition.txt`
2. Set model parameters in `preset.txt`
3. Prepare dataset files (dataset*.pickle format)
4. Run training with appropriate command line arguments
5. Monitor training via TensorBoard logs in output directories

## File Structure Notes

- `checkpoints/`: Model checkpoints during training
- `model_save/`: Final saved models
- `output/`: Training outputs and logs
- `images/`: Image data for conditioning (when using image inputs)
- `input_data/`: Configuration files and input specifications

## Git Integration

A convenience script `git-auto-push.sh` is available for automated git commits with timestamps.