# PCA Preprocessor Usage Guide

This guide explains how to use the PCAPreprocessor with saved pickle files for image preprocessing in the SimulGenVAE system.

## Overview

The PCAPreprocessor reduces image dimensionality while preserving spatial structure. It supports both full-image PCA and patch-based PCA, with automatic model saving/loading via pickle files.

## Basic Usage

### 1. Loading a Pre-trained PCA Model

```python
from modules.pca_preprocessor import PCAPreprocessor

# Initialize preprocessor with same parameters as when fitted
pca_preprocessor = PCAPreprocessor(
    n_components=1024,        # Must match original training
    patch_size=None,          # Must match original (None for full image PCA)
    save_dir='model_save'     # Directory where .pkl file is saved (default)
)

# Load the fitted PCA model
pca_preprocessor.load()  # Auto-loads from model_save/pca_full_comp1024.pkl
```

### 2. Transform New Images Using Saved PCA

```python
import numpy as np
from PIL import Image

# Prepare your image data
image_paths = ['image1.png', 'image2.png', ...]  # Your image files
raw_images = np.zeros((len(image_paths), 256, 256))  # Assuming 256x256 images

# Load and preprocess images
for i, img_path in enumerate(image_paths):
    img = Image.open(img_path).convert('L')  # Convert to grayscale
    resized_img = img.resize((256, 256))
    raw_images[i] = np.array(resized_img) / 255.0  # Normalize to [0,1]

# Transform using loaded PCA
pca_features = pca_preprocessor.transform(raw_images)  # Returns torch.Tensor
print(f"PCA output shape: {pca_features.shape}")
```

## PCA Model Types and File Naming

### Full Image PCA
- **File name**: `pca_full_comp{n_components}.pkl`
- **Example**: `pca_full_comp1024.pkl`
- **Use case**: Global image features

### Patch-based PCA
- **File name**: `pca_patch{patch_size}_comp{n_components}.pkl`
- **Example**: `pca_patch32_comp512.pkl`
- **Use case**: Preserves spatial structure better

## Integration with Existing Training Pipeline

### Option 1: Load Existing PCA (Recommended)
```python
def load_or_fit_pca(training_images, n_components=1024, patch_size=None):
    """Load existing PCA or fit new one if not found"""
    pca_preprocessor = PCAPreprocessor(
        n_components=n_components, 
        patch_size=patch_size,
        save_dir='model_save'
    )
    
    try:
        pca_preprocessor.load()
        print("✓ Loaded existing PCA model")
    except FileNotFoundError:
        print("No existing PCA found, fitting new model...")
        pca_preprocessor.fit(training_images)
        print("✓ New PCA model fitted and saved")
    
    return pca_preprocessor
```

### Option 2: Force Reload Existing PCA
```python
# For inference/testing with pre-trained models
pca_preprocessor = PCAPreprocessor(n_components=1024, save_dir='model_save')
pca_preprocessor.load()  # Will raise error if file doesn't exist

# Transform your test images
test_features = pca_preprocessor.transform(test_images)
```

## Key Methods and Properties

### Core Methods
- `load()`: Load saved PCA model from pickle file
- `save()`: Save fitted PCA model to pickle file
- `fit(images)`: Fit PCA on training images and auto-save
- `transform(images)`: Transform images using fitted PCA (returns torch.Tensor)

### Utility Methods
- `get_output_shape()`: Get transformed image dimensions
- `get_output_channels()`: Get number of channels for CNN input
- `_get_pca_filename()`: Get automatic filename for current configuration

## Configuration Parameters

### Constructor Parameters
```python
PCAPreprocessor(
    n_components=1024,      # Number of PCA components to keep
    patch_size=None,        # None for full-image, int for patch-based
    save_dir='model_save'   # Directory for pickle files
)
```

### Pickle File Contents
Each saved PCA model contains:
- `pca`: The fitted sklearn PCA object
- `n_components`: Number of components used
- `patch_size`: Patch size (None for full-image)
- `original_shape`: Original image dimensions
- `output_shape`: Output dimensions after PCA
- `is_fitted`: Training status flag

## Example Workflows

### Workflow 1: First-time Training
```python
# Fit and save new PCA model
pca_preprocessor = PCAPreprocessor(n_components=512, save_dir='model_save')
pca_preprocessor.fit(training_images)  # Auto-saves to model_save/pca_full_comp512.pkl

# Use for training
train_features = pca_preprocessor.transform(training_images)
```

### Workflow 2: Inference with Pre-trained Model
```python
# Load existing model
pca_preprocessor = PCAPreprocessor(n_components=512, save_dir='model_save')
pca_preprocessor.load()  # Loads model_save/pca_full_comp512.pkl

# Transform new data
inference_features = pca_preprocessor.transform(new_images)
```

### Workflow 3: Patch-based Processing
```python
# Use patch-based PCA for spatial preservation
pca_preprocessor = PCAPreprocessor(
    n_components=256, 
    patch_size=32,
    save_dir='model_save'
)

# This will save as: model_save/pca_patch32_comp256.pkl
pca_preprocessor.fit(training_images)
patch_features = pca_preprocessor.transform(test_images)
```

## Error Handling

### Common Issues and Solutions

1. **FileNotFoundError**: PCA model file doesn't exist
   ```python
   try:
       pca_preprocessor.load()
   except FileNotFoundError:
       print("PCA model not found. Train model first or check file path.")
   ```

2. **ValueError**: PCA not fitted
   ```python
   if not pca_preprocessor.is_fitted:
       print("PCA model needs to be loaded or fitted first")
   ```

3. **Shape Mismatch**: Wrong parameters for existing model
   ```python
   # Ensure parameters match the saved model
   # Check actual file: model_save/pca_full_comp1024.pkl
   pca_preprocessor = PCAPreprocessor(n_components=1024)  # Must match
   ```

## Integration with SimulGenVAE

The PCA preprocessor integrates with the latent conditioner system:

- **Default save directory**: `model_save/` (matches other model checkpoints)
- **Auto-loading**: System can automatically load existing PCA models
- **Training pipeline**: PCA fitting happens before VAE training
- **Output format**: Returns torch.Tensor ready for CNN input

For more details, see `modules/pca_preprocessor.py` and `modules/latent_conditioner.py`.