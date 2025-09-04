"""Ultra-Simple CNN-based Latent Conditioner Model

A minimal CNN architecture designed specifically for small datasets (hundreds of samples).
This model prioritizes simplicity and strong regularization over architectural sophistication
to prevent overfitting on limited training data.

Key Features:
- Only 3 convolutional layers with basic Conv→ReLU→Pool progression
- No ResNet blocks, attention mechanisms, or skip connections  
- High dropout (0.5) for strong regularization
- Minimal parameter count (~100K vs 1M+ in complex models)
- Direct dual-head output for main and hierarchical latent predictions

Author: SiHun Lee, Ph.D.
Email: kevin1007kr@gmail.com
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SimpleLatentConditionerImg(nn.Module):
    """Ultra-simple CNN-based latent conditioner for small datasets.
    
    Designed for datasets with hundreds (not thousands) of training samples.
    Uses aggressive regularization and minimal architecture complexity to
    prevent overfitting while maintaining dual latent prediction capability.
    
    Args:
        latent_conditioner_filter (list): Channel progression [32, 64, 128] (simplified)
        latent_dim_end (int): Main latent space dimension (typically 32)
        input_shape (tuple): Input image shape (C, H, W) - typically (1, 256, 256)
        latent_dim (int): Hierarchical latent dimension (typically 8)
        size2 (int): Hierarchical latent multiplier (typically 3)
        latent_conditioner_data_shape (tuple): Expected data shape (unused in simple model)
        dropout_rate (float): Dropout probability for regularization (default 0.5)
        use_attention (bool): Ignored in simple model (no attention)
        return_dict (bool): Return predictions as dictionary
    
    Architecture:
        Input (256x256) → 3 Conv layers → Global Pool → 2 Dense → Dual Heads
        Total parameters: ~100K (vs 1M+ in complex models)
    """
    
    def __init__(self, latent_conditioner_filter, latent_dim_end, input_shape, latent_dim, size2, 
                 latent_conditioner_data_shape, dropout_rate=0.5, use_attention=False, return_dict=False):
        super().__init__()
        
        # Store configuration
        self.latent_dim = latent_dim
        self.size2 = size2
        self.latent_dim_end = latent_dim_end
        self.dropout_rate = dropout_rate
        self.return_dict = return_dict
        
        # Input processing - handle both grayscale and RGB
        if isinstance(input_shape, (list, tuple)) and len(input_shape) >= 2:
            input_channels = 1 if len(input_shape) == 2 or input_shape[0] == 1 else input_shape[0]
        else:
            input_channels = 1  # Default to grayscale
        
        # Ultra-simple CNN backbone - only 3 layers with dropout
        # Layer 1: 1→16 channels (reduced capacity), large receptive field
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_channels, 16, kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(16),
            nn.GELU(),
            nn.Dropout2d(0.3),  # Add conv dropout for regularization
            nn.MaxPool2d(kernel_size=2, stride=2)  # 256→128
        )
        
        # Layer 2: 16→32 channels (reduced capacity)
        self.conv2 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=5, stride=1, padding=2, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.Dropout2d(0.3),  # Add conv dropout for regularization
            nn.MaxPool2d(kernel_size=2, stride=2)  # 128→64
        )
        
        # Layer 3: 32→64 channels (reduced capacity)
        self.conv3 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.Dropout2d(0.3),  # Add conv dropout for regularization
            nn.MaxPool2d(kernel_size=2, stride=2)  # 64→32
        )
        
        # Global pooling to handle any input size
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Simple feature processing with very strong regularization
        self.feature_processor = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(64, 128),  # Reduced from 128→256 to 64→128
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),  # Reduced from 256→128 to 128→64
            nn.GELU(),
            nn.Dropout(dropout_rate * 0.8)  # Higher dropout before final outputs
        )
        
        # Dual output heads - simple single-layer projections (reduced input size)
        self.latent_main_head = nn.Linear(64, latent_dim_end)  # Changed from 128 to 64
        self.xs_head = nn.Linear(64, latent_dim * size2)  # Changed from 128 to 64
        
        # Initialize weights properly for GELU
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize weights using Xavier initialization for GELU."""
        if isinstance(module, nn.Conv2d):
            nn.init.xavier_uniform_(module.weight, gain=1.0)
        elif isinstance(module, nn.Linear):
            # Xavier initialization for all linear layers with GELU
            nn.init.xavier_uniform_(module.weight, gain=1.0)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, nn.BatchNorm2d):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        """Forward pass through the simple CNN architecture.
        
        Args:
            x (torch.Tensor): Input tensor, can be:
                - Flattened image: [batch, height*width] → reshaped to [batch, 1, height, width]
                - Image tensor: [batch, channels, height, width]
        
        Returns:
            tuple: (latent_main, xs) where:
                - latent_main: Main latent predictions [batch, latent_dim_end]
                - xs: Hierarchical latent predictions [batch, size2, latent_dim]
        """
        batch_size = x.shape[0]
        spatial_dim = int(math.sqrt(x.shape[-1]))
        x = x.reshape(batch_size, 1, spatial_dim, spatial_dim)
        
        # Simple CNN forward pass
        x = self.conv1(x)    # [batch, 32, 128, 128]
        x = self.conv2(x)    # [batch, 64, 64, 64]
        x = self.conv3(x)    # [batch, 128, 32, 32]
        
        # Global pooling and flattening
        x = self.global_pool(x).flatten(1)  # [batch, 128]
        
        # Feature processing with regularization
        features = self.feature_processor(x)  # [batch, 128]
        
        # Dual predictions - simple projections
        latent_main = self.latent_main_head(features)  # [batch, 32]
        xs_flat = self.xs_head(features)  # [batch, 24]
        
        # Reshape hierarchical latents to expected format
        xs = xs_flat.unflatten(1, (self.size2, self.latent_dim))  # [batch, 3, 8]
        
        if self.return_dict:
            return {
                'latent_main': latent_main,
                'xs': xs,
                'features': features.detach()  # For analysis
            }
        
        return latent_main, xs


def count_parameters(model):
    """Count total trainable parameters in the model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Example usage and parameter comparison
if __name__ == "__main__":
    # Test the simple model
    model = SimpleLatentConditionerImg(
        latent_conditioner_filter=[32, 64, 128],  # Simplified progression
        latent_dim_end=32,
        input_shape=(1, 256, 256),
        latent_dim=8,
        size2=3,
        latent_conditioner_data_shape=None,
        dropout_rate=0.5
    )
    
    # Test forward pass
    batch_size = 4
    test_input = torch.randn(batch_size, 256*256)  # Flattened input
    
    with torch.no_grad():
        latent_main, xs = model(test_input)
    
    print(f"Simple CNN Model Summary:")
    print(f"  Total parameters: {count_parameters(model):,}")
    print(f"  Input shape: {test_input.shape}")
    print(f"  Main latent output: {latent_main.shape}")
    print(f"  Hierarchical latent output: {xs.shape}")
    print(f"  Memory footprint: ~{count_parameters(model) * 4 / 1024 / 1024:.1f} MB")