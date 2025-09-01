"""State-of-the-Art CNN-based Latent Conditioner Model

Implements a high-performance CNN architecture for image-to-latent regression in SimulGenVAE.
This model processes 256×256 images to predict latent space representations with modern
architectural components including ResNet blocks, attention mechanisms, and spectral normalization.

Key Features:
- ResNet-style backbone with GroupNorm and Squeeze-and-Excitation attention
- Adaptive average pooling for flexible input sizes
- Spectral normalization for training stability
- Progressive channel scaling with modern activations (SiLU)
- Dual output heads for main and hierarchical latent predictions
- Optional spatial attention mechanisms
- Comprehensive dropout and regularization

Author: SiHun Lee, Ph.D.
Email: kevin1007kr@gmail.com

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.utils import spectral_norm
from modules.common import add_sn
import numpy as np


class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation attention module.
    
    Implements channel-wise attention mechanism that adaptively recalibrates
    channel-wise feature responses by explicitly modelling interdependencies
    between channels.
    
    Reference:
        Squeeze-and-Excitation Networks (Hu et al., 2018)
    """
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc1 = nn.Linear(channels, channels // reduction)
        self.fc2 = nn.Linear(channels // reduction, channels)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.global_pool(x).view(b, c)
        y = F.relu(self.fc1(y))
        y = self.sigmoid(self.fc2(y)).view(b, c, 1, 1)
        return x * y


class SpatialAttention(nn.Module):
    """Spatial attention module for spatial feature enhancement.
    
    Computes attention weights across spatial dimensions to focus on
    important regions in the feature maps.
    """
    def __init__(self, kernel_size=7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        attention = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(attention))
        return x * attention


class ResNetBlock(nn.Module):
    """Enhanced ResNet block with deeper architecture and modern components.
    
    Features:
    - 3-convolution layer architecture for richer representations
    - GroupNormalization for better performance across batch sizes
    - SiLU activation for improved gradient flow
    - Squeeze-and-Excitation attention
    - Optional spatial attention
    - Spectral normalization for stability
    - Gradient scaling for deeper networks
    """
    def __init__(self, in_channels, out_channels, stride=1, use_attention=True, use_spatial_attention=False):
        super().__init__()
        
        # Three-layer convolution path for richer feature extraction
        mid_channels = out_channels // 2  # Bottleneck design
        
        # First conv: channel reduction + spatial downsampling
        self.conv1 = add_sn(nn.Conv2d(in_channels, mid_channels, 1, 1, 0, bias=False))
        self.gn1 = nn.GroupNorm(self._get_num_groups(mid_channels), mid_channels)
        
        # Second conv: spatial processing with stride
        self.conv2 = add_sn(nn.Conv2d(mid_channels, mid_channels, 3, stride, 1, bias=False))
        self.gn2 = nn.GroupNorm(self._get_num_groups(mid_channels), mid_channels)
        
        # Third conv: channel expansion
        self.conv3 = add_sn(nn.Conv2d(mid_channels, out_channels, 1, 1, 0, bias=False))
        self.gn3 = nn.GroupNorm(self._get_num_groups(out_channels), out_channels)
        
        # Skip connection
        self.skip = nn.Identity()
        if stride != 1 or in_channels != out_channels:
            self.skip = nn.Sequential(
                add_sn(nn.Conv2d(in_channels, out_channels, 1, stride, bias=False)),
                nn.GroupNorm(self._get_num_groups(out_channels), out_channels)
            )
        
        # Attention mechanisms
        self.use_attention = use_attention
        self.use_spatial_attention = use_spatial_attention
        
        if use_attention:
            self.se = SqueezeExcitation(out_channels)
        
        if use_spatial_attention:
            self.spatial_attn = SpatialAttention()
    
    def _get_num_groups(self, channels):
        """Calculate appropriate number of groups that divides evenly."""
        # Common divisors for GroupNorm
        possible_groups = [1, 2, 4, 8, 16, 32]
        for groups in reversed(possible_groups):
            if channels % groups == 0 and groups <= channels:
                return groups
        return 1  # Fallback to LayerNorm equivalent
    
    def forward(self, x):
        identity = x
        
        # Three-layer convolution path
        out = F.silu(self.gn1(self.conv1(x)))
        out = F.silu(self.gn2(self.conv2(out)))
        out = self.gn3(self.conv3(out))
        
        # Apply attention mechanisms
        if self.use_attention:
            out = self.se(out)
        
        if self.use_spatial_attention:
            out = self.spatial_attn(out)
        
        # Skip connection with gradient scaling for deeper networks
        out = out + 0.1 * self.skip(identity)
        out = F.silu(out)
        
        return out


class LatentConditionerImg(nn.Module):
    """State-of-the-art CNN-based latent conditioner for image inputs.
    
    Processes 256×256 images to predict latent space representations using a modern
    ResNet-style architecture with attention mechanisms and advanced regularization.
    
    Args:
        latent_conditioner_filter (list): Channel progression for CNN layers
        latent_dim_end (int): Main latent space dimension (typically 32)
        input_shape (tuple): Input image shape (C, H, W)
        latent_dim (int): Hierarchical latent dimension (typically 8)
        size2 (int): Hierarchical latent multiplier
        latent_conditioner_data_shape (tuple): Expected data shape
        dropout_rate (float): Dropout probability for regularization
        use_attention (bool): Enable squeeze-excitation attention
        return_dict (bool): Return predictions as dictionary
    
    Architecture:
        - Initial convolution with large receptive field
        - Progressive ResNet blocks with attention mechanisms
        - Adaptive global pooling for flexible input sizes
        - Dual prediction heads with proper regularization
        - Spectral normalization throughout for training stability
    """
    
    def __init__(self, latent_conditioner_filter, latent_dim_end, input_shape, latent_dim, size2, 
                 latent_conditioner_data_shape, dropout_rate=0.3, use_attention=True, return_dict=False):
        super().__init__()
        
        # Store configuration
        self.latent_dim = latent_dim
        self.size2 = size2
        self.latent_dim_end = latent_dim_end
        self.dropout_rate = dropout_rate
        self.use_attention = use_attention
        self.return_dict = return_dict
        
        # Input processing - handle both grayscale and RGB
        if isinstance(input_shape, (list, tuple)) and len(input_shape) >= 2:
            input_channels = 1 if len(input_shape) == 2 or input_shape[0] == 1 else input_shape[0]
        else:
            input_channels = 1  # Default to grayscale
        
        # Initial convolution with large receptive field
        self.initial_conv = nn.Sequential(
            add_sn(nn.Conv2d(input_channels, latent_conditioner_filter[0], 7, 2, 3, bias=False)),
            nn.GroupNorm(self._get_num_groups(latent_conditioner_filter[0]), latent_conditioner_filter[0]),
            nn.SiLU(),
            nn.MaxPool2d(3, 2, 1)
        )
        
        # ResNet backbone layers
        self.layers = nn.ModuleList()
        in_channels = latent_conditioner_filter[0]
        
        for i, out_channels in enumerate(latent_conditioner_filter[1:]):
            # Early-heavy downsampling: only at layers 1 and 3
            stride = 2 if i in [1, 3] else 1
            
            # Enable spatial attention for more layers and SE for all layers
            use_spatial_attn = use_attention and i >= 1 and i <= 6  # More layers with spatial attention
            use_se_attention = use_attention  # All layers get SE attention
            
            layer = ResNetBlock(
                in_channels, out_channels, 
                stride=stride,
                use_attention=use_se_attention,
                use_spatial_attention=use_spatial_attn
            )
            self.layers.append(layer)
            in_channels = out_channels
        
        # Global adaptive pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        
        # Enhanced feature processing with 4x capacity
        final_channels = latent_conditioner_filter[-1]
        hidden_dim = final_channels * 4  # 4x capacity increase
        intermediate_dim = hidden_dim * 2  # Even larger intermediate processing
        
        self.feature_processor = nn.Sequential(
            nn.Dropout(dropout_rate * 0.3),
            # First processing layer
            add_sn(nn.Linear(final_channels, hidden_dim)),
            nn.GroupNorm(self._get_num_groups(hidden_dim), hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate * 0.4),
            
            # Second processing layer with expansion
            add_sn(nn.Linear(hidden_dim, intermediate_dim)),
            nn.GroupNorm(self._get_num_groups(intermediate_dim), intermediate_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate * 0.5),
            
            # Third processing layer with compression back to hidden_dim
            add_sn(nn.Linear(intermediate_dim, hidden_dim)),
            nn.GroupNorm(self._get_num_groups(hidden_dim), hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout_rate * 0.4)
        )
        
        # Deeper main latent prediction head (5 layers)
        self.latent_main_head = nn.Sequential(
            # Layer 1
            add_sn(nn.Linear(hidden_dim, hidden_dim // 2)),
            nn.GroupNorm(self._get_num_groups(hidden_dim // 2), hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout_rate * 0.3),
            
            # Layer 2
            add_sn(nn.Linear(hidden_dim // 2, hidden_dim // 3)),
            nn.GroupNorm(self._get_num_groups(hidden_dim // 3), hidden_dim // 3),
            nn.SiLU(),
            nn.Dropout(dropout_rate * 0.3),
            
            # Layer 3
            add_sn(nn.Linear(hidden_dim // 3, hidden_dim // 4)),
            nn.GroupNorm(self._get_num_groups(hidden_dim // 4), hidden_dim // 4),
            nn.SiLU(),
            nn.Dropout(dropout_rate * 0.2),
            
            # Layer 4
            add_sn(nn.Linear(hidden_dim // 4, hidden_dim // 6)),
            nn.GroupNorm(self._get_num_groups(hidden_dim // 6), hidden_dim // 6),
            nn.SiLU(),
            nn.Dropout(dropout_rate * 0.1),
            
            # Layer 5 - Output
            nn.Linear(hidden_dim // 6, latent_dim_end)
        )
        
        # Deeper hierarchical latent prediction head (5 layers)
        self.xs_head = nn.Sequential(
            # Layer 1
            add_sn(nn.Linear(hidden_dim, hidden_dim // 2)),
            nn.GroupNorm(self._get_num_groups(hidden_dim // 2), hidden_dim // 2),
            nn.SiLU(),
            nn.Dropout(dropout_rate * 0.3),
            
            # Layer 2
            add_sn(nn.Linear(hidden_dim // 2, hidden_dim // 3)),
            nn.GroupNorm(self._get_num_groups(hidden_dim // 3), hidden_dim // 3),
            nn.SiLU(),
            nn.Dropout(dropout_rate * 0.3),
            
            # Layer 3
            add_sn(nn.Linear(hidden_dim // 3, hidden_dim // 4)),
            nn.GroupNorm(self._get_num_groups(hidden_dim // 4), hidden_dim // 4),
            nn.SiLU(),
            nn.Dropout(dropout_rate * 0.2),
            
            # Layer 4
            add_sn(nn.Linear(hidden_dim // 4, hidden_dim // 6)),
            nn.GroupNorm(self._get_num_groups(hidden_dim // 6), hidden_dim // 6),
            nn.SiLU(),
            nn.Dropout(dropout_rate * 0.1),
            
            # Layer 5 - Output
            nn.Linear(hidden_dim // 6, latent_dim * size2)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _get_num_groups(self, channels):
        """Calculate appropriate number of groups that divides evenly."""
        # Common divisors for GroupNorm
        possible_groups = [1, 2, 4, 8, 16, 32]
        for groups in reversed(possible_groups):
            if channels % groups == 0 and groups <= channels:
                return groups
        return 1  # Fallback to LayerNorm equivalent
    
    def _init_weights(self, module):
        """Initialize weights using modern best practices."""
        if isinstance(module, nn.Conv2d):
            # Kaiming initialization for SiLU/ReLU-like activations
            nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
        elif isinstance(module, nn.Linear):
            # Xavier initialization for final layers, Kaiming for intermediate
            if module.out_features in [self.latent_dim_end, self.latent_dim * self.size2]:
                nn.init.xavier_normal_(module.weight, gain=1.0)  # Normal gain for output layers
            else:
                nn.init.kaiming_normal_(module.weight, mode='fan_out')
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
        elif isinstance(module, (nn.GroupNorm, nn.BatchNorm2d)):
            nn.init.constant_(module.weight, 1)
            nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        """Forward pass through the CNN architecture.
        
        Args:
            x (torch.Tensor): Input tensor, can be:
                - Flattened image: [batch, height*width] -> reshaped to [batch, 1, height, width]
                - Image tensor: [batch, channels, height, width]
        
        Returns:
            tuple: (latent_main, xs) where:
                - latent_main: Main latent predictions [batch, latent_dim_end]
                - xs: Hierarchical latent predictions [batch, size2, latent_dim]
        """
        # Handle input reshaping
        # Flattened input -> reshape to square image
        batch_size = x.shape[0]
        spatial_dim = int(math.sqrt(x.shape[-1]))
        x = x.reshape(batch_size, 1, spatial_dim, spatial_dim)
        
        # Ensure input is in expected range [0, 1] or [-1, 1]
        if x.min() < -0.1:  # Likely in [-1, 1] range
            x = (x + 1) / 2  # Convert to [0, 1]
        
        # Initial convolution
        x = self.initial_conv(x)
        
        # ResNet backbone
        for layer in self.layers:
            x = layer(x)
        
        # Global pooling and feature processing
        x = self.global_pool(x).flatten(1)
        features = self.feature_processor(x)
        
        # Prediction heads
        latent_main = self.latent_main_head(features)
        xs = self.xs_head(features)
        
        # Reshape hierarchical latents
        xs = xs.unflatten(1, (self.size2, self.latent_dim))
        
        if self.return_dict:
            return {
                'latent_main': latent_main,
                'xs': xs,
                'features': features.detach()  # For analysis
            }
        
        return latent_main, xs