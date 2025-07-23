import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math
import time
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

# Data reading functions moved to latent_conditioner.py to avoid duplication

# ============================================================================
# CNN-BASED LATENT CONDITIONERS (moved from latent_conditioner.py)
# ============================================================================

class LatentConditioner(nn.Module):
    """MLP-based latent conditioner for parametric data"""
    def __init__(self, latent_conditioner_filter, latent_dim_end, input_shape, latent_dim, size2, dropout_rate=0.3):
        super(LatentConditioner, self).__init__()
        self.latent_dim = latent_dim
        self.size2 = size2
        self.latent_conditioner_filter = latent_conditioner_filter
        self.latent_dim_end = latent_dim_end
        self.input_shape = input_shape
        self.num_latent_conditioner_filter = len(self.latent_conditioner_filter)
        self.dropout_rate = dropout_rate

        # Backbone feature extractor
        modules = []
        modules.append(nn.Linear(self.input_shape, self.latent_conditioner_filter[0]))
        for i in range(1, self.num_latent_conditioner_filter-1):
            modules.append(nn.Linear(self.latent_conditioner_filter[i-1], self.latent_conditioner_filter[i]))
            modules.append(nn.LeakyReLU(0.2))
            modules.append(nn.Dropout(0.1))
        self.latent_conditioner = nn.Sequential(*modules)

        # Simplified output heads
        final_feature_size = self.latent_conditioner_filter[-2]
        
        # ULTRA-EXTREME bottleneck with single output heads
        hidden_size = max(8, final_feature_size // 32)
        
        # Single prediction head for latent output
        self.latent_out = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(final_feature_size, hidden_size),
            nn.GELU(),
            nn.Dropout(0.15), 
            nn.Linear(hidden_size, self.latent_dim_end),
            nn.Tanh()
        )
        
        # Single prediction head for xs output
        self.xs_out = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(final_feature_size, hidden_size),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_size, self.latent_dim * self.size2),
            nn.Tanh()
        )

    def forward(self, x):
        features = self.latent_conditioner(x)
        
        # Direct prediction from single heads
        latent_out = self.latent_out(features)
        xs_out = self.xs_out(features)
        xs_out = xs_out.unflatten(1, (self.size2, self.latent_dim))

        return latent_out, xs_out


class ImprovedConvResBlock(nn.Module):
    """Improved convolutional residual block"""
    def __init__(self, in_channel, out_channel, stride=1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=True)
        self.gn1 = nn.GroupNorm(min(32, max(1, out_channel//4)), out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, bias=True)
        self.gn2 = nn.GroupNorm(min(32, max(1, out_channel//4)), out_channel)
        
        # Skip connection handling
        self.skip = nn.Sequential()
        if stride != 1 or in_channel != out_channel:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, bias=True),
                nn.GroupNorm(min(32, max(1, out_channel//4)), out_channel)
            )

    def forward(self, x):
        residual = self.skip(x)
        
        out = F.gelu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out += residual
        out = F.gelu(out)
        return out


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention"""
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.GELU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        if x is None or x.size(0) == 0:
            return x
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class ConvBlock(nn.Module):
    """Simple convolutional block with configurable dropout"""
    def __init__(self, in_channel, out_channel, dropout_rate=0.1):
        super().__init__()

        self.seq = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1),
            nn.GroupNorm(min(32, max(1, out_channel//4)), out_channel),
            nn.GELU(),
            nn.Dropout2d(dropout_rate * 0.5),  # Use 2D dropout for conv layers
            nn.AvgPool2d(2)
        )

    def forward(self, x):
        return self.seq(x)

import math

class LatentConditionerImg(nn.Module):
    """CNN-based latent conditioner for image data"""
    def __init__(self, latent_conditioner_filter, latent_dim_end, input_shape, latent_dim, size2, latent_conditioner_data_shape, dropout_rate=0.3):
        super(LatentConditionerImg, self).__init__()
        self.latent_dim = latent_dim
        self.size2 = size2
        self.latent_conditioner_filter = latent_conditioner_filter
        self.latent_dim_end = latent_dim_end
        self.input_shape = input_shape
        self.num_latent_conditioner_filter = len(self.latent_conditioner_filter)
        self.latent_conditioner_data_shape = latent_conditioner_data_shape
        self.dropout_rate = dropout_rate

        # Shared feature extractor backbone
        self.backbone = nn.ModuleList()
        
        # Initial conv
        self.backbone.append(nn.Sequential(
            nn.Conv2d(1, self.latent_conditioner_filter[0], kernel_size=7, stride=2, padding=3, bias=True),
            nn.GroupNorm(min(32, max(1, self.latent_conditioner_filter[0]//4)), self.latent_conditioner_filter[0]),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        ))
        
        # MINIMAL feature extraction - only 1 additional layer to prevent overfitting
        if self.num_latent_conditioner_filter > 1:
            # Single additional conv layer only
            block = ConvBlock(self.latent_conditioner_filter[0], self.latent_conditioner_filter[1], dropout_rate=self.dropout_rate)
            self.backbone.append(block)
        
        # Adaptive pooling and feature size calculation
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        final_feature_size = self.latent_conditioner_filter[-1] * 16  # 4*4
        
        # ULTRA-EXTREME bottleneck with single output heads
        hidden_size = max(4, final_feature_size // 64)  # Even smaller bottleneck
        
        # Single prediction head for latent output - using configurable dropout
        self.latent_out = nn.Sequential(
            nn.Dropout(self.dropout_rate),
            nn.Linear(final_feature_size, hidden_size),
            nn.GELU(),
            nn.Dropout(self.dropout_rate * 0.8),  # Slightly less dropout in second layer
            nn.Linear(hidden_size, self.latent_dim_end),
            nn.Tanh()
        )
        
        # Single prediction head for xs output - using configurable dropout
        self.xs_out = nn.Sequential(
            nn.Dropout(self.dropout_rate),
            nn.Linear(final_feature_size, hidden_size),
            nn.GELU(),
            nn.Dropout(self.dropout_rate * 0.8),  # Slightly less dropout in second layer
            nn.Linear(hidden_size, self.latent_dim * self.size2),
            nn.Tanh()
        )

    def forward(self, x):
        # Reshape flattened input to 4D tensor for CNN processing
        im_size = int(math.sqrt(x.shape[-1]))
        x = x.reshape(-1, 1, im_size, im_size)
        
        # Shared feature extraction
        features = x
        for block in self.backbone:
            features = block(features)
        
        # Global feature pooling
        features = self.adaptive_pool(features)
        features = features.flatten(1)
        
        # Direct prediction from single heads
        latent_out = self.latent_out(features)
        xs_out = self.xs_out(features)
        xs_out = xs_out.unflatten(1, (self.size2, self.latent_dim))

        return latent_out, xs_out

# ============================================================================
# VIT-BASED LATENT CONDITIONERS (new implementation)
# ============================================================================


class PatchEmbedding(nn.Module):
    """Convert image to patches and embed them"""
    def __init__(self, img_size=128, patch_size=16, in_channels=1, embed_dim=64, dropout=0.3):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = in_channels * patch_size * patch_size
        
        # Ultra-simple patch projection to prevent overfitting
        self.projection = nn.Sequential(
            nn.Linear(self.patch_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
        # Learnable position embeddings
        self.position_embeddings = nn.Parameter(torch.randn(1, self.num_patches, embed_dim) * 0.02)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, C, H, W = x.shape
        
        # Convert to patches: (B, C, H, W) -> (B, num_patches, patch_dim)
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.contiguous().view(B, C, -1, self.patch_size, self.patch_size)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(B, -1, self.patch_dim)
        
        # Project patches to embedding dimension
        x = self.projection(x)
        
        # Add position embeddings
        x = x + self.position_embeddings
        x = self.dropout(x)
        
        return x


class MultiHeadSelfAttention(nn.Module):
    """Self-attention with extreme regularization"""
    def __init__(self, embed_dim=64, num_heads=4, dropout=0.4, attention_dropout=0.3):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0
        
        # Single linear layer for Q, K, V to reduce parameters
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.projection = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Learnable temperature for attention sharpening
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, x):
        B, N, C = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention with temperature
        attention_scores = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5) / self.temperature.clamp(min=0.1)
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.attention_dropout(attention_weights)
        
        # Apply attention to values
        out = (attention_weights @ v).transpose(1, 2).reshape(B, N, C)
        
        # Final projection
        out = self.projection(out)
        out = self.dropout(out)
        
        return out, attention_weights


class TransformerBlock(nn.Module):
    """Minimal transformer block with heavy regularization"""
    def __init__(self, embed_dim=64, num_heads=4, mlp_ratio=2, dropout=0.4):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Tiny MLP to prevent overfitting
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(dropout)
        )
        
        # Stochastic depth for regularization
        self.drop_path_prob = dropout * 0.5

    def forward(self, x):
        # Self-attention with residual connection
        normed_x = self.norm1(x)
        attn_out, attention_weights = self.attention(normed_x)
        
        # Stochastic depth
        if self.training and torch.rand(1) < self.drop_path_prob:
            x = x  # Skip attention
        else:
            x = x + attn_out
        
        # MLP with residual connection
        normed_x = self.norm2(x)
        mlp_out = self.mlp(normed_x)
        
        # Stochastic depth for MLP
        if self.training and torch.rand(1) < self.drop_path_prob:
            pass  # Skip MLP
        else:
            x = x + mlp_out
            
        return x, attention_weights


class TinyViTLatentConditioner(nn.Module):
    """Ultra-minimal ViT for latent conditioning with extreme anti-overfitting"""
    def __init__(self, latent_dim_end, latent_dim, size2, 
                 img_size=128, patch_size=16, embed_dim=64, num_layers=2, num_heads=4, 
                 mlp_ratio=2, dropout=0.5):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.latent_dim_end = latent_dim_end
        self.latent_dim = latent_dim
        self.size2 = size2
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, 1, embed_dim, dropout)
        
        # Minimal transformer layers
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Global average pooling instead of CLS token to reduce parameters
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # EXTREME bottleneck - even smaller than CNN version
        hidden_size = max(4, embed_dim // 8)  # Minimum 4 features
        
        # Single prediction head for latent output
        self.latent_out = nn.Sequential(
            nn.Dropout(0.6),
            nn.Linear(embed_dim, hidden_size),
            nn.GELU(), 
            nn.Dropout(0.5),
            nn.Linear(hidden_size, latent_dim_end),
            nn.Tanh()
        )
        
        # Single prediction head for xs output
        self.xs_out = nn.Sequential(
            nn.Dropout(0.6),
            nn.Linear(embed_dim, hidden_size),
            nn.GELU(),
            nn.Dropout(0.5), 
            nn.Linear(hidden_size, latent_dim * size2),
            nn.Tanh()
        )

    def forward(self, x):
        B = x.shape[0]
        
        # Reshape flattened input to image
        x = x.reshape(B, 1, self.img_size, self.img_size)
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)
        
        # Apply transformer blocks
        attention_maps = []
        for block in self.transformer_blocks:
            x, attn_weights = block(x)
            attention_maps.append(attn_weights)
        
        x = self.norm(x)
        
        # Global average pooling over patches
        x = x.transpose(1, 2)  # (B, embed_dim, num_patches)
        x = self.global_pool(x).squeeze(-1)  # (B, embed_dim)
        
        # Direct predictions from single heads
        latent_out = self.latent_out(x)
        xs_out = self.xs_out(x)
        xs_out = xs_out.unflatten(1, (self.size2, self.latent_dim))
        
        return latent_out, xs_out


# ViT is ONLY for image data - parametric data should use the original CNN latent conditioner
# or a simple MLP-based approach. This was a conceptual error.


def safe_cuda_initialization():
    """Safely check CUDA availability with error handling"""
    try:
        if torch.cuda.is_available():
            test_tensor = torch.zeros(1).cuda()
            del test_tensor
            print("✓ CUDA initialized successfully")
            return "cuda"
        else:
            print("CUDA not available, using CPU")
            return "cpu"
    except RuntimeError as e:
        print(f"⚠️ CUDA initialization error: {e}")
        print("Falling back to CPU")
        return "cpu"