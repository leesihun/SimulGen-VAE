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


class SpatialAttention(nn.Module):
    """Lightweight spatial attention module"""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Global average and max pooling across channels
        avg_out = torch.mean(x, dim=1, keepdim=True)  # (B,1,H,W)
        max_out, _ = torch.max(x, dim=1, keepdim=True)  # (B,1,H,W)
        
        # Combine both pooling results
        combined = torch.cat([avg_out, max_out], dim=1)  # (B,2,H,W)
        
        # Generate attention map
        attention = self.sigmoid(self.conv(combined))  # (B,1,H,W)
        
        return x * attention


class ConvBlock(nn.Module):
    """Enhanced convolutional block with spatial attention and configurable dropout"""
    def __init__(self, in_channel, out_channel, dropout_rate=0.1, use_attention=True):
        super().__init__()
        
        self.conv = nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(min(32, max(1, out_channel//4)), out_channel)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout2d(dropout_rate * 0.5)
        self.pool = nn.AvgPool2d(2)
        
        # Optional spatial attention
        self.use_attention = use_attention
        if use_attention:
            self.spatial_attention = SpatialAttention()

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        
        # Apply spatial attention before dropout/pooling
        if self.use_attention:
            x = self.spatial_attention(x)
            
        x = self.dropout(x)
        x = self.pool(x)
        return x

import math

class LatentConditionerImg(nn.Module):
    """CNN-based latent conditioner for image data with spatial attention"""
    def __init__(self, latent_conditioner_filter, latent_dim_end, input_shape, latent_dim, size2, latent_conditioner_data_shape, dropout_rate=0.3, use_attention=True):
        super(LatentConditionerImg, self).__init__()
        self.latent_dim = latent_dim
        self.size2 = size2
        self.latent_conditioner_filter = latent_conditioner_filter
        self.latent_dim_end = latent_dim_end
        self.input_shape = input_shape
        self.num_latent_conditioner_filter = len(self.latent_conditioner_filter)
        self.latent_conditioner_data_shape = latent_conditioner_data_shape
        self.dropout_rate = dropout_rate
        self.use_attention = use_attention

        # Edge detection preprocessing - helps with outline detection
        self.edge_enhance = nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=1, bias=False)
        # Initialize with edge detection kernels (Sobel, Laplacian, etc.)
        with torch.no_grad():
            # Sobel X
            self.edge_enhance.weight[0, 0] = torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=torch.float32)
            # Sobel Y  
            self.edge_enhance.weight[1, 0] = torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=torch.float32)
            # Laplacian
            self.edge_enhance.weight[2, 0] = torch.tensor([[0, -1, 0], [-1, 4, -1], [0, -1, 0]], dtype=torch.float32)
            # Identity (preserve original)
            self.edge_enhance.weight[3, 0] = torch.tensor([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=torch.float32)
        
        # Shared feature extractor backbone
        self.backbone = nn.ModuleList()
        
        # Edge-aware initial processing for 256x256 outline detection
        self.backbone.append(nn.Sequential(
            # Process both original + edge-enhanced features
            nn.Conv2d(5, self.latent_conditioner_filter[0], kernel_size=7, stride=1, padding=3, bias=True),  # 1+4=5 input channels
            nn.GroupNorm(min(32, max(1, self.latent_conditioner_filter[0]//4)), self.latent_conditioner_filter[0]),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2)  # 256->128
        ))
        
        # Multi-scale feature extraction for 256x256 outline detection
        if self.num_latent_conditioner_filter > 1:
            # Scale 1: 128->64 with spatial attention for edge preservation
            block1 = ConvBlock(self.latent_conditioner_filter[0], self.latent_conditioner_filter[1], 
                             dropout_rate=self.dropout_rate, use_attention=self.use_attention)
            self.backbone.append(block1)
            
            # Scale 2: 64->32 with spatial attention
            if self.num_latent_conditioner_filter > 2:
                block2 = ConvBlock(self.latent_conditioner_filter[1], self.latent_conditioner_filter[2], 
                                 dropout_rate=self.dropout_rate, use_attention=self.use_attention)
                self.backbone.append(block2)
                
            # Scale 3: 32->16 with spatial attention for deeper edge understanding
            if self.num_latent_conditioner_filter > 3:
                block3 = ConvBlock(self.latent_conditioner_filter[2], self.latent_conditioner_filter[3], 
                                 dropout_rate=self.dropout_rate, use_attention=self.use_attention)
                self.backbone.append(block3)
        
        # Multi-scale pooling - richer features for outline analysis
        self.adaptive_pool = nn.AdaptiveAvgPool2d((8, 8))  # Larger than 4x4 for 256x256
        self.max_pool = nn.AdaptiveMaxPool2d((8, 8))
        
        # Calculate final feature size accounting for avg+max pooling
        final_filter_idx = min(len(self.latent_conditioner_filter)-1, 3)
        final_feature_size = self.latent_conditioner_filter[final_filter_idx] * 64 * 2  # 8*8*2 (avg+max)
        
        # Smarter bottleneck - not too extreme since we have richer features now
        hidden_size = max(8, final_feature_size // 32)  # Less extreme bottleneck
        
        # Residual prediction head for latent output - better gradient flow
        self.latent_out_1 = nn.Sequential(
            nn.Dropout(self.dropout_rate),
            nn.Linear(final_feature_size, hidden_size),
            nn.GELU(),
            nn.Dropout(self.dropout_rate * 0.8)
        )
        self.latent_out_2 = nn.Linear(hidden_size, self.latent_dim_end)
        self.latent_out_skip = nn.Linear(final_feature_size, self.latent_dim_end)  # Skip connection
        
        # Residual prediction head for xs output - better gradient flow  
        self.xs_out_1 = nn.Sequential(
            nn.Dropout(self.dropout_rate),
            nn.Linear(final_feature_size, hidden_size),
            nn.GELU(),
            nn.Dropout(self.dropout_rate * 0.8)
        )
        self.xs_out_2 = nn.Linear(hidden_size, self.latent_dim * self.size2)
        self.xs_out_skip = nn.Linear(final_feature_size, self.latent_dim * self.size2)  # Skip connection

    def forward(self, x):
        x = x.reshape([-1, 1, int(math.sqrt(x.shape[-1])), int(math.sqrt(x.shape[-1]))])
        
        # Edge enhancement preprocessing for outline detection
        edge_features = self.edge_enhance(x)  # (B, 4, H, W) - Sobel X, Y, Laplacian, Identity
        
        # Combine original + edge features
        x_enhanced = torch.cat([x, edge_features], dim=1)  # (B, 5, H, W)
        
        # Shared feature extraction
        features = x_enhanced
        for block in self.backbone:
            features = block(features)
        
        # Multi-scale pooling - combine average and max pooling for richer representations
        avg_features = self.adaptive_pool(features)
        max_features = self.max_pool(features)
        features = torch.cat([avg_features, max_features], dim=1)  # Concatenate along channel dimension
        features = features.flatten(1)
        
        # Residual prediction heads with skip connections
        # Latent output with residual connection
        latent_main = self.latent_out_1(features)
        latent_main = self.latent_out_2(latent_main)
        latent_skip = self.latent_out_skip(features)
        latent_out = torch.tanh(latent_main + latent_skip * 0.1)  # Small residual weight
        
        # XS output with residual connection
        xs_main = self.xs_out_1(features)
        xs_main = self.xs_out_2(xs_main)
        xs_skip = self.xs_out_skip(features)
        xs_out = torch.tanh(xs_main + xs_skip * 0.1)  # Small residual weight
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