import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.utils import spectral_norm
from modules.common import add_sn
import numpy as np

def get_valid_groups(channels, max_groups=32):
    """Find the largest valid group count for GroupNorm that divides channels evenly"""
    max_groups = min(max_groups, channels)
    for groups in range(max_groups, 0, -1):
        if channels % groups == 0:
            return groups
    return 1

class SpectralDropout(nn.Module):
    """Spectral Dropout: More effective than standard dropout for conv features"""
    def __init__(self, channels, dropout_rate=0.3):
        super(SpectralDropout, self).__init__()
        self.dropout_rate = dropout_rate
        self.channels = channels
        
    def forward(self, x):
        if not self.training or self.dropout_rate == 0:
            return x
            
        # For 1D features (after avgpool)
        if x.dim() == 2:
            # Apply dropout to random spectral components
            mask = torch.bernoulli(torch.full_like(x, 1 - self.dropout_rate))
            return x * mask / (1 - self.dropout_rate)
        return x

class DropBlock2D(nn.Module):
    """DropBlock: Structured dropout for spatial features"""
    def __init__(self, drop_rate=0.3, block_size=7):
        super(DropBlock2D, self).__init__()
        self.drop_rate = drop_rate
        self.block_size = block_size
        
    def forward(self, x):
        if not self.training or self.drop_rate == 0:
            return x
            
        # For 1D features, apply as regular dropout
        if x.dim() == 2:
            mask = torch.bernoulli(torch.full_like(x, 1 - self.drop_rate))
            return x * mask / (1 - self.drop_rate)
            
        # For 2D features, would implement structured dropout
        # Here we simplify to regular dropout for compatibility
        return F.dropout2d(x, p=self.drop_rate, training=self.training)

class CrossLevelAttention(nn.Module):
    """Cross-attention between 32D and 8D latent levels"""
    def __init__(self, d_model, latent_32_dim, latent_8_dim, dropout=0.1):
        super(CrossLevelAttention, self).__init__()
        self.d_model = d_model
        self.latent_32_dim = latent_32_dim
        self.latent_8_dim = latent_8_dim
        
        # Projection layers
        self.query_32 = add_sn(nn.Linear(d_model, d_model))
        self.key_8 = add_sn(nn.Linear(d_model, d_model))
        self.value_8 = add_sn(nn.Linear(d_model, d_model))
        
        self.query_8 = add_sn(nn.Linear(d_model, d_model))
        self.key_32 = add_sn(nn.Linear(d_model, d_model))
        self.value_32 = add_sn(nn.Linear(d_model, d_model))
        
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, shared_features):
        # shared_features: [batch, d_model]
        batch_size = shared_features.size(0)
        
        # Compute attention weights between different latent target spaces
        q_32 = self.query_32(shared_features)  # Query for 32D space
        k_8 = self.key_8(shared_features)      # Key for 8D space  
        v_8 = self.value_8(shared_features)    # Value for 8D space
        
        q_8 = self.query_8(shared_features)    # Query for 8D space
        k_32 = self.key_32(shared_features)    # Key for 32D space
        v_32 = self.value_32(shared_features)  # Value for 32D space
        
        # Cross-attention: 32D attends to 8D and vice versa
        attn_32_to_8 = torch.softmax(torch.matmul(q_32.unsqueeze(1), k_8.unsqueeze(-1)).squeeze() / math.sqrt(self.d_model), dim=-1)
        attn_8_to_32 = torch.softmax(torch.matmul(q_8.unsqueeze(1), k_32.unsqueeze(-1)).squeeze() / math.sqrt(self.d_model), dim=-1)
        
        # Apply attention
        enhanced_32 = shared_features + attn_32_to_8.unsqueeze(-1) * v_8
        enhanced_8 = shared_features + attn_8_to_32.unsqueeze(-1) * v_32
        
        # Combine and normalize
        combined = self.norm(enhanced_32 + enhanced_8)
        return self.dropout(combined)

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            add_sn(nn.Conv2d(channels, channels // reduction, 1, bias=False)),
            nn.SiLU(inplace=True),
            add_sn(nn.Conv2d(channels // reduction, channels, 1, bias=False))
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv1 = add_sn(nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False))
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x_cat = torch.cat([avg_out, max_out], dim=1)
        x_out = self.conv1(x_cat)
        return self.sigmoid(x_out)

class CBAM(nn.Module):
    def __init__(self, channels, reduction=8, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)
        
    def forward(self, x):
        # Channel attention
        x = x * self.channel_attention(x)
        # Spatial attention  
        x = x * self.spatial_attention(x)
        return x

class SqueezeExcitation(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SqueezeExcitation, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            add_sn(nn.Linear(channels, channels // reduction, bias=False)),
            nn.SiLU(inplace=True),
            add_sn(nn.Linear(channels // reduction, channels, bias=False)),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        squeezed = self.squeeze(x).view(b, c)
        excited = self.excitation(squeezed).view(b, c, 1, 1)
        attention_weights = excited.expand_as(x)
        return x * (0.5 + 0.5 * attention_weights)


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, use_attention=True, drop_rate=0.2):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = add_sn(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False))
        self.gn1 = nn.GroupNorm(get_valid_groups(out_channels), out_channels)
        self.dropout1 = nn.Dropout2d(drop_rate * 0.5)  # Spatial dropout after first conv
        
        self.conv2 = add_sn(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False))
        self.gn2 = nn.GroupNorm(get_valid_groups(out_channels), out_channels)
        self.dropout2 = nn.Dropout2d(drop_rate)  # Stronger dropout after second conv
        
        self.downsample = downsample
        self.silu = nn.SiLU(inplace=True)
        
        self.use_attention = use_attention
        if use_attention:
            self.attention = CBAM(out_channels, reduction=8)

        self.drop_rate = drop_rate
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.gn1(out)
        out = self.silu(out)
        out = self.dropout1(out)  # Add spatial dropout
        
        out = self.conv2(out)
        out = self.gn2(out)
        out = self.dropout2(out)  # Add stronger spatial dropout
        
        if self.use_attention:
            out = self.attention(out)
            # Apply spectral dropout after attention
            out = F.dropout2d(out, p=self.drop_rate * 0.3, training=self.training)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out = out + identity
        out = self.silu(out)
        
        return out


class LatentConditionerImg(nn.Module):
    def __init__(self, latent_conditioner_filter, latent_dim_end, input_shape, latent_dim, size2, latent_conditioner_data_shape, dropout_rate=0.3, use_attention=True, return_dict=False):
        
        super(LatentConditionerImg, self).__init__()
        self.latent_dim = latent_dim
        self.size2 = size2
        self.latent_conditioner_filter = latent_conditioner_filter
        self.latent_dim_end = latent_dim_end
        self.input_shape = input_shape
        self.latent_conditioner_data_shape = latent_conditioner_data_shape
        self.num_layers = len(latent_conditioner_filter)
        self.return_dict = return_dict
        
        self.conv1 = nn.Conv2d(1, latent_conditioner_filter[0], kernel_size=7, stride=2, padding=3, bias=False)
        self.gn1 = nn.GroupNorm(get_valid_groups(latent_conditioner_filter[0]), latent_conditioner_filter[0])
        self.silu = nn.SiLU(inplace=True)
        
        self.layers = nn.ModuleList()
        in_channels = latent_conditioner_filter[0]
        
        for i, out_channels in enumerate(latent_conditioner_filter):
            # Only downsample every other layer to preserve spatial information
            stride = 2 if i % 2 == 0 else 1
            layer = self._make_layer(in_channels, out_channels, 1, stride, True, dropout_rate)
            self.layers.append(layer)
            in_channels = out_channels
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Single-scale features
        final_features = latent_conditioner_filter[-1]
        shared_dim = latent_conditioner_filter[-1]
        encoder_dim = latent_dim_end
        
        # Cross-attention between latent levels
        self.cross_attention = CrossLevelAttention(
            d_model=shared_dim, 
            latent_32_dim=latent_dim_end,
            latent_8_dim=latent_dim * size2,
            dropout=dropout_rate * 0.5
        )
        
        # Enhanced latent heads with spectral dropout
        self.latent_head = nn.Sequential(
            SpectralDropout(shared_dim, dropout_rate * 0.7),
            add_sn(nn.Linear(shared_dim, shared_dim // 2)),
            nn.SiLU(inplace=True),
            DropBlock2D(drop_rate=dropout_rate * 0.5, block_size=1),
            add_sn(nn.Linear(shared_dim // 2, latent_dim_end)),
            nn.Tanh()
        )
        self.xs_head = nn.Sequential(
            SpectralDropout((shared_dim), dropout_rate * 0.7),
            add_sn(nn.Linear(shared_dim, (shared_dim) // 2)),
            nn.SiLU(inplace=True), 
            DropBlock2D(drop_rate=dropout_rate * 0.5, block_size=1),
            add_sn(nn.Linear((shared_dim) // 2, latent_dim * size2)),
            nn.Tanh()   
        )

        self._initialize_weights()
    
    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                # He initialization for SiLU activation
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0.0)
                
            elif isinstance(m, nn.Linear):
                # He initialization for SiLU activation in intermediate layers
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def _make_layer(self, in_channels, out_channels, blocks, stride=1, use_attention=True, drop_rate=0.2):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(get_valid_groups(out_channels), out_channels),
            )
        
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride, downsample, use_attention, drop_rate))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels, use_attention=use_attention, drop_rate=drop_rate))
            
        return nn.Sequential(*layers)

    def forward(self, x):
        x = x.reshape([-1, 1, int(math.sqrt(x.shape[-1])), int(math.sqrt(x.shape[-1]))])
        
        # if self.training:
        #     noise = torch.randn_like(x) * 0.02
        #     x = x + noise
        
        x = self.conv1(x)
        x = self.gn1(x)
        x = self.silu(x)
        
        # Simple feature extraction
        for layer in self.layers:
            x = layer(x)
        
        # Extract final features
        final_features = self.avgpool(x).flatten(1)
        
        # Apply cross-level attention enhancement
        enhanced_features = self.cross_attention(final_features)
        
        # Generate latent representations with enhanced features
        latent_main = self.latent_head(enhanced_features)
        xs_main = self.xs_head(enhanced_features)
        xs_main = xs_main.unflatten(1, (self.size2, self.latent_dim))
        
        if self.return_dict:
            return {
                'latent_main': latent_main,
                'xs_main': xs_main
            }
        
        return latent_main, xs_main