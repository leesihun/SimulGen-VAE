import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.utils import spectral_norm
from modules.common import add_sn

def get_valid_groups(channels, max_groups=32):
    """Find the largest valid group count for GroupNorm that divides channels evenly"""
    max_groups = min(max_groups, channels)
    for groups in range(max_groups, 0, -1):
        if channels % groups == 0:
            return groups
    return 1

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
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(get_valid_groups(out_channels), out_channels)
        self.conv2 = add_sn(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False))
        self.gn2 = nn.GroupNorm(get_valid_groups(out_channels), out_channels)
        
        self.downsample = downsample
        self.silu = nn.SiLU(inplace=True)
        
        self.use_attention = use_attention
        if use_attention:
            self.attention = SqueezeExcitation(out_channels, reduction=8)
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.gn1(out)
        out = self.silu(out)
        
        out = self.conv2(out)
        out = self.gn2(out)
        
        if self.use_attention:
            out = self.attention(out)
        
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
        
        self.latent_encoder = nn.Sequential(
            add_sn(nn.Linear(shared_dim, encoder_dim)),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout_rate),
        )
        
        self.xs_encoder = nn.Sequential(
            add_sn(nn.Linear(shared_dim, encoder_dim)),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout_rate),
        )
        
        self.latent_head = nn.Sequential(
            add_sn(nn.Linear(encoder_dim, latent_dim_end // 2)),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout_rate),
            add_sn(nn.Linear(latent_dim_end // 2, latent_dim_end)),
            nn.Tanh()
        )
        self.xs_head = nn.Sequential(
            add_sn(nn.Linear(encoder_dim, (latent_dim * size2) // 2)),
            nn.SiLU(inplace=True), 
            nn.Dropout(dropout_rate),
            add_sn(nn.Linear((latent_dim * size2) // 2, latent_dim * size2)),
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
                if 'head' in name:
                    # Very conservative initialization for output heads with Tanh activation and 1000x loss scaling
                    nn.init.xavier_uniform_(m.weight, gain=0.1)
                else:
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
        
        latent_encoded = self.latent_encoder(final_features)
        xs_encoded = self.xs_encoder(final_features)
        
        latent_main = self.latent_head(latent_encoded)
        xs_main = self.xs_head(xs_encoded)
        xs_main = xs_main.unflatten(1, (self.size2, self.latent_dim))
        
        if self.return_dict:
            return {
                'latent_main': latent_main,
                'xs_main': xs_main
            }
        
        return latent_main, xs_main