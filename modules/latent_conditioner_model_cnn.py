import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SpatialAttention(nn.Module):
    """Simple spatial attention module"""
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # Channel-wise average and max pooling
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        
        # Concatenate and apply convolution
        combined = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(combined))
        
        return x * attention


class ResidualBlock(nn.Module):
    """Basic residual block with optional downsampling and spatial attention"""
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, use_attention=False):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = downsample
        self.relu = nn.ReLU(inplace=True)
        
        # Optional spatial attention
        self.use_attention = use_attention
        if use_attention:
            self.attention = SpatialAttention()
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.relu(out)
        
        # Apply spatial attention if enabled
        if self.use_attention:
            out = self.attention(out)
        
        return out


class LatentConditionerImg(nn.Module):
    """Parametric ResNet-based latent conditioner for image data"""
    def __init__(self, latent_conditioner_filter, latent_dim_end, input_shape, latent_dim, size2, latent_conditioner_data_shape, dropout_rate=0.3, use_attention=False):
        
        super(LatentConditionerImg, self).__init__()
        self.latent_dim = latent_dim
        self.size2 = size2
        self.latent_conditioner_filter = latent_conditioner_filter
        self.latent_dim_end = latent_dim_end
        self.input_shape = input_shape
        self.latent_conditioner_data_shape = latent_conditioner_data_shape
        self.num_layers = len(latent_conditioner_filter)
        
        # Initial strided convolution (replaces conv + maxpool)
        self.conv1 = nn.Conv2d(1, latent_conditioner_filter[0], kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(latent_conditioner_filter[0])
        self.relu = nn.ReLU(inplace=True)
        
        # Parametric residual layers
        self.layers = nn.ModuleList()
        in_channels = latent_conditioner_filter[0]
        
        for i, out_channels in enumerate(latent_conditioner_filter):
            # First layer has stride=1, others have stride=2 for downsampling
            stride = 1 if i == 0 else 2
            layer = self._make_layer(in_channels, out_channels, 2, stride, use_attention)
            self.layers.append(layer)
            in_channels = out_channels
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_rate)
        
        # Output heads
        final_channels = latent_conditioner_filter[-1]
        self.latent_head = nn.Linear(final_channels, latent_dim_end)
        self.xs_head = nn.Linear(final_channels, latent_dim * size2)
    
    def _make_layer(self, in_channels, out_channels, blocks, stride=1, use_attention=False):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride, downsample, use_attention))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels, use_attention=use_attention))
            
        return nn.Sequential(*layers)

    def forward(self, x):
        # Reshape input to image format
        x = x.reshape([-1, 1, int(math.sqrt(x.shape[-1])), int(math.sqrt(x.shape[-1]))])
        
        # Initial strided convolution
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        # Pass through all parametric residual layers
        for layer in self.layers:
            x = layer(x)
        
        # Global pooling and flatten
        x = self.avgpool(x)
        features = x.flatten(1)
        features = self.dropout(features)
        
        # Generate outputs
        latent_main = self.latent_head(features)
        xs_main = self.xs_head(features)
        xs_main = xs_main.unflatten(1, (self.size2, self.latent_dim))

        return latent_main, xs_main