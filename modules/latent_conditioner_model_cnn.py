import torch
import torch.nn as nn
import torch.nn.functional as F
import math
# from torch.nn.utils import spectral_norm  # Removed for debugging

class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation block for channel attention"""
    def __init__(self, channels, reduction=16):
        super(SqueezeExcitation, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),  # Temporarily removed spectral_norm
            nn.SiLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),  # Temporarily removed spectral_norm
            nn.Sigmoid()
        )

    def forward(self, x):
        """Apply squeeze-and-excitation attention.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch, channels, height, width]
            
        Returns:
            torch.Tensor: Attention-weighted input tensor
        """
        b, c, _, _ = x.size()
        # Squeeze: Global average pooling
        squeezed = self.squeeze(x).view(b, c)
        # Excitation: Two FC layers with SiLU activation
        excited = self.excitation(squeezed).view(b, c, 1, 1)
        # Scale the input
        return x * excited.expand_as(x)


class ResidualBlock(nn.Module):
    """Enhanced residual block with GroupNorm, DropBlock, and SE attention"""
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, use_attention=False, drop_rate=0.1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)  # Temporarily removed spectral_norm
        # Simplified GroupNorm configuration
        self.gn1 = nn.GroupNorm(8, out_channels)  # Standard 8 groups
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)  # Temporarily removed spectral_norm
        self.gn2 = nn.GroupNorm(8, out_channels)  # Standard 8 groups
        
        self.downsample = downsample
        self.silu = nn.SiLU(inplace=True)
        
        # Optional SE attention
        self.use_attention = use_attention
        if use_attention:
            self.attention = SqueezeExcitation(out_channels)
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.gn1(out)
        out = self.silu(out)
        
        out = self.conv2(out)
        out = self.gn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out += identity
        out = self.silu(out)
        
        # Apply SE attention if enabled
        if self.use_attention:
            out = self.attention(out)
        
        return out


class LatentConditionerImg(nn.Module):
    """Enhanced ResNet-based latent conditioner with SE blocks, SiLU activation, and multi-scale feature fusion"""
    def __init__(self, latent_conditioner_filter, latent_dim_end, input_shape, latent_dim, size2, latent_conditioner_data_shape, dropout_rate=0.3, use_attention=False, return_dict=False):
        
        super(LatentConditionerImg, self).__init__()
        self.latent_dim = latent_dim
        self.size2 = size2
        self.latent_conditioner_filter = latent_conditioner_filter
        self.latent_dim_end = latent_dim_end
        self.input_shape = input_shape
        self.latent_conditioner_data_shape = latent_conditioner_data_shape
        self.num_layers = len(latent_conditioner_filter)
        self.return_dict = return_dict  # Backward compatibility flag
        
        # Initial strided convolution (replaces conv + maxpool)
        self.conv1 = nn.Conv2d(1, latent_conditioner_filter[0], kernel_size=7, stride=2, padding=3, bias=False)
        # Simplified GroupNorm configuration
        self.gn1 = nn.GroupNorm(8, latent_conditioner_filter[0])  # Standard 8 groups
        self.silu = nn.SiLU(inplace=True)
        
        # Parametric residual layers with multi-scale feature collection
        self.layers = nn.ModuleList()
        self.feature_projections = nn.ModuleList()  # For multi-scale fusion
        in_channels = latent_conditioner_filter[0]
        
        for i, out_channels in enumerate(latent_conditioner_filter):
            # Downsample by 2
            stride = 2
            layer = self._make_layer(in_channels, out_channels, 2, stride, use_attention, dropout_rate)
            self.layers.append(layer)
            in_channels = out_channels
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_rate)  # Reduced for debugging
        
        # Separate encoders for different outputs
        shared_dim = latent_conditioner_filter[-1]
        encoder_dim = shared_dim // 2
        
        # Latent encoder pathway
        self.latent_encoder = nn.Sequential(
            nn.Linear(shared_dim, encoder_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(encoder_dim, encoder_dim // 2),
            nn.SiLU(inplace=True)
        )
        
        # XS encoder pathway  
        self.xs_encoder = nn.Sequential(
            nn.Linear(shared_dim, encoder_dim),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(encoder_dim, encoder_dim // 2),
            nn.SiLU(inplace=True)
        )
        
        # Output heads with Tanh activation for [-1, 1] scaling
        self.latent_head = nn.Sequential(
            nn.Linear(encoder_dim // 2, latent_dim_end),
            nn.Tanh()
        )
        self.xs_head = nn.Sequential(
            nn.Linear(encoder_dim//2, latent_dim * size2),
            nn.Tanh()
        )
        
        # Uncertainty estimation heads
        self.latent_uncertainty = nn.Sequential(
            nn.Linear(encoder_dim // 2, latent_dim_end),
            nn.Softplus()  # Ensures positive uncertainty values
        )
        self.xs_uncertainty = nn.Sequential(
            nn.Linear(encoder_dim, latent_dim * size2),
            nn.Softplus()
        )
        
        # Auxiliary classification heads for intermediate supervision
        self.aux_heads = nn.ModuleList()
        for i, channels in enumerate(latent_conditioner_filter):
            aux_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(channels, channels // 4),  # Removed spectral_norm
                nn.SiLU(inplace=True),
                nn.Linear(channels // 4, latent_dim_end),  # Removed spectral_norm
                nn.Tanh()
            )
            self.aux_heads.append(aux_head)
        
        # Custom initialization for GroupNorm compatibility
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Proper weight initialization compatible with spectral norm + GroupNorm"""
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
                
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    
    
    
    
    def _make_layer(self, in_channels, out_channels, blocks, stride=1, use_attention=False, drop_rate=0.1):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),  # Temporarily removed spectral_norm
                nn.GroupNorm(8, out_channels),  # Simplified GroupNorm
            )
        
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride, downsample, use_attention, drop_rate))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels, use_attention=use_attention, drop_rate=drop_rate))
            
        return nn.Sequential(*layers)

    def forward(self, x):
        # Reshape input to image format
        x = x.reshape([-1, 1, int(math.sqrt(x.shape[-1])), int(math.sqrt(x.shape[-1]))])
        
        # Perform convolution
        x = self.conv1(x)
        x = self.gn1(x)
        x = self.silu(x)
        
        for layer in self.layers:
            x = layer(x)
        
        # Global pooling for final features
        final_features = self.avgpool(x).flatten(1)
        
        # Skip multi-scale fusion - use final features directly
        shared_features = self.dropout(final_features)
        
        # Separate encoding pathways
        latent_encoded = self.latent_encoder(shared_features)
        xs_encoded = self.xs_encoder(shared_features)
        
        # Generate main outputs
        latent_main = self.latent_head(latent_encoded)
        xs_main = self.xs_head(xs_encoded)
        xs_main = xs_main.unflatten(1, (self.size2, self.latent_dim))
        
        return latent_main, xs_main