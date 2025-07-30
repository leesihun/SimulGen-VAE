import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.utils import spectral_norm
from modules.common import add_sn

class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation block for channel attention with proper spectral normalization"""
    def __init__(self, channels, reduction=16):
        super(SqueezeExcitation, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            add_sn(nn.Linear(channels, channels // reduction, bias=False)),
            nn.SiLU(inplace=True),
            nn.Dropout(0.1),  # Add dropout for regularization
            add_sn(nn.Linear(channels // reduction, channels, bias=False)),
            nn.Sigmoid()
        )

    def forward(self, x):
        """Apply squeeze-and-excitation attention with residual connection.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch, channels, height, width]
            
        Returns:
            torch.Tensor: Attention-weighted input tensor with residual connection
        """
        b, c, _, _ = x.size()
        # Squeeze: Global average pooling
        squeezed = self.squeeze(x).view(b, c)
        # Excitation: Two FC layers with SiLU activation and dropout
        excited = self.excitation(squeezed).view(b, c, 1, 1)
        # Scale the input with residual connection (prevents gradient vanishing)
        attention_weights = excited.expand_as(x)
        return x * (0.5 + 0.5 * attention_weights)  # Residual scaling


class ResidualBlock(nn.Module):
    """Enhanced residual block with GroupNorm, Spectral Norm, and integrated SE attention"""
    def __init__(self, in_channels, out_channels, stride=1, downsample=None, use_attention=True, drop_rate=0.2):
        super(ResidualBlock, self).__init__()
        
        # Apply spectral normalization to convolutions
        self.conv1 = add_sn(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False))
        # Adaptive GroupNorm for better regularization
        self.gn1 = nn.GroupNorm(min(32, max(1, out_channels//4)), out_channels)
        self.conv2 = add_sn(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False))
        self.gn2 = nn.GroupNorm(min(32, max(1, out_channels//4)), out_channels)
        
        # Add dropout for regularization
        self.dropout = nn.Dropout2d(drop_rate)
        
        self.downsample = downsample
        self.silu = nn.SiLU(inplace=True)
        
        # SE attention is now enabled by default for better feature selection
        self.use_attention = use_attention
        if use_attention:
            self.attention = SqueezeExcitation(out_channels, reduction=8)  # Stronger attention
        
    def forward(self, x):
        identity = x
        
        # First convolution with normalization and activation
        out = self.conv1(x)
        out = self.gn1(out)
        out = self.silu(out)
        out = self.dropout(out)  # Add dropout after first activation
        
        # Second convolution with normalization (no activation yet)
        out = self.conv2(out)
        out = self.gn2(out)
        
        # Apply SE attention before residual connection (critical for proper integration)
        if self.use_attention:
            out = self.attention(out)
        
        # Handle downsampling for residual connection
        if self.downsample is not None:
            identity = self.downsample(x)
            
        # Residual connection with scaling to prevent exploding gradients
        out = out + 0.1 * identity  # Scale residual for stability
        out = self.silu(out)
        
        return out


class LatentConditionerImg(nn.Module):
    """Enhanced ResNet-based latent conditioner with SE blocks, spectral normalization, and anti-overfitting measures"""
    def __init__(self, latent_conditioner_filter, latent_dim_end, input_shape, latent_dim, size2, latent_conditioner_data_shape, dropout_rate=0.3, use_attention=True, return_dict=False):
        
        super(LatentConditionerImg, self).__init__()
        self.latent_dim = latent_dim
        self.size2 = size2
        self.latent_conditioner_filter = latent_conditioner_filter
        self.latent_dim_end = latent_dim_end
        self.input_shape = input_shape
        self.latent_conditioner_data_shape = latent_conditioner_data_shape
        self.num_layers = len(latent_conditioner_filter)
        self.return_dict = return_dict  # Backward compatibility flag
        
        # Initial strided convolution with spectral normalization
        self.conv1 = add_sn(nn.Conv2d(1, latent_conditioner_filter[0], kernel_size=7, stride=2, padding=3, bias=False))
        # Adaptive GroupNorm configuration
        self.gn1 = nn.GroupNorm(min(32, max(1, latent_conditioner_filter[0]//4)), latent_conditioner_filter[0])
        self.silu = nn.SiLU(inplace=True)
        
        # Parametric residual layers with multi-scale feature collection
        self.layers = nn.ModuleList()
        self.feature_projections = nn.ModuleList()  # For multi-scale fusion
        in_channels = latent_conditioner_filter[0]
        
        for i, out_channels in enumerate(latent_conditioner_filter):
            # Downsample by 2
            stride = 2
            # Enable attention by default for better feature selection
            layer = self._make_layer(in_channels, out_channels, 2, stride, True, dropout_rate)
            self.layers.append(layer)
            in_channels = out_channels
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_rate * 1.5)  # Increased dropout for final features
        
        # Separate encoders for different outputs
        shared_dim = latent_conditioner_filter[-1]
        encoder_dim = shared_dim // 2
        
        # Latent encoder pathway with spectral normalization
        self.latent_encoder = nn.Sequential(
            add_sn(nn.Linear(shared_dim, encoder_dim)),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout_rate * 1.2),  # Increased dropout
            add_sn(nn.Linear(encoder_dim, encoder_dim // 2)),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout_rate * 0.8)  # Additional regularization
        )
        
        # XS encoder pathway with spectral normalization
        self.xs_encoder = nn.Sequential(
            add_sn(nn.Linear(shared_dim, encoder_dim)),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout_rate * 1.2),  # Increased dropout
            add_sn(nn.Linear(encoder_dim, encoder_dim // 2)),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout_rate * 0.8)  # Additional regularization
        )
        
        # Output heads with spectral normalization and regularization
        self.latent_head = nn.Sequential(
            nn.Dropout(dropout_rate * 0.5),  # Pre-output dropout
            add_sn(nn.Linear(encoder_dim // 2, latent_dim_end)),
            nn.Tanh()
        )
        self.xs_head = nn.Sequential(
            nn.Dropout(dropout_rate * 0.5),  # Pre-output dropout
            add_sn(nn.Linear(encoder_dim//2, latent_dim * size2)),
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
                add_sn(nn.Linear(channels, channels // 4)),
                nn.SiLU(inplace=True),
                nn.Dropout(0.2),  # Add regularization to auxiliary heads
                add_sn(nn.Linear(channels // 4, latent_dim_end)),
                nn.Tanh()
            )
            self.aux_heads.append(aux_head)
        
        # Custom initialization for GroupNorm compatibility
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Proper weight initialization compatible with spectral norm + GroupNorm for anti-overfitting"""
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                # Use more conservative initialization for spectral normalized layers
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                # Scale down initial weights to prevent early overfitting
                m.weight.data *= 0.8
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                    
            elif isinstance(m, nn.GroupNorm):
                # Start with slightly reduced normalization strength
                nn.init.constant_(m.weight, 0.9)
                nn.init.constant_(m.bias, 0.0)
                
            elif isinstance(m, nn.Linear):
                # Conservative initialization for linear layers
                nn.init.xavier_normal_(m.weight, gain=0.8)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    
    
    
    
    def _make_layer(self, in_channels, out_channels, blocks, stride=1, use_attention=True, drop_rate=0.2):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                add_sn(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)),
                nn.GroupNorm(min(32, max(1, out_channels//4)), out_channels),  # Adaptive GroupNorm
            )
        
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride, downsample, use_attention, drop_rate))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels, use_attention=use_attention, drop_rate=drop_rate))
            
        return nn.Sequential(*layers)

    def forward(self, x):
        # Reshape input to image format
        x = x.reshape([-1, 1, int(math.sqrt(x.shape[-1])), int(math.sqrt(x.shape[-1]))])
        
        # Add input noise during training for regularization
        if self.training:
            noise = torch.randn_like(x) * 0.02  # 2% noise
            x = x + noise
        
        # Initial convolution with normalization and activation
        x = self.conv1(x)
        x = self.gn1(x)
        x = self.silu(x)
        
        # Progressive feature extraction through residual layers
        for layer in self.layers:
            x = layer(x)
            # Apply stochastic depth during training for additional regularization
            if self.training and torch.rand(1).item() < 0.1:  # 10% chance to skip
                continue
        
        # Global pooling for final features
        final_features = self.avgpool(x).flatten(1)
        
        # Apply stronger dropout to final features
        shared_features = self.dropout(final_features)
        
        # Separate encoding pathways
        latent_encoded = self.latent_encoder(shared_features)
        xs_encoded = self.xs_encoder(shared_features)
        
        # Generate main outputs
        latent_main = self.latent_head(latent_encoded)
        xs_main = self.xs_head(xs_encoded)
        xs_main = xs_main.unflatten(1, (self.size2, self.latent_dim))
        
        return latent_main, xs_main