import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch.nn.utils import spectral_norm


class DropBlock2D(nn.Module):
    """DropBlock regularization for 2D feature maps"""
    def __init__(self, drop_rate=0.1, block_size=7):
        super(DropBlock2D, self).__init__()
        self.drop_rate = drop_rate
        self.block_size = block_size

    def forward(self, x):
        if not self.training:
            return x
        
        # Calculate gamma (keep probability)
        gamma = self.drop_rate / (self.block_size ** 2)
        
        # Sample mask
        batch_size, channels, height, width = x.shape
        w_i, h_i = torch.meshgrid(torch.arange(width, device=x.device), torch.arange(height, device=x.device), indexing='ij')
        valid_block = ((w_i >= self.block_size // 2) & (w_i < width - self.block_size // 2) &
                      (h_i >= self.block_size // 2) & (h_i < height - self.block_size // 2))
        valid_block = torch.reshape(valid_block, (1, 1, height, width)).float().to(x.device)
        
        uniform_noise = torch.rand_like(x)
        block_mask = ((uniform_noise * valid_block) <= gamma).float()
        block_mask = -F.max_pool2d(-block_mask, kernel_size=self.block_size,
                                  stride=1, padding=self.block_size // 2)
        
        # Fix division by zero that causes NaN
        mask_sum = block_mask.sum()
        normalize_scale = block_mask.numel() / torch.clamp(mask_sum, min=1e-8)
        return x * block_mask * normalize_scale


class SqueezeExcitation(nn.Module):
    """Squeeze-and-Excitation block for channel attention"""
    def __init__(self, channels, reduction=16):
        super(SqueezeExcitation, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            spectral_norm(nn.Linear(channels, channels // reduction, bias=False)),
            nn.SiLU(inplace=True),
            spectral_norm(nn.Linear(channels // reduction, channels, bias=False)),
            nn.Sigmoid()
        )

    def forward(self, x):
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
        
        self.conv1 = spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False))
        # Fix GroupNorm to prevent NaN with small channel counts
        num_groups1 = min(32, max(1, out_channels // 4))
        self.gn1 = nn.GroupNorm(num_groups1, out_channels, eps=1e-6)
        self.conv2 = spectral_norm(nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False))
        num_groups2 = min(32, max(1, out_channels // 4))
        self.gn2 = nn.GroupNorm(num_groups2, out_channels, eps=1e-6)
        
        self.downsample = downsample
        self.silu = nn.SiLU(inplace=True)
        self.dropblock = DropBlock2D(drop_rate=drop_rate)
        
        # Optional SE attention
        self.use_attention = use_attention
        if use_attention:
            self.attention = SqueezeExcitation(out_channels)
        
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.gn1(out)
        out = self.silu(out)
        out = self.dropblock(out)
        
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
        
        # Initial strided convolution (replaces conv + maxpool) - NO spectral norm on first layer
        self.conv1 = nn.Conv2d(1, latent_conditioner_filter[0], kernel_size=7, stride=2, padding=3, bias=False)
        # Fix GroupNorm to prevent NaN with small channel counts
        num_groups_init = min(32, max(1, latent_conditioner_filter[0] // 4))
        self.gn1 = nn.GroupNorm(num_groups_init, latent_conditioner_filter[0], eps=1e-6)
        self.silu = nn.SiLU(inplace=True)
        # Reduce DropBlock rate for initial stability
        self.initial_dropblock = DropBlock2D(drop_rate=max(0.01, dropout_rate * 0.1))
        
        # Parametric residual layers with multi-scale feature collection
        self.layers = nn.ModuleList()
        self.feature_projections = nn.ModuleList()  # For multi-scale fusion
        in_channels = latent_conditioner_filter[0]
        
        for i, out_channels in enumerate(latent_conditioner_filter):
            # First layer has stride=1, others have stride=2 for downsampling
            stride = 1 if i == 0 else 2
            layer = self._make_layer(in_channels, out_channels, 2, stride, use_attention, dropout_rate)
            self.layers.append(layer)
            
            # Add projection layers for multi-scale feature fusion
            self.feature_projections.append(
                nn.Sequential(
                    nn.AdaptiveAvgPool2d(1),
                    nn.Conv2d(out_channels, latent_conditioner_filter[-1] // 4, 1),
                    nn.SiLU(inplace=True)
                )
            )
            in_channels = out_channels
        
        # Global average pooling
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(dropout_rate)
        
        # Multi-scale feature fusion layer
        fusion_channels = latent_conditioner_filter[-1] + (latent_conditioner_filter[-1] // 4) * len(latent_conditioner_filter)
        self.feature_fusion = nn.Sequential(
            spectral_norm(nn.Linear(fusion_channels, latent_conditioner_filter[-1])),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout_rate // 2)
        )
        
        # Separate encoders for different outputs
        shared_dim = latent_conditioner_filter[-1]
        encoder_dim = shared_dim // 2
        
        # Latent encoder pathway
        self.latent_encoder = nn.Sequential(
            spectral_norm(nn.Linear(shared_dim, encoder_dim)),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout_rate // 3),
            spectral_norm(nn.Linear(encoder_dim, encoder_dim // 2)),
            nn.SiLU(inplace=True)
        )
        
        # XS encoder pathway  
        self.xs_encoder = nn.Sequential(
            spectral_norm(nn.Linear(shared_dim, encoder_dim)),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout_rate // 3),
            spectral_norm(nn.Linear(encoder_dim, encoder_dim)),
            nn.SiLU(inplace=True)
        )
        
        # Output heads with Tanh activation for [-1, 1] scaling
        self.latent_head = nn.Sequential(
            spectral_norm(nn.Linear(encoder_dim // 2, latent_dim_end)),
            nn.Tanh()
        )
        self.xs_head = nn.Sequential(
            spectral_norm(nn.Linear(encoder_dim, latent_dim * size2)),
            nn.Tanh()
        )
        
        # Uncertainty estimation heads
        self.latent_uncertainty = nn.Sequential(
            spectral_norm(nn.Linear(encoder_dim // 2, latent_dim_end)),
            nn.Softplus()  # Ensures positive uncertainty values
        )
        self.xs_uncertainty = nn.Sequential(
            spectral_norm(nn.Linear(encoder_dim, latent_dim * size2)),
            nn.Softplus()
        )
        
        # Auxiliary classification heads for intermediate supervision
        self.aux_heads = nn.ModuleList()
        for i, channels in enumerate(latent_conditioner_filter):
            aux_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                spectral_norm(nn.Linear(channels, channels // 4)),
                nn.SiLU(inplace=True),
                spectral_norm(nn.Linear(channels // 4, latent_dim_end)),
                nn.Tanh()
            )
            self.aux_heads.append(aux_head)
        
        # Custom initialization for GroupNorm compatibility
        self._initialize_weights()
        
        # Validate initialization worked
        self._validate_weights_after_init()
        
        # Add hooks to monitor weight corruption during training
        self._add_weight_monitoring_hooks()
    
    def _initialize_weights(self):
        """Conservative weight initialization to prevent NaN with spectral norm + GroupNorm"""
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                # Manual initialization to avoid potential nn.init bugs
                std = 0.001  # Ultra-small fixed std
                with torch.no_grad():
                    # Manual normal distribution
                    m.weight.data = torch.randn_like(m.weight) * std
                    if m.bias is not None:
                        m.bias.data.zero_()
                    
                    # Immediate validation
                    if torch.isnan(m.weight).any():
                        print(f"❌ NaN in {name} weight after manual init!")
                        m.weight.data.zero_()  # Fallback to zeros
                    else:
                        print(f"   ✅ Manually initialized {name}: std={std:.4f}, range=[{m.weight.min():.6f}, {m.weight.max():.6f}]")
            elif isinstance(m, nn.GroupNorm):
                # Initialize GroupNorm parameters carefully
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                # Manual linear layer initialization
                std = 0.01  # Very small for stability
                with torch.no_grad():
                    m.weight.data = torch.randn_like(m.weight) * std
                    if m.bias is not None:
                        m.bias.data.zero_()
                    
                    # Validation
                    if torch.isnan(m.weight).any():
                        print(f"❌ NaN in {name} linear weight after manual init!")
                        m.weight.data.zero_()
                    else:
                        print(f"   ✅ Manually initialized {name}: std={std:.4f}, range=[{m.weight.min():.6f}, {m.weight.max():.6f}]")
    
    def _validate_weights_after_init(self):
        """Check if weights contain NaN immediately after initialization"""
        print("🔍 Validating weights after initialization:")
        for name, param in self.named_parameters():
            if torch.isnan(param).any():
                print(f"❌ CRITICAL: {name} contains NaN immediately after init!")
                print(f"   Shape: {param.shape}, NaN count: {torch.isnan(param).sum()}")
                # Force fix the NaN weights
                param.data = torch.where(torch.isnan(param), torch.zeros_like(param), param)
            else:
                print(f"✅ {name}: OK (min={param.min():.6f}, max={param.max():.6f})")
    
    def _add_weight_monitoring_hooks(self):
        """Add hooks to detect when weights become NaN during training"""
        def check_weight_nan_hook(module, grad_input, grad_output):
            if hasattr(module, 'weight') and torch.isnan(module.weight).any():
                print(f"🚨 WEIGHT CORRUPTION: {module.__class__.__name__} weights became NaN during training!")
                module.weight.data = torch.where(torch.isnan(module.weight), torch.zeros_like(module.weight), module.weight)
        
        # Add hooks to critical layers
        self.conv1.register_backward_hook(check_weight_nan_hook)
        for layer in self.layers:
            layer.register_backward_hook(check_weight_nan_hook)
    
    def _make_layer(self, in_channels, out_channels, blocks, stride=1, use_attention=False, drop_rate=0.1):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                spectral_norm(nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)),
                nn.GroupNorm(min(32, max(1, out_channels // 4)), out_channels, eps=1e-6),
            )
        
        layers = []
        layers.append(ResidualBlock(in_channels, out_channels, stride, downsample, use_attention, drop_rate))
        for _ in range(1, blocks):
            layers.append(ResidualBlock(out_channels, out_channels, use_attention=use_attention, drop_rate=drop_rate))
            
        return nn.Sequential(*layers)

    def forward(self, x):
        # Reshape input to image format
        x = x.reshape([-1, 1, int(math.sqrt(x.shape[-1])), int(math.sqrt(x.shape[-1]))])
        
        # Comprehensive input validation
        if torch.isnan(x).any():
            print(f"🚨 NaN detected in input to LatentConditionerImg")
            x = torch.nan_to_num(x, nan=0.0)
        
        # Check for extreme values that could cause NaN after conv
        if torch.isinf(x).any():
            print(f"🚨 Inf detected in input - clamping")
            x = torch.clamp(x, min=-1e6, max=1e6)
        
        if x.abs().max() > 1e3:
            print(f"🚨 Extreme input values detected: max={x.abs().max():.2f}")
            print(f"   Input stats: min={x.min():.4f}, max={x.max():.4f}, mean={x.mean():.4f}, std={x.std():.4f}")
            # Normalize extreme inputs
            x = torch.clamp(x, min=-10.0, max=10.0)
        
        # Initial strided convolution with detailed debugging
        x = self.conv1(x)
        if torch.isnan(x).any():
            print(f"🚨 NaN detected after conv1")
            print(f"   Conv1 weight stats: min={self.conv1.weight.min():.6f}, max={self.conv1.weight.max():.6f}")
            print(f"   Conv1 weight has NaN: {torch.isnan(self.conv1.weight).any()}")
            print(f"   Output stats before fix: min={x.min():.6f}, max={x.max():.6f}")
            x = torch.nan_to_num(x, nan=0.0)
        
        x = self.gn1(x)
        if torch.isnan(x).any():
            print(f"🚨 NaN detected after GroupNorm1")
            print(f"   GroupNorm stats: mean={x.mean():.6f}, std={x.std():.6f}")
            print(f"   Channel count: {x.shape[1]}, Groups: {self.gn1.num_groups}")
            x = torch.nan_to_num(x, nan=0.0)
        
        x = self.silu(x)
        if torch.isnan(x).any():
            print(f"🚨 NaN detected after SiLU activation")
            x = torch.nan_to_num(x, nan=0.0)
        
        # Skip DropBlock during initial debugging
        if self.training:
            x = self.initial_dropblock(x)
            if torch.isnan(x).any():
                print(f"🚨 NaN detected after DropBlock")
                x = torch.nan_to_num(x, nan=0.0)
        
        # Collect multi-scale features
        multi_scale_features = []
        
        # Pass through all parametric residual layers
        for i, layer in enumerate(self.layers):
            x = layer(x)
            # Extract and project features from each scale
            projected_feat = self.feature_projections[i](x)
            multi_scale_features.append(projected_feat.flatten(1))
        
        # Global pooling for final features
        final_features = self.avgpool(x).flatten(1)
        
        # Concatenate multi-scale features
        all_features = torch.cat([final_features] + multi_scale_features, dim=1)
        
        # Fuse multi-scale features
        fused_features = self.feature_fusion(all_features)
        shared_features = self.dropout(fused_features)
        
        # Separate encoding pathways
        latent_encoded = self.latent_encoder(shared_features)
        xs_encoded = self.xs_encoder(shared_features)
        
        # Generate main outputs
        latent_main = self.latent_head(latent_encoded)
        xs_main = self.xs_head(xs_encoded)
        xs_main = xs_main.unflatten(1, (self.size2, self.latent_dim))
        
        # Final NaN check and replacement
        if torch.isnan(latent_main).any():
            print(f"🚨 NaN detected in latent_main output - replacing with zeros")
            latent_main = torch.nan_to_num(latent_main, nan=0.0)
        if torch.isnan(xs_main).any():
            print(f"🚨 NaN detected in xs_main output - replacing with zeros")
            xs_main = torch.nan_to_num(xs_main, nan=0.0)
        
        # Generate uncertainty estimates
        latent_uncertainty = self.latent_uncertainty(latent_encoded)
        xs_uncertainty = self.xs_uncertainty(xs_encoded)
        xs_uncertainty = xs_uncertainty.unflatten(1, (self.size2, self.latent_dim))
        
        # Generate auxiliary outputs for intermediate supervision
        aux_outputs = []
        x_aux = x.reshape([-1, 1, int(math.sqrt(x.shape[-1])), int(math.sqrt(x.shape[-1]))])
        x_aux = self.conv1(x_aux)
        x_aux = self.gn1(x_aux)
        x_aux = self.silu(x_aux)
        x_aux = self.initial_dropblock(x_aux)
        
        for i, layer in enumerate(self.layers):
            x_aux = layer(x_aux)
            aux_out = self.aux_heads[i](x_aux)
            aux_outputs.append(aux_out)

        # Return format based on compatibility mode
        if self.return_dict:
            return {
                'latent_main': latent_main,
                'xs_main': xs_main,
                'latent_uncertainty': latent_uncertainty,
                'xs_uncertainty': xs_uncertainty,
                'aux_outputs': aux_outputs
            }
        else:
            # Backward compatible tuple format
            return latent_main, xs_main