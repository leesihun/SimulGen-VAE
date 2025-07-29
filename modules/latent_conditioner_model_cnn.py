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
        """Apply DropBlock regularization during training.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch, channels, height, width]
            
        Returns:
            torch.Tensor: Regularized tensor with dropped blocks during training,
                         original tensor during evaluation
        """
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
        # Robust GroupNorm configuration to prevent NaN with small channel counts
        num_groups1 = min(16, max(2, out_channels // 2))  # At least 2 groups, max 16
        self.gn1 = nn.GroupNorm(num_groups1, out_channels, eps=1e-5)  # Increased epsilon for stability
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)  # Temporarily removed spectral_norm
        num_groups2 = min(16, max(2, out_channels // 2))  # At least 2 groups, max 16
        self.gn2 = nn.GroupNorm(num_groups2, out_channels, eps=1e-5)  # Increased epsilon for stability
        
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
        
        # Initial strided convolution (replaces conv + maxpool) - NO spectral norm on first layer for stability
        self.conv1 = nn.Conv2d(1, latent_conditioner_filter[0], kernel_size=7, stride=2, padding=3, bias=False)
        # Robust GroupNorm configuration to prevent NaN with small channel counts
        num_groups_init = min(16, max(2, latent_conditioner_filter[0] // 2))  # At least 2 groups, max 16
        self.gn1 = nn.GroupNorm(num_groups_init, latent_conditioner_filter[0], eps=1e-5)  # Increased epsilon for stability
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
        
        # Add automatic NaN recovery mechanism
        self.nan_recovery_count = 0
        self.max_nan_recoveries = 5
    
    def _initialize_weights(self):
        """Proper weight initialization compatible with spectral norm + GroupNorm"""
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                # Use proper He initialization for ReLU/SiLU activations
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                
                # Validation after proper initialization
                if torch.isnan(m.weight).any():
                    print(f"❌ NaN in {name} weight after He init!")
                    # Fallback to Xavier if He fails
                    nn.init.xavier_normal_(m.weight)
                    if torch.isnan(m.weight).any():
                        print(f"❌ NaN persists after Xavier init! Using zeros.")
                        m.weight.data.zero_()
                else:
                    print(f"   ✅ He initialized {name}: range=[{m.weight.min():.6f}, {m.weight.max():.6f}], std={m.weight.std():.6f}")
                    
            elif isinstance(m, nn.GroupNorm):
                # Standard GroupNorm initialization
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
                print(f"   ✅ GroupNorm initialized {name}: weight=1.0, bias=0.0")
                
            elif isinstance(m, nn.Linear):
                # Use Xavier initialization for linear layers (works well with Tanh output)
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
                
                # Validation
                if torch.isnan(m.weight).any():
                    print(f"❌ NaN in {name} linear weight after Xavier init!")
                    m.weight.data.zero_()
                else:
                    print(f"   ✅ Xavier initialized {name}: range=[{m.weight.min():.6f}, {m.weight.max():.6f}], std={m.weight.std():.6f}")
    
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
        """Add hooks to monitor gradients and detect weight corruption"""
        def gradient_monitor_hook(module, grad_input, grad_output):
            if hasattr(module, 'weight') and module.weight.grad is not None:
                grad_norm = module.weight.grad.norm()
                grad_max = module.weight.grad.abs().max()
                
                print(f"🔍 {module.__class__.__name__} gradient: norm={grad_norm:.6f}, max={grad_max:.6f}")
                
                # Check for exploding gradients
                if grad_norm > 1000 or grad_max > 1000:
                    print(f"🚨 EXPLODING GRADIENT in {module.__class__.__name__}!")
                    print(f"   Gradient norm: {grad_norm:.2f}, max: {grad_max:.2f}")
                    # Clip the exploding gradient
                    torch.nn.utils.clip_grad_norm_([module.weight], max_norm=1.0)
                    print(f"   Clipped gradient norm: {module.weight.grad.norm():.6f}")
                
                # Check for NaN gradients
                if torch.isnan(module.weight.grad).any():
                    print(f"🚨 NaN GRADIENT in {module.__class__.__name__}!")
                    module.weight.grad.data.zero_()
            
            # Check weights after gradient update (this runs after optimizer.step())
            if hasattr(module, 'weight') and torch.isnan(module.weight).any():
                print(f"🚨 WEIGHT CORRUPTION: {module.__class__.__name__} weights became NaN after gradient update!")
                module.weight.data = torch.where(torch.isnan(module.weight), torch.zeros_like(module.weight), module.weight)
        
        # Monitor conv1 gradients closely
        self.conv1.register_backward_hook(gradient_monitor_hook)
    
    def _recover_from_nan(self, layer_name="unknown"):
        """Automatic NaN recovery mechanism"""
        if self.nan_recovery_count >= self.max_nan_recoveries:
            print(f"🚨 MAX NaN RECOVERIES REACHED ({self.max_nan_recoveries})! Stopping automatic recovery.")
            return False
            
        self.nan_recovery_count += 1
        print(f"🔧 AUTOMATIC NaN RECOVERY #{self.nan_recovery_count} for {layer_name}")
        
        # Strategy 1: Reinitialize the problematic layer
        if layer_name == "conv1" and hasattr(self, 'conv1'):
            print("   Reinitializing conv1 weights...")
            nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
            print(f"   Conv1 reinitialized: range=[{self.conv1.weight.min():.6f}, {self.conv1.weight.max():.6f}]")
            
        # Strategy 2: Reset GroupNorm parameters
        if hasattr(self, 'gn1'):
            print("   Resetting GroupNorm1 parameters...")
            nn.init.constant_(self.gn1.weight, 1.0)
            nn.init.constant_(self.gn1.bias, 0.0)
            
        return True
    
    def _emergency_reset(self):
        """Emergency reset of the entire model to a safe state"""
        print("🚨 EMERGENCY MODEL RESET - Reinitializing all weights to safe values")
        
        # Reset conv1 with very conservative initialization
        with torch.no_grad():
            self.conv1.weight.data = torch.randn_like(self.conv1.weight) * 0.01
            print(f"   Conv1 emergency reset: range=[{self.conv1.weight.min():.6f}, {self.conv1.weight.max():.6f}]")
            
        # Reset GroupNorm
        if hasattr(self, 'gn1'):
            nn.init.constant_(self.gn1.weight, 1.0)
            nn.init.constant_(self.gn1.bias, 0.0)
            print("   GroupNorm1 emergency reset")
            
        # Reset all other layers conservatively
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d) and m != self.conv1:
                with torch.no_grad():
                    m.weight.data = torch.randn_like(m.weight) * 0.01
            elif isinstance(m, nn.Linear):
                with torch.no_grad():
                    m.weight.data = torch.randn_like(m.weight) * 0.01
                    if m.bias is not None:
                        m.bias.data.zero_()
                        
        print("   ✅ Emergency reset completed")
    
    def _make_layer(self, in_channels, out_channels, blocks, stride=1, use_attention=False, drop_rate=0.1):
        downsample = None
        if stride != 1 or in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),  # Temporarily removed spectral_norm
                nn.GroupNorm(min(16, max(2, out_channels // 2)), out_channels, eps=1e-5),  # More stable GroupNorm
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
        
        # Enhanced conv1 debugging with more detailed analysis
        print(f"🔍 CONV1 DETAILED ANALYSIS:")
        print(f"   Input: shape={x.shape}, dtype={x.dtype}, device={x.device}")
        print(f"   Input stats: min={x.min():.6f}, max={x.max():.6f}, mean={x.mean():.6f}, std={x.std():.6f}")
        print(f"   Input NaN/Inf: has_nan={torch.isnan(x).any()}, has_inf={torch.isinf(x).any()}")
        
        # Check conv1 weights in detail
        w = self.conv1.weight
        print(f"   Weight: shape={w.shape}, dtype={w.dtype}, device={w.device}")
        print(f"   Weight stats: min={w.min():.6f}, max={w.max():.6f}, mean={w.mean():.6f}, std={w.std():.6f}")
        print(f"   Weight NaN/Inf: has_nan={torch.isnan(w).any()}, has_inf={torch.isinf(w).any()}")
        
        # Check for weight corruption patterns
        if torch.isnan(w).any():
            print(f"   🚨 WEIGHT CORRUPTION: {torch.isnan(w).sum()} NaN values in conv1 weights!")
            nan_locations = torch.where(torch.isnan(w))
            print(f"   NaN locations: {[(i.item(), j.item(), k.item(), l.item()) for i, j, k, l in zip(*nan_locations)][:5]}...")
        
        # Perform convolution with error handling
        try:
            x_before_conv = x.clone()  # Keep copy for debugging
            x = self.conv1(x)
            
            print(f"   Conv1 output: shape={x.shape}, dtype={x.dtype}")
            print(f"   Output stats: min={x.min():.6f}, max={x.max():.6f}, mean={x.mean():.6f}, std={x.std():.6f}")
            print(f"   Output NaN/Inf: has_nan={torch.isnan(x).any()}, has_inf={torch.isinf(x).any()}")
            
            if torch.isnan(x).any():
                print(f"🚨 CONV1 NaN DETECTED!")
                print(f"   NaN count: {torch.isnan(x).sum()}/{x.numel()} ({100*torch.isnan(x).sum()/x.numel():.2f}%)")
                nan_mask = torch.isnan(x)
                print(f"   NaN pattern: {nan_mask.sum(dim=(2,3)).float().mean(dim=0)}")  # Average NaN per channel
                
                # Try automatic recovery
                if self._recover_from_nan("conv1"):
                    print("   Attempting recovery by re-running conv1...")
                    try:
                        x = self.conv1(x_before_conv)
                        if not torch.isnan(x).any():
                            print("   ✅ NaN recovery successful!")
                        else:
                            print("   ❌ NaN recovery failed, using zero replacement")
                            x = torch.nan_to_num(x, nan=0.0)
                    except:
                        print("   ❌ Recovery failed, using zero replacement")
                        x = torch.nan_to_num(x, nan=0.0)
                else:
                    # Replace NaN with zeros as fallback
                    x = torch.nan_to_num(x, nan=0.0)
                    print(f"   NaN values replaced with zeros")
                
        except Exception as e:
            print(f"🚨 CONV1 OPERATION FAILED: {e}")
            print(f"   Input was: shape={x_before_conv.shape}, range=[{x_before_conv.min():.6f}, {x_before_conv.max():.6f}]")
            raise e
        
        # Enhanced GroupNorm debugging
        print(f"🔍 GROUPNORM1 ANALYSIS:")
        print(f"   Input to GN1: shape={x.shape}, range=[{x.min():.6f}, {x.max():.6f}]")
        print(f"   GN1 config: channels={self.gn1.num_channels}, groups={self.gn1.num_groups}, eps={self.gn1.eps}")
        
        try:
            x_before_gn = x.clone()
            x = self.gn1(x)
            
            print(f"   GN1 output: shape={x.shape}, range=[{x.min():.6f}, {x.max():.6f}]")
            print(f"   GN1 NaN check: has_nan={torch.isnan(x).any()}")
            
            if torch.isnan(x).any():
                print(f"🚨 GROUPNORM1 NaN DETECTED!")
                print(f"   NaN count: {torch.isnan(x).sum()}/{x.numel()}")
                print(f"   Input to GN1 was: min={x_before_gn.min():.6f}, max={x_before_gn.max():.6f}")
                
                # Check if input has problematic values
                if (x_before_gn == 0).all():
                    print(f"   Problem: All input values are zero!")
                elif x_before_gn.std() < self.gn1.eps:
                    print(f"   Problem: Input std ({x_before_gn.std():.2e}) < eps ({self.gn1.eps:.2e})")
                
                x = torch.nan_to_num(x, nan=0.0)
                print(f"   NaN values replaced with zeros")
                
        except Exception as e:
            print(f"🚨 GROUPNORM1 OPERATION FAILED: {e}")
            raise e
        
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