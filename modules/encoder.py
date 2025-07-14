import torch.nn as nn
from modules.common import *

class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel, small):
        super().__init__()

        if small:
            self._seq = nn.Sequential(
                nn.Conv1d(in_channel, out_channel, kernel_size=1, padding=0),
                nn.BatchNorm1d(out_channel),
                nn.GELU(),
            )
        else:
            self._seq = nn.Sequential(
                nn.Conv1d(in_channel, out_channel, kernel_size=1, padding=0),
                nn.BatchNorm1d(out_channel),
                nn.GELU(),
                nn.Conv1d(out_channel, out_channel, kernel_size=3, padding=1),
                nn.BatchNorm1d(out_channel),
                nn.GELU(),
            )
    
    def forward(self, x):
        return self._seq(x)

class EncoderBlock(nn.Module):
    def __init__(self, channels, small):
        super().__init__()
        self.channels = channels
        modules = []

        for i in range(len(channels)-1):
            modules.append(ConvBlock(channels[i], channels[i+1], small))
        self.module_list = nn.ModuleList(modules)

    def forward(self, x):
        for module in self.module_list:
            x = module(x)
        return x

class Encoder(nn.Module):
    def __init__(self, z_dim, hierarchical_dim, num_filter_enc, num_node, num_time, small):
        super().__init__()

        self.encoder_blocks = nn.ModuleList([
            EncoderBlock([num_node, num_filter_enc[0]], small),
        ])

        for i in range(0, len(num_filter_enc)-1):
            self.encoder_blocks.append(EncoderBlock([num_filter_enc[i], num_filter_enc[i+1]], small))

        self.encoder_residual_blocks = nn.ModuleList([])

        for i in range(0, len(num_filter_enc)):
            if i==len(num_filter_enc):
                self.encoder_residual_blocks.append(EncoderResidualBlock(num_filter_enc[i], num_filter_enc[i], small))
            else:
                self.encoder_residual_blocks.append(EncoderResidualBlock(num_filter_enc[i], num_filter_enc[i], small))

        self.z_dim = z_dim
        self.num_filter_enc = num_filter_enc
        hierarchical_dim = hierarchical_dim

        self.xs_linear = nn.ModuleList([])
        for i in range(len(num_filter_enc)):
            self.xs_linear.append(nn.Linear(num_filter_enc[i]*num_time,int(hierarchical_dim)))

        self.last_x_linear = nn.Linear(num_filter_enc[-1]*num_time, 2*z_dim)

        self.small = small

    def forward(self, x):
        # Check input data quality
        if torch.isnan(x).any() or torch.isinf(x).any():
            print(f"Warning: NaN/Inf detected in encoder input data")
            print(f"Input shape: {x.shape}")
            print(f"Input stats: min={x.min().item():.4f}, max={x.max().item():.4f}")
            x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        xs = []

        B, _, _ = x.shape
        i=0

        for encoder_block, residual_block in zip(self.encoder_blocks, self.encoder_residual_blocks):

            x=encoder_block(x)
            
            # Check for NaN after encoder block
            if torch.isnan(x).any() or torch.isinf(x).any():
                print(f"Warning: NaN/Inf detected after encoder block {i}")
                x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
            
            x=residual_block(x)
            
            # Check for NaN after residual block
            if torch.isnan(x).any() or torch.isinf(x).any():
                print(f"Warning: NaN/Inf detected after residual block {i}")
                x = torch.nan_to_num(x, nan=0.0, posinf=1.0, neginf=-1.0)
            
            last_x = x

            xs_reshaped = last_x.view(B, -1)
            
            # Check for NaN/Inf before hierarchical linear layer
            if torch.isnan(xs_reshaped).any() or torch.isinf(xs_reshaped).any():
                print(f"Warning: NaN/Inf in hierarchical input at block {i}")
                print(f"Stats: min={xs_reshaped.min().item():.4f}, max={xs_reshaped.max().item():.4f}")
                xs_reshaped = torch.nan_to_num(xs_reshaped, nan=0.0, posinf=1.0, neginf=-1.0)
            
            xs_reshaped = self.xs_linear[i](xs_reshaped)
            
            # Check hierarchical linear output
            if torch.isnan(xs_reshaped).any() or torch.isinf(xs_reshaped).any():
                print(f"Warning: NaN/Inf in hierarchical linear output at block {i}")
                xs_reshaped = torch.nan_to_num(xs_reshaped, nan=0.0, posinf=0.1, neginf=-0.1)
            
            xs.append(xs_reshaped)
            i=i+1

        last_x = last_x.view(B, -1)
        
        # Check for NaN in encoder input to final linear layer
        if torch.isnan(last_x).any() or torch.isinf(last_x).any():
            print(f"Warning: NaN/Inf detected in encoder before final linear layer")
            print(f"Input stats: min={last_x.min().item():.4f}, max={last_x.max().item():.4f}")
            last_x = torch.nan_to_num(last_x, nan=0.0, posinf=1.0, neginf=-1.0)
        
        last_x = self.last_x_linear(last_x)
        
        # Check and fix NaN in linear layer output
        if torch.isnan(last_x).any() or torch.isinf(last_x).any():
            print(f"Warning: NaN/Inf detected in encoder final linear output")
            last_x = torch.nan_to_num(last_x, nan=0.0, posinf=0.1, neginf=-0.1)
        
        mu = last_x[:, :self.z_dim]
        log_var = last_x[:, self.z_dim:]
        
        # Only clamp log_var to prevent extreme variance, allow mu more freedom
        log_var = torch.clamp(log_var, min=-10, max=5)  # Prevent extreme variance
        
        # Final NaN check
        if torch.isnan(mu).any() or torch.isnan(log_var).any():
            print(f"Warning: NaN in encoder outputs after processing")
            mu = torch.nan_to_num(mu, nan=0.0)
            log_var = torch.nan_to_num(log_var, nan=-2.0)  # log(0.135) ≈ -2

        return mu, log_var, xs[:-1][::-1]