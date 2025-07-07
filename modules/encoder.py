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
        xs = []

        B, _, _ = x.shape
        i=0

        for encoder_block, residual_block in zip(self.encoder_blocks, self.encoder_residual_blocks):

            x=encoder_block(x)
            x=residual_block(x)
            last_x = x

            xs_reshaped = last_x.view(B, -1)
            xs_reshaped = self.xs_linear[i](xs_reshaped)
            xs.append(xs_reshaped)
            i=i+1

        last_x = last_x.view(B, -1)
        last_x = self.last_x_linear(last_x)
        mu = last_x[:, :self.z_dim]
        log_var = last_x[:, self.z_dim:]

        return mu, log_var, xs[:-1][::-1]