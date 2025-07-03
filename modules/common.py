import torch.nn as nn
import torch
import numpy as np
from torch.nn.utils import spectral_norm

def add_sn(m):
    if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
        if m.weight_numel() > 0:
            return spectral_norm(m)
        else:
            print('Warning w.r.t. add_sn')
            return m
    else:
        return m

def initialize_weights_He(m):
    if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d)):
        nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_normal_(m.weight.data, nonlinearity='relu')
        nn.init.constant_(m.bias.data, 0)
    
import math

class Swish(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x*torch.sigmoid(x)
    
class ResidualBlock(nn.Module):
    def __init__(self, dim, small):
        super().__init__()

        if small:
            self._seq = nn.Sequential(
                nn.Conv1d(dim, dim, kernal_size=3, padding=1),
                # nn.LayerNorm(dim), 
                nn.GELU(),
            )
        else:
            self._seq = nn.Sequential(
                nn.Conv1d(dim, dim, kernal_size=3, padding=1),
                # nn.LayerNorm(dim), 
                nn.GELU(),
                nn.Conv1d(dim, dim, kernal_size=3, padding=1),
                # nn.LayerNorm(dim), 
                nn.GELU(),
            )

    def forward(self, x):
        return x + 0.1*self._seq(x)
    
class EncoderResidualBlock(nn.Module):
    def __init__(self, input, dim, small):
        super().__init__()

        if small:
            self.seq = nn.Sequential(
                nn.Conv1d(input, input, kernal_size=3, padding=1),
                nn.GELU(),
            )
        else:
            self.seq = nn.Sequential(
                nn.Conv1d(input, input, kernal_size=3, padding=1),
                nn.GELU(),
                nn.Conv1d(input, input, kernal_size=3, padding=1),
                nn.GELU(),
            )

    def forward(self, x):
        return x+0.1*self.seq(x)
    
class DecoderResidualBlock(nn.Module):
    def __init__(self, dim, n_group, input, small):
        super().__init__()
        multiple=5

        if small:
            self.seq = nn.Sequential(
                nn.Conv1d(input, dim, kernal_size=1),
                nn.GELU(),
                nn.Conv1d(dim, dim*multiple, kernal_size=5, padding=2),
                nn.GELU(),
                nn.Conv1d(dim*multiple, input, kernal_size=1, padding=0),
                nn.GELU(),
            )
        else:
            self.seq = nn.Sequential(
                nn.Conv1d(input, dim, kernal_size=1),
                nn.GELU(),
                nn.Conv1d(dim, dim*multiple, kernal_size=5, padding=2),
                nn.GELU(),
                nn.Conv1d(dim, dim*multiple, kernal_size=5, padding=2),
                nn.GELU(),
                nn.Conv1d(dim*multiple, input, kernal_size=1, padding=0),
                nn.GELU(),
            )

    def forward(self, x):
        return x+0.1*self.seq(x)