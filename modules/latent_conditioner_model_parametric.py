"""Parametric Latent Conditioner Model

Implements MLP-based latent conditioning for parametric/tabular data in SimulGenVAE.
This model processes numerical features to predict latent space representations
for both main and hierarchical latent variables.

Features:
- Multi-layer perceptron with progressive dropout
- Extreme bottleneck compression for regularization
- Dual output heads for latent and hierarchical predictions
- GELU activations for smooth gradients

Author: SiHun Lee, Ph.D.
Email: kevin1007kr@gmail.com
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentConditioner(nn.Module):
    """MLP-based latent conditioner for parametric data.
    
    Processes tabular/parametric input data to predict latent space representations.
    Uses a multi-layer perceptron with progressive dropout and extreme bottleneck
    compression to prevent overfitting on small datasets.
    
    Args:
        latent_conditioner_filter (list): Number of neurons in each hidden layer
        latent_dim_end (int): Dimension of main latent space (typically 32)
        input_shape (int): Number of input features/parameters
        latent_dim (int): Dimension of hierarchical latent space (typically 8)
        size2 (int): Size multiplier for hierarchical latent output
        dropout_rate (float): Dropout rate for regularization (default: 0.3)
    
    Attributes:
        latent_conditioner (nn.Sequential): Main feature extraction backbone
        latent_out (nn.Sequential): Output head for main latent predictions
        xs_out (nn.Sequential): Output head for hierarchical latent predictions
    
    Note:
        Uses extreme bottleneck compression (32:1 ratio) and progressive dropout
        (20% -> 15%) to prevent overfitting on parametric data.
    """
    def __init__(self, latent_conditioner_filter, latent_dim_end, input_shape, latent_dim, size2, dropout_rate=0.3):
        super(LatentConditioner, self).__init__()
        self.latent_dim = latent_dim
        self.size2 = size2
        self.latent_conditioner_filter = latent_conditioner_filter
        self.latent_dim_end = latent_dim_end
        self.input_shape = input_shape
        self.num_latent_conditioner_filter = len(self.latent_conditioner_filter)
        self.dropout_rate = dropout_rate

        # Backbone feature extractor
        modules = []
        modules.append(nn.Linear(self.input_shape, self.latent_conditioner_filter[0]))
        modules.append(nn.GELU())
        modules.append(nn.Dropout(0.05))  # Reduced dropout
        
        for i in range(1, self.num_latent_conditioner_filter):
            modules.append(nn.Linear(self.latent_conditioner_filter[i-1], self.latent_conditioner_filter[i]))
            modules.append(nn.GELU())
            modules.append(nn.Dropout(0.05))  # Reduced dropout
        
        # Add final activation after backbone
        modules.append(nn.GELU())
        self.latent_conditioner = nn.Sequential(*modules)

        # Simplified output heads
        final_feature_size = self.latent_conditioner_filter[-1]
        
        # Less severe bottleneck - use quarter of backbone output instead of tiny bottleneck
        hidden_size = max(64, final_feature_size // 4)  # 512 // 4 = 128, much better than 64
        
        # Single prediction head for latent output
        self.latent_out = nn.Sequential(
            nn.Dropout(0.1),  # Reduced dropout
            nn.Linear(final_feature_size, hidden_size),
            nn.GELU(),
            nn.Dropout(0.05),  # Reduced dropout
            nn.Linear(hidden_size, self.latent_dim_end),
            nn.Tanh()
        )
        
        # Single prediction head for xs output
        self.xs_out = nn.Sequential(
            nn.Dropout(0.1),  # Reduced dropout
            nn.Linear(final_feature_size, hidden_size),
            nn.GELU(),
            nn.Dropout(0.05),  # Reduced dropout
            nn.Linear(hidden_size, self.latent_dim * self.size2),
            nn.Tanh()
        )

    def forward(self, x):
        features = self.latent_conditioner(x)
        
        # Direct prediction from single heads
        latent_out = self.latent_out(features)
        xs_out = self.xs_out(features)
        xs_out = xs_out.view(xs_out.size(0), self.size2, self.latent_dim)

        return latent_out, xs_out