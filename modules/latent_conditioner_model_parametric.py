import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentConditioner(nn.Module):
    """MLP-based latent conditioner for parametric data"""
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
        for i in range(1, self.num_latent_conditioner_filter-1):
            modules.append(nn.Linear(self.latent_conditioner_filter[i-1], self.latent_conditioner_filter[i]))
            modules.append(nn.LeakyReLU(0.2))
            modules.append(nn.Dropout(0.1))
        self.latent_conditioner = nn.Sequential(*modules)

        # Simplified output heads
        final_feature_size = self.latent_conditioner_filter[-2]
        
        # ULTRA-EXTREME bottleneck with single output heads
        hidden_size = max(8, final_feature_size // 32)
        
        # Single prediction head for latent output
        self.latent_out = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(final_feature_size, hidden_size),
            nn.GELU(),
            nn.Dropout(0.15), 
            nn.Linear(hidden_size, self.latent_dim_end),
            nn.Tanh()
        )
        
        # Single prediction head for xs output
        self.xs_out = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(final_feature_size, hidden_size),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_size, self.latent_dim * self.size2),
            nn.Tanh()
        )

    def forward(self, x):
        features = self.latent_conditioner(x)
        
        # Direct prediction from single heads
        latent_out = self.latent_out(features)
        xs_out = self.xs_out(features)
        xs_out = xs_out.unflatten(1, (self.size2, self.latent_dim))

        return latent_out, xs_out