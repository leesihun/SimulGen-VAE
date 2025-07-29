import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class LatentConditionerImg(nn.Module):
    """Simplified CNN-based latent conditioner for image data"""
    def __init__(self, latent_conditioner_filter, latent_dim_end, input_shape, latent_dim, size2, latent_conditioner_data_shape, dropout_rate=0.3):
        super(LatentConditionerImg, self).__init__()
        self.latent_dim = latent_dim
        self.size2 = size2
        self.latent_conditioner_filter = latent_conditioner_filter
        self.latent_dim_end = latent_dim_end
        self.input_shape = input_shape
        self.latent_conditioner_data_shape = latent_conditioner_data_shape
        
        # Simple CNN feature extractor
        self.conv_layers = nn.Sequential(
            # First conv layer
            nn.Conv2d(1, latent_conditioner_filter[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(latent_conditioner_filter[0]),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Second conv layer
            nn.Conv2d(latent_conditioner_filter[0], latent_conditioner_filter[1], kernel_size=3, padding=1),
            nn.BatchNorm2d(latent_conditioner_filter[1]),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Third conv layer
            nn.Conv2d(latent_conditioner_filter[1], latent_conditioner_filter[2], kernel_size=3, padding=1),
            nn.BatchNorm2d(latent_conditioner_filter[2]),
            nn.ReLU(),
            nn.MaxPool2d(2),
            
            # Global average pooling
            nn.AdaptiveAvgPool2d((1, 1))
        )
        
        # Output heads
        self.latent_head = nn.Linear(latent_conditioner_filter[2], latent_dim_end)
        self.xs_head = nn.Linear(latent_conditioner_filter[2], latent_dim * size2)

    def forward(self, x):
        # Reshape input to image format
        x = x.reshape([-1, 1, int(math.sqrt(x.shape[-1])), int(math.sqrt(x.shape[-1]))])
        
        # Extract features through CNN
        features = self.conv_layers(x)
        features = features.flatten(1)
        
        # Generate outputs
        latent_main = self.latent_head(features)
        xs_main = self.xs_head(features)
        xs_main = xs_main.unflatten(1, (self.size2, self.latent_dim))

        return latent_main, xs_main