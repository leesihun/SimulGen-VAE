"""Enhanced Loss Functions for CNN Latent Conditioner

This module implements advanced loss functions specifically designed for the 
CNN-based latent conditioner in SimulGenVAE. Provides three core enhancements:

1. Multi-Scale Robust Loss: Combines MSE, MAE, and Huber for outlier robustness
2. Perceptual Loss: Semantic understanding through feature-level comparison  
3. Consistency Regularization: Stable predictions across augmentations

Author: SiHun Lee, Ph.D.
Email: kevin1007kr@gmail.com
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math


class MultiScaleRobustLoss(nn.Module):
    """Multi-scale robust loss combining MSE, MAE, and Huber losses.
    
    Provides robustness to outliers while maintaining sensitivity to small errors.
    Replaces simple MSE with a more stable loss formulation.
    
    Args:
        mse_weight (float): Weight for MSE loss component (default: 1.0)
        mae_weight (float): Weight for MAE loss component (default: 0.1)  
        huber_weight (float): Weight for Huber loss component (default: 0.05)
        huber_beta (float): Huber loss transition point (default: 0.1)
        main_weight (float): Weight multiplier for main latent (default: 10.0)
        hier_weight (float): Weight multiplier for hierarchical latent (default: 1.0)
    """
    
    def __init__(self, mse_weight=0.4, mae_weight=0.1, huber_weight=0.5, 
                 huber_beta=0.1, main_weight=10.0, hier_weight=1.0):
        super().__init__()
        self.mse_weight = mse_weight
        self.mae_weight = mae_weight 
        self.huber_weight = huber_weight
        self.huber_beta = huber_beta
        self.main_weight = main_weight
        self.hier_weight = hier_weight
        
    def forward(self, pred_main, pred_hier, target_main, target_hier):
        """Compute multi-scale robust loss for dual outputs.
        
        Args:
            pred_main: Predicted main latent (batch_size, latent_dim_end)
            pred_hier: Predicted hierarchical latent (batch_size, size2, latent_dim)
            target_main: Target main latent
            target_hier: Target hierarchical latent
            
        Returns:
            total_loss: Combined loss value
            loss_info: Dictionary with component losses and weights
        """
        # Main latent loss components
        mse_main = F.mse_loss(pred_main, target_main)
        mae_main = F.l1_loss(pred_main, target_main)
        huber_main = F.smooth_l1_loss(pred_main, target_main, beta=self.huber_beta)
        
        loss_main = (self.mse_weight * mse_main + 
                     self.mae_weight * mae_main + 
                     self.huber_weight * huber_main)
        
        # Hierarchical latent loss components  
        mse_hier = F.mse_loss(pred_hier, target_hier)
        mae_hier = F.l1_loss(pred_hier, target_hier)
        huber_hier = F.smooth_l1_loss(pred_hier, target_hier, beta=self.huber_beta)
        
        loss_hier = (self.mse_weight * mse_hier + 
                     self.mae_weight * mae_hier + 
                     self.huber_weight * huber_hier)
        
        # Apply standard weighting (main=10x, hier=1x)
        total_loss = self.main_weight * loss_main + self.hier_weight * loss_hier
        
        loss_info = {
            'loss_main': loss_main.item(),
            'loss_hier': loss_hier.item(),
            'mse_main': mse_main.item(),
            'mae_main': mae_main.item(), 
            'huber_main': huber_main.item(),
            'mse_hier': mse_hier.item(),
            'mae_hier': mae_hier.item(),
            'huber_hier': huber_hier.item(),
            'main_weight': self.main_weight,
            'hier_weight': self.hier_weight
        }
        
        return total_loss, loss_info


class PerceptualLatentLoss(nn.Module):
    """Perceptual loss for latent space using feature similarity networks.
    
    Creates small feature extractors that map latent codes to intermediate 
    representations and computes similarity losses to ensure semantic consistency.
    
    Args:
        main_latent_dim (int): Dimension of main latent space (default: 32)
        hier_latent_dim (int): Dimension of hierarchical latent space (default: 8)
        feature_layers (list): Hidden dimensions for feature networks (default: [16, 8])
        similarity_type (str): Type of similarity ('cosine' or 'mse') (default: 'cosine')
    """
    
    def __init__(self, main_latent_dim=32, hier_latent_dim=8, 
                 feature_layers=[16, 8], similarity_type='cosine'):
        super().__init__()
        self.similarity_type = similarity_type
        
        # Feature networks for main latent (32D → features)
        self.main_feature_networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(main_latent_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, main_latent_dim // 2)
            ) for hidden_dim in feature_layers
        ])
        
        # Feature networks for hierarchical latent (8D → features)
        self.hier_feature_networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hier_latent_dim, max(4, hidden_dim // 2)),  # Scale down for smaller dim
                nn.SiLU(),
                nn.Linear(max(4, hidden_dim // 2), max(4, hidden_dim // 2)),
                nn.SiLU(),
                nn.Linear(max(4, hidden_dim // 2), hier_latent_dim // 2)
            ) for hidden_dim in feature_layers
        ])
        
        # Initialize with small weights to prevent dominance
        self._initialize_weights()
        
    def _initialize_weights(self):
        """Initialize networks with conservative weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def _compute_similarity_loss(self, pred_features, target_features):
        """Compute similarity loss between feature representations."""
        if self.similarity_type == 'cosine':
            # Cosine similarity loss (1 - similarity)
            cosine_sim = F.cosine_similarity(pred_features, target_features, dim=1)
            return (1 - cosine_sim.mean())
        elif self.similarity_type == 'mse':
            return F.mse_loss(pred_features, target_features)
        else:
            raise ValueError(f"Unknown similarity type: {self.similarity_type}")
    
    def forward(self, pred_main, pred_hier, target_main, target_hier):
        """Compute perceptual loss for both latent outputs.
        
        Args:
            pred_main: Predicted main latent (batch_size, main_latent_dim)
            pred_hier: Predicted hierarchical latent (batch_size, size2, hier_latent_dim) 
            target_main: Target main latent
            target_hier: Target hierarchical latent
            
        Returns:
            perceptual_loss: Combined perceptual loss
            loss_info: Dictionary with component losses
        """
        perceptual_main = 0
        perceptual_hier = 0
        
        # Main latent perceptual loss
        for feature_net in self.main_feature_networks:
            pred_features = feature_net(pred_main)
            # Stop gradient on target for stability
            target_features = feature_net(target_main.detach())  
            perceptual_main += self._compute_similarity_loss(pred_features, target_features)
        
        perceptual_main /= len(self.main_feature_networks)
        
        # Hierarchical latent perceptual loss (process each timestep)
        batch_size, size2, hier_dim = pred_hier.shape
        pred_hier_flat = pred_hier.view(-1, hier_dim)
        target_hier_flat = target_hier.view(-1, hier_dim)
        
        for feature_net in self.hier_feature_networks:
            pred_features = feature_net(pred_hier_flat)
            target_features = feature_net(target_hier_flat.detach())
            perceptual_hier += self._compute_similarity_loss(pred_features, target_features)
        
        perceptual_hier /= len(self.hier_feature_networks)
        
        total_perceptual_loss = perceptual_main + perceptual_hier
        
        loss_info = {
            'perceptual_main': perceptual_main.item(),
            'perceptual_hier': perceptual_hier.item(),
            'perceptual_total': total_perceptual_loss.item()
        }
        
        return total_perceptual_loss, loss_info


class ConsistencyLoss(nn.Module):
    """Consistency regularization ensuring predictions are stable across augmentations.
    
    Compares model predictions on original vs augmented inputs to enforce 
    robustness to small input variations.
    
    Args:
        consistency_weight (float): Weight for consistency loss (default: 0.1)
        temperature (float): Temperature scaling for smooth gradients (default: 1.0)
        detach_original (bool): Whether to detach original predictions (default: True)
    """
    
    def __init__(self, consistency_weight=0.1, temperature=1.0, detach_original=True):
        super().__init__()
        self.consistency_weight = consistency_weight
        self.temperature = temperature
        self.detach_original = detach_original
        
    def forward(self, model, x_original, x_augmented):
        """Compute consistency loss between original and augmented inputs.
        
        Args:
            model: Latent conditioner model
            x_original: Original input images (batch_size, features)
            x_augmented: Augmented input images (batch_size, features)
            
        Returns:
            consistency_loss: Consistency regularization loss
            loss_info: Dictionary with component losses
        """
        # Get predictions for original input 
        if self.detach_original:
            # Detach to prevent gradient loops and stabilize training
            with torch.no_grad():
                y1_orig, y2_orig = model(x_original)
        else:
            y1_orig, y2_orig = model(x_original)
        
        # Get predictions for augmented input
        y1_aug, y2_aug = model(x_augmented)
        
        # Compute consistency losses
        consistency_main = F.mse_loss(y1_aug, y1_orig)
        consistency_hier = F.mse_loss(y2_aug, y2_orig)
        
        # Temperature scaling for smoother gradients
        if self.temperature != 1.0:
            consistency_main = consistency_main / self.temperature
            consistency_hier = consistency_hier / self.temperature
        
        total_consistency = self.consistency_weight * (consistency_main + consistency_hier)
        
        loss_info = {
            'consistency_main': consistency_main.item(),
            'consistency_hier': consistency_hier.item(),
            'consistency_total': total_consistency.item(),
            'consistency_weight': self.consistency_weight
        }
        
        return total_consistency, loss_info


class EnhancedLossConfig:
    """Configuration class for enhanced loss function settings.
    
    Provides preset configurations for different training scenarios and
    centralizes all hyperparameter settings for enhanced loss functions.
    """
    
    def __init__(self, 
                 # Feature toggles
                 use_multiscale_loss=True,
                 use_perceptual_loss=True,
                 use_consistency_loss=True,
                 
                 # Multi-scale loss settings
                 mse_weight=1.0,
                 mae_weight=0.1,
                 huber_weight=0.05,
                 huber_beta=0.1,
                 main_weight=10.0,  # Standard 10:1 ratio
                 hier_weight=1.0,
                 
                 # Perceptual loss settings
                 perceptual_weight=0.1,
                 perceptual_feature_layers=[16, 8],
                 perceptual_similarity='cosine',
                 
                 # Consistency loss settings
                 consistency_weight=0.1,
                 consistency_temperature=1.0,
                 consistency_detach_original=True):
        
        # Feature toggles
        self.use_multiscale_loss = use_multiscale_loss
        self.use_perceptual_loss = use_perceptual_loss
        self.use_consistency_loss = use_consistency_loss
        
        # Multi-scale loss parameters
        self.mse_weight = mse_weight
        self.mae_weight = mae_weight
        self.huber_weight = huber_weight
        self.huber_beta = huber_beta
        self.main_weight = main_weight
        self.hier_weight = hier_weight
        
        # Perceptual loss parameters
        self.perceptual_weight = perceptual_weight
        self.perceptual_feature_layers = perceptual_feature_layers
        self.perceptual_similarity = perceptual_similarity
        
        # Consistency loss parameters
        self.consistency_weight = consistency_weight
        self.consistency_temperature = consistency_temperature
        self.consistency_detach_original = consistency_detach_original
    
    @classmethod
    def balanced_config(cls):
        """Balanced configuration for general use (recommended default)."""
        return cls(
            use_multiscale_loss=True,
            use_perceptual_loss=True,
            use_consistency_loss=True,
            perceptual_weight=0.1,
            consistency_weight=0.1
        )
    
    @classmethod
    def robust_config(cls):
        """Configuration emphasizing robustness and stability."""
        return cls(
            use_multiscale_loss=True,
            use_perceptual_loss=True,
            use_consistency_loss=True,
            mae_weight=0.2,        # Higher MAE weight for outlier robustness
            huber_weight=0.1,      # Higher Huber weight
            perceptual_weight=0.05, # Lower perceptual to avoid instability
            consistency_weight=0.15 # Higher consistency for stability
        )
    
    @classmethod
    def semantic_config(cls):
        """Configuration emphasizing semantic understanding."""
        return cls(
            use_multiscale_loss=True,
            use_perceptual_loss=True,
            use_consistency_loss=True,
            perceptual_weight=0.2,  # Higher perceptual weight
            perceptual_feature_layers=[32, 16, 8],  # More feature networks
            consistency_weight=0.05  # Lower consistency to not interfere
        )
    
    @classmethod
    def fast_config(cls):
        """Configuration optimized for faster training."""
        return cls(
            use_multiscale_loss=True,
            use_perceptual_loss=False,  # Disable expensive perceptual loss
            use_consistency_loss=False, # Disable expensive consistency loss
            mae_weight=0.05,            # Minimal MAE
            huber_weight=0.02           # Minimal Huber
        )
    
    @classmethod
    def small_dataset_config(cls):
        """Configuration optimized for small datasets (< 1000 images)."""
        return cls(
            use_multiscale_loss=True,
            use_perceptual_loss=True,
            use_consistency_loss=True,
            perceptual_weight=0.2,      # Higher for better feature learning
            consistency_weight=0.15,    # Higher for generalization
            perceptual_feature_layers=[8, 4]  # Smaller networks to prevent overfitting
        )


def create_enhanced_loss_system(config, main_latent_dim=32, hier_latent_dim=8, device='cuda'):
    """Factory function to create complete enhanced loss system.
    
    Args:
        config: EnhancedLossConfig instance
        main_latent_dim: Dimension of main latent space (default: 32)
        hier_latent_dim: Dimension of hierarchical latent space (default: 8)
        device: Device to place loss networks on (default: 'cuda')
        
    Returns:
        Dictionary containing all active loss components
    """
    loss_system = {
        'config': config,
        'active_components': []
    }
    
    # Multi-scale robust loss (always lightweight, usually enabled)
    if config.use_multiscale_loss:
        loss_system['multiscale_loss'] = MultiScaleRobustLoss(
            mse_weight=config.mse_weight,
            mae_weight=config.mae_weight,
            huber_weight=config.huber_weight,
            huber_beta=config.huber_beta,
            main_weight=config.main_weight,
            hier_weight=config.hier_weight
        )
        loss_system['active_components'].append('multiscale_loss')
    
    # Perceptual loss (computational overhead, optional)
    if config.use_perceptual_loss:
        perceptual_loss = PerceptualLatentLoss(
            main_latent_dim=main_latent_dim,
            hier_latent_dim=hier_latent_dim,
            feature_layers=config.perceptual_feature_layers,
            similarity_type=config.perceptual_similarity
        )
        # Move to appropriate device
        perceptual_loss = perceptual_loss.to(device)
        loss_system['perceptual_loss'] = perceptual_loss
        loss_system['active_components'].append('perceptual_loss')
    
    # Consistency loss (high computational overhead, optional)
    if config.use_consistency_loss:
        loss_system['consistency_loss'] = ConsistencyLoss(
            consistency_weight=config.consistency_weight,
            temperature=config.consistency_temperature,
            detach_original=config.consistency_detach_original
        )
        loss_system['active_components'].append('consistency_loss')
    
    return loss_system


def get_preset_config(preset_name):
    """Get a preset configuration by name.
    
    Args:
        preset_name (str): Name of preset ('balanced', 'robust', 'semantic', 'fast', 'small_dataset')
        
    Returns:
        EnhancedLossConfig instance
    """
    preset_map = {
        'balanced': EnhancedLossConfig.balanced_config,
        'robust': EnhancedLossConfig.robust_config,
        'semantic': EnhancedLossConfig.semantic_config,
        'fast': EnhancedLossConfig.fast_config,
        'small_dataset': EnhancedLossConfig.small_dataset_config
    }
    
    if preset_name not in preset_map:
        raise ValueError(f"Unknown preset: {preset_name}. Available: {list(preset_map.keys())}")
    
    return preset_map[preset_name]()