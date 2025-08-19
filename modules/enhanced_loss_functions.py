"""Enhanced Loss Functions for CNN Latent Conditioner

This module implements advanced loss functions and training enhancements specifically
designed for the CNN-based latent conditioner in SimulGenVAE.

Features:
- Multi-scale robust loss (MSE + MAE + Huber)
- Perceptual loss for latent space semantic understanding
- Consistency regularization across augmentations
- Adaptive loss component weighting
- Spectral regularization for feature stability

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
    Uses adaptive weighting based on relative loss magnitudes.
    """
    
    def __init__(self, mse_weight=1.0, mae_weight=0.1, huber_weight=0.05, 
                 huber_beta=0.1, adaptive_weighting=True):
        super().__init__()
        self.mse_weight = mse_weight
        self.mae_weight = mae_weight 
        self.huber_weight = huber_weight
        self.huber_beta = huber_beta
        self.adaptive_weighting = adaptive_weighting
        
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
        
        # Adaptive weighting based on relative magnitudes
        if self.adaptive_weighting:
            main_weight = 10.0  # Base weight for main latent
            # Adjust hierarchical weight based on relative loss ratio
            loss_ratio = loss_main.detach() / (loss_hier.detach() + 1e-8)
            hier_weight = max(1.0, min(5.0, loss_ratio * 0.5))
        else:
            main_weight = 10.0
            hier_weight = 1.0
        
        total_loss = main_weight * loss_main + hier_weight * loss_hier
        
        loss_info = {
            'loss_main': loss_main.item(),
            'loss_hier': loss_hier.item(),
            'mse_main': mse_main.item(),
            'mae_main': mae_main.item(), 
            'huber_main': huber_main.item(),
            'mse_hier': mse_hier.item(),
            'mae_hier': mae_hier.item(),
            'huber_hier': huber_hier.item(),
            'main_weight': main_weight,
            'hier_weight': hier_weight if isinstance(hier_weight, float) else hier_weight.item()
        }
        
        return total_loss, loss_info


class PerceptualLatentLoss(nn.Module):
    """Perceptual loss for latent space using feature similarity networks.
    
    Creates feature extractors that map latent codes to intermediate representations
    and computes similarity losses to ensure semantic consistency.
    """
    
    def __init__(self, main_latent_dim=32, hier_latent_dim=8, 
                 feature_layers=[16, 8], similarity_type='cosine'):
        super().__init__()
        self.similarity_type = similarity_type
        
        # Feature networks for main latent
        self.main_feature_networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(main_latent_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, main_latent_dim // 2)
            ) for hidden_dim in feature_layers
        ])
        
        # Feature networks for hierarchical latent
        self.hier_feature_networks = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hier_latent_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.SiLU(),
                nn.Linear(hidden_dim, hier_latent_dim // 2)
            ) for hidden_dim in feature_layers
        ])
        
        # Initialize with small weights to prevent dominance
        self._initialize_weights()
        
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight, gain=0.1)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
    
    def _compute_similarity_loss(self, pred_features, target_features):
        """Compute similarity loss between feature representations."""
        if self.similarity_type == 'cosine':
            # Cosine similarity loss
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
            target_features = feature_net(target_main.detach())  # Stop gradient for stability
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
            'perceptual_hier': perceptual_hier.item()
        }
        
        return total_perceptual_loss, loss_info


class ConsistencyLoss(nn.Module):
    """Consistency regularization ensuring predictions are stable across augmentations."""
    
    def __init__(self, consistency_weight=0.1, temperature=1.0):
        super().__init__()
        self.consistency_weight = consistency_weight
        self.temperature = temperature
        
    def forward(self, model, x_original, x_augmented):
        """Compute consistency loss between original and augmented inputs.
        
        Args:
            model: Latent conditioner model
            x_original: Original input images
            x_augmented: Augmented input images
            
        Returns:
            consistency_loss: Consistency regularization loss
            loss_info: Dictionary with component losses
        """
        # Get predictions for original input (detached to prevent gradient flow)
        with torch.no_grad():
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
            'consistency_hier': consistency_hier.item()
        }
        
        return total_consistency, loss_info


class AdaptiveLossWeighter(nn.Module):
    """Learnable adaptive weighting for multiple loss components."""
    
    def __init__(self, num_components=4, init_weights=None, min_weight=0.01):
        super().__init__()
        
        if init_weights is None:
            init_weights = torch.ones(num_components)
        else:
            init_weights = torch.tensor(init_weights)
            
        self.log_weights = nn.Parameter(torch.log(init_weights))
        self.min_weight = min_weight
        
    def forward(self, losses):
        """Compute weighted combination of losses with adaptive weights.
        
        Args:
            losses: List of loss tensors
            
        Returns:
            weighted_loss: Adaptively weighted total loss
            current_weights: Current weight values
        """
        # Compute normalized weights with minimum threshold
        raw_weights = F.softmax(self.log_weights, dim=0)
        weights = torch.clamp(raw_weights * len(losses), min=self.min_weight)
        
        # Weighted combination
        weighted_loss = sum(w * loss for w, loss in zip(weights, losses))
        
        return weighted_loss, weights


class SpectralFeatureRegularization(nn.Module):
    """Spectral regularization for intermediate feature maps."""
    
    def __init__(self, reg_weight=1e-4, max_singular_values=5):
        super().__init__()
        self.reg_weight = reg_weight
        self.max_singular_values = max_singular_values
        
    def forward(self, feature_maps):
        """Compute spectral regularization on feature maps.
        
        Args:
            feature_maps: List of feature tensors from network layers
            
        Returns:
            spectral_reg: Spectral regularization loss
        """
        if not feature_maps:
            return torch.tensor(0.0, device=next(self.parameters()).device)
            
        spectral_reg = 0
        
        for feature_map in feature_maps:
            # Reshape feature map for SVD
            batch_size = feature_map.size(0)
            feature_flat = feature_map.view(batch_size, -1)
            
            # Compute SVD (only top singular values for efficiency)
            try:
                U, S, V = torch.svd(feature_flat)
                # Penalize large singular values
                top_singular_values = S[:, :self.max_singular_values]
                spectral_reg += torch.sum(top_singular_values**2)
            except RuntimeError:
                # Handle SVD convergence issues
                spectral_reg += torch.sum(feature_flat**2) * 1e-6
                
        return self.reg_weight * spectral_reg / len(feature_maps)


class EnhancedLossConfig:
    """Configuration class for enhanced loss function settings."""
    
    def __init__(self, 
                 use_multiscale_loss=True,
                 use_perceptual_loss=True,
                 use_consistency_loss=True,
                 use_adaptive_weighting=False,  # Disabled by default for stability
                 use_spectral_regularization=True,
                 
                 # Multi-scale loss weights
                 mse_weight=1.0,
                 mae_weight=0.1,
                 huber_weight=0.05,
                 huber_beta=0.1,
                 
                 # Perceptual loss settings
                 perceptual_weight=0.1,
                 perceptual_feature_layers=[16, 8],
                 perceptual_similarity='cosine',
                 
                 # Consistency loss settings
                 consistency_weight=0.1,
                 consistency_temperature=1.0,
                 
                 # Spectral regularization settings
                 spectral_weight=1e-4,
                 spectral_max_sv=5,
                 
                 # Adaptive weighting settings
                 adaptive_min_weight=0.01):
        
        self.use_multiscale_loss = use_multiscale_loss
        self.use_perceptual_loss = use_perceptual_loss
        self.use_consistency_loss = use_consistency_loss
        self.use_adaptive_weighting = use_adaptive_weighting
        self.use_spectral_regularization = use_spectral_regularization
        
        self.mse_weight = mse_weight
        self.mae_weight = mae_weight
        self.huber_weight = huber_weight
        self.huber_beta = huber_beta
        
        self.perceptual_weight = perceptual_weight
        self.perceptual_feature_layers = perceptual_feature_layers
        self.perceptual_similarity = perceptual_similarity
        
        self.consistency_weight = consistency_weight
        self.consistency_temperature = consistency_temperature
        
        self.spectral_weight = spectral_weight
        self.spectral_max_sv = spectral_max_sv
        
        self.adaptive_min_weight = adaptive_min_weight
    
    @classmethod
    def small_dataset_config(cls):
        """Configuration optimized for small datasets (< 1000 images)."""
        return cls(
            use_multiscale_loss=True,
            use_perceptual_loss=True,
            use_consistency_loss=True,
            use_adaptive_weighting=False,
            use_spectral_regularization=True,
            
            perceptual_weight=0.2,
            consistency_weight=0.15,
            spectral_weight=1e-3
        )
    
    @classmethod
    def large_dataset_config(cls):
        """Configuration optimized for large datasets (> 5000 images)."""
        return cls(
            use_multiscale_loss=True,
            use_perceptual_loss=True,
            use_consistency_loss=True,
            use_adaptive_weighting=True,
            use_spectral_regularization=True,
            
            perceptual_weight=0.1,
            consistency_weight=0.1,
            spectral_weight=1e-4
        )
    
    @classmethod
    def fast_training_config(cls):
        """Configuration optimized for faster training with minimal enhancements."""
        return cls(
            use_multiscale_loss=True,
            use_perceptual_loss=False,
            use_consistency_loss=False,
            use_adaptive_weighting=False,
            use_spectral_regularization=True,
            
            spectral_weight=1e-5
        )


def create_enhanced_loss_system(config, main_latent_dim=32, hier_latent_dim=8, size2=200):
    """Factory function to create complete enhanced loss system.
    
    Args:
        config: EnhancedLossConfig instance
        main_latent_dim: Dimension of main latent space
        hier_latent_dim: Dimension of hierarchical latent space
        size2: Size of second dimension for hierarchical latent
        
    Returns:
        Dictionary containing all loss components and config
    """
    loss_system = {'config': config}
    
    # Multi-scale robust loss
    if config.use_multiscale_loss:
        loss_system['multiscale_loss'] = MultiScaleRobustLoss(
            mse_weight=config.mse_weight,
            mae_weight=config.mae_weight,
            huber_weight=config.huber_weight,
            huber_beta=config.huber_beta,
            adaptive_weighting=True
        )
    
    # Perceptual loss
    if config.use_perceptual_loss:
        loss_system['perceptual_loss'] = PerceptualLatentLoss(
            main_latent_dim=main_latent_dim,
            hier_latent_dim=hier_latent_dim,
            feature_layers=config.perceptual_feature_layers,
            similarity_type=config.perceptual_similarity
        )
    
    # Consistency loss
    if config.use_consistency_loss:
        loss_system['consistency_loss'] = ConsistencyLoss(
            consistency_weight=config.consistency_weight,
            temperature=config.consistency_temperature
        )
    
    # Spectral regularization
    if config.use_spectral_regularization:
        loss_system['spectral_regularization'] = SpectralFeatureRegularization(
            reg_weight=config.spectral_weight,
            max_singular_values=config.spectral_max_sv
        )
    
    # Adaptive weighting
    if config.use_adaptive_weighting:
        # Count active loss components
        num_components = sum([
            config.use_multiscale_loss,
            config.use_perceptual_loss, 
            config.use_consistency_loss,
            config.use_spectral_regularization
        ])
        loss_system['adaptive_weighter'] = AdaptiveLossWeighter(
            num_components=num_components,
            min_weight=config.adaptive_min_weight
        )
    
    return loss_system