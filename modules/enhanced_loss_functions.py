import torch
import torch.nn as nn
import torch.nn.functional as F

def compute_enhanced_loss(y_pred1, y_pred2, y1, y2, config):
    """Simple enhanced loss computation with configurable weights."""
    
    # Basic MSE losses
    mse_main = F.mse_loss(y_pred1, y1)
    mse_hier = F.mse_loss(y_pred2, y2)
    
    # Multi-scale loss components
    mae_main = F.l1_loss(y_pred1, y1) if config.get('mae_weight', 0) > 0 else 0
    mae_hier = F.l1_loss(y_pred2, y2) if config.get('mae_weight', 0) > 0 else 0
    
    huber_main = F.smooth_l1_loss(y_pred1, y1, beta=config.get('huber_beta', 0.1)) if config.get('huber_weight', 0) > 0 else 0
    huber_hier = F.smooth_l1_loss(y_pred2, y2, beta=config.get('huber_beta', 0.1)) if config.get('huber_weight', 0) > 0 else 0
    
    # Combine losses for each output
    loss_main = (config.get('mse_weight', 0.2) * mse_main + 
                 config.get('mae_weight', 0.3) * mae_main + 
                 config.get('huber_weight', 0.5) * huber_main)
    
    loss_hier = (config.get('mse_weight', 0.2) * mse_hier + 
                 config.get('mae_weight', 0.3) * mae_hier + 
                 config.get('huber_weight', 0.5) * huber_hier)
    
    # Apply standard weighting (main=10x, hier=1x)
    total_loss = config.get('main_weight', 0.9) * loss_main + config.get('hier_weight', 0.1) * loss_hier
    
    return total_loss

def compute_perceptual_loss(y_pred1, y_pred2, y1, y2, config):
    """Simple perceptual loss using cosine similarity."""
    if config.get('perceptual_weight', 0) <= 0:
        return 0
    
    # Cosine similarity loss for main latent
    cosine_main = F.cosine_similarity(y_pred1, y1, dim=1).mean()
    perceptual_main = 1 - cosine_main
    
    # Cosine similarity loss for hierarchical latent (flatten timesteps)
    batch_size, size2, hier_dim = y_pred2.shape
    y_pred2_flat = y_pred2.view(-1, hier_dim)
    y2_flat = y2.view(-1, hier_dim)
    cosine_hier = F.cosine_similarity(y_pred2_flat, y2_flat, dim=1).mean()
    perceptual_hier = 1 - cosine_hier
    
    perceptual_loss = (perceptual_main + perceptual_hier) * config.get('perceptual_weight', 0.1)
    return perceptual_loss

# def compute_consistency_loss(model, x_original, x_augmented, config):
#     """Simple consistency loss between original and augmented inputs."""
#     if config.get('consistency_weight', 0) <= 0:
#         return 0
    
#     # Get predictions for both inputs
#     with torch.no_grad():
#         y1_orig, y2_orig = model(x_original)
#     y1_aug, y2_aug = model(x_augmented)
    
#     # MSE between predictions
#     consistency_main = F.mse_loss(y1_aug, y1_orig)
#     consistency_hier = F.mse_loss(y2_aug, y2_orig)
    
#     consistency_loss = (consistency_main + consistency_hier) * config.get('consistency_weight', 0.1)
#     return consistency_loss