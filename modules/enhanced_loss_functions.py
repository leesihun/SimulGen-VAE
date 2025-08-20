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