"""
Improved End-to-End Latent Conditioner Training with Adaptive Scheduling

This module addresses loss stagnation issues by implementing:
1. Adaptive learning rate scheduling
2. Improved regularization scheduling  
3. Gradient health monitoring
4. Loss stagnation detection and recovery
"""

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import cv2
import os
import pandas as pd
import natsort
import time
import math
import psutil
import gc
import threading
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
from modules.pca_preprocessor import PCAPreprocessor
import matplotlib.pyplot as plt
import pickle
from joblib import load

# Import original utilities
def load_vae_model(vae_model_path, device):
    """Load and prepare VAE model for end-to-end training."""
    try:
        # Try loading complete model first
        if os.path.exists(vae_model_path):
            vae_model = torch.load(vae_model_path, map_location=device, weights_only=False)
            print(f"Loaded VAE model from {vae_model_path}")
        else:
            # output error message
            print(f"VAE model not found at {vae_model_path}")
            raise FileNotFoundError(f"VAE model not found at {vae_model_path}")
        
        # Freeze VAE weights if it's a real model
        if hasattr(vae_model, 'parameters'):
            for param in vae_model.parameters():
                param.requires_grad = False
            vae_model.eval()
            print("VAE model weights frozen for end-to-end training")
        
        return vae_model
        
    except Exception as e:
        print(f"Error loading VAE model: {e}")

def load_scaler(scaler_path):
    """Load a scaler from a pickle file."""
    try:
        with open(scaler_path, 'rb') as f:
            scaler = load(f)
        return scaler
    except Exception as e:
        print(f"Error loading scaler from {scaler_path}: {e}")
        return None

def descale_latent_predictions(y_pred1, y_pred2, latent_vectors_scaler, xs_scaler):
    """Descale latent conditioner predictions to match VAE decoder expectations."""
    if latent_vectors_scaler is None or xs_scaler is None:
        # Return original predictions without descaling
        return y_pred1, y_pred2
    
    # Convert to numpy for scaler operations
    y_pred1_np = y_pred1.detach().cpu().numpy()
    y_pred2_np = y_pred2.detach().cpu().numpy()
    
    # Descale predictions
    y_pred1_descaled_np = latent_vectors_scaler.inverse_transform(y_pred1_np)
    
    # Handle hierarchical latent vectors (y_pred2) - need to reshape for scaler
    if len(y_pred2_np.shape) == 3:
        original_shape = y_pred2_np.shape
        y_pred2_reshaped = y_pred2_np.reshape(original_shape[0], -1)
        y_pred2_descaled_np = xs_scaler.inverse_transform(y_pred2_reshaped)
        y_pred2_descaled_np = y_pred2_descaled_np.reshape(original_shape)
    else:
        y_pred2_descaled_np = xs_scaler.inverse_transform(y_pred2_np)
    
    # Convert back to tensors on the same device
    y_pred1_descaled = torch.from_numpy(y_pred1_descaled_np).to(y_pred1.device).float()
    y_pred2_descaled = torch.from_numpy(y_pred2_descaled_np).to(y_pred2.device).float()
    
    return y_pred1_descaled, y_pred2_descaled

def verify_tensor_devices(tensors_dict, expected_device):
    """Verify all tensors are on expected device and warn about mismatches."""
    device_issues = []
    
    for name, tensor in tensors_dict.items():
        if torch.is_tensor(tensor):
            if tensor.device != expected_device:
                device_issues.append(f"{name}: {tensor.device} (expected {expected_device})")
        elif isinstance(tensor, (list, tuple)):
            for i, t in enumerate(tensor):
                if torch.is_tensor(t) and t.device != expected_device:
                    device_issues.append(f"{name}[{i}]: {t.device} (expected {expected_device})")
    
    if device_issues:
        print(f"⚠️ DEVICE MISMATCH DETECTED:")
        for issue in device_issues:
            print(f"   • {issue}")
        return False

from modules.latent_conditioner import (
    DEFAULT_IMAGE_SIZE, INTERPOLATION_METHOD,
    read_latent_conditioner_dataset_img, read_latent_conditioner_dataset_img_pca,
    read_latent_conditioner_dataset, apply_outline_preserving_augmentations,
    safe_cuda_initialization, safe_initialize_weights_He, setup_device_and_model
)

def setup_improved_optimizer_and_scheduler(latent_conditioner, latent_conditioner_lr, 
                                         weight_decay, latent_conditioner_epoch):
    """
    Setup improved optimizer with adaptive learning rate scheduling.
    
    Key improvements:
    2. Adaptive plateau detection
    3. Longer warmup period
    4. More conservative decay
    """
    
    optimizer = torch.optim.AdamW(
        latent_conditioner.parameters(), 
        lr=latent_conditioner_lr, 
        weight_decay=weight_decay,
        betas=(0.9, 0.999),  # Standard Adam betas
        eps=1e-8
    )
    
    # Use adaptive scheduler instead of fixed cosine
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=latent_conditioner_epoch,
        eta_min=1e-8
    )
    
    return optimizer, scheduler

def data_augmentation(x, target_data, y1, y2, is_image_data, device, use_latent_regularization):
    """
    Improved data augmentation that's more conservative and physics-aware.
    """
    
    # 1. Very light Gaussian noise (reduced from 0.5% to 0.2%)
    if torch.rand(1, device=device) < 1:
        noise_std = 0.1 # 0.2% of input range
        noise = torch.randn_like(x) * noise_std
        x = x + noise
    
    # 2. Conservative dropout-style occlusion
    # if torch.rand(1, device=device) < 0.3:
    #     mask = torch.ones_like(x)
    #     num_patches = int(0.01 * x.numel())  # Mask only 1% of pixels
    #     indices = torch.randperm(x.numel(), device=device)[:num_patches]
    #     mask.view(-1)[indices] = 0.95  # Very light masking
    #     x = x * mask
    
    # 3. Very mild Gaussian blur (only if image data)
    # if is_image_data and torch.rand(1, device=device) < 0.2:
    #     im_size = int(math.sqrt(x.shape[-1]))
    #     x_2d = x.reshape(-1, im_size, im_size)
    #     # Extremely light smoothing
    #     x_smooth = 0.9 * x_2d + 0.05 * torch.roll(x_2d, 1, dims=1) + 0.05 * torch.roll(x_2d, 1, dims=2)
    #     x = x_smooth.reshape(x.shape[0], -1)
    
    # 4. Ultra-conservative mixup (reduced probability and strength)
    # if torch.rand(1, device=device) < 0.1 and x.size(0) > 1:
    #     # Only mix very similar samples
    #     target_dists = torch.cdist(target_data.view(target_data.size(0), -1), 
    #                              target_data.view(target_data.size(0), -1))
    #     similar_threshold = torch.quantile(target_dists, 0.2)  # Top 20% most similar
    #     similar_pairs = target_dists < similar_threshold
        
    #     if similar_pairs.sum() > x.size(0):
    #         alpha = 0.05  # Very conservative mixing (was 0.1)
    #         lam = torch.tensor(np.random.beta(alpha, alpha), device=device, dtype=x.dtype)
    #         batch_size = x.size(0)
    #         index = torch.randperm(batch_size, device=device)
            
    #         x = lam * x + (1 - lam) * x[index, :]
    #         target_data = lam * target_data + (1 - lam) * target_data[index, :]
    #         if use_latent_regularization:
    #             y1 = lam * y1 + (1 - lam) * y1[index, :]
    #             y2 = lam * y2 + (1 - lam) * y2[index, :]
    
    # 5. Very light brightness/contrast (reduced from ±10% to ±3%)
    # if torch.rand(1, device=device) < 0.3:
    #     brightness_factor = 1.0 + (torch.rand(1, device=device) - 0.5) * 0.06  # ±3%
    #     contrast_factor = 1.0 + (torch.rand(1, device=device) - 0.5) * 0.06
    #     x = torch.clamp(x * contrast_factor + (brightness_factor - 1.0), 0, 1)

    # Implement output augmentation as well
    if torch.rand(1, device=device) < 1:
        noise_std = 0.1 # 0.2% of input range
        noise = torch.randn_like(target_data) * noise_std
        target_data = target_data + noise
        noise = torch.randn_like(y1) * noise_std
        y1 = y1 + noise
        noise = torch.randn_like(y2) * noise_std
        y2 = y2 + noise
    
    return x, target_data, y1, y2

def train_latent_conditioner_e2e(latent_conditioner_epoch, e2e_dataloader, e2e_validation_dataloader, 
                                         latent_conditioner, latent_conditioner_lr, weight_decay, 
                                         is_image_data, image_size, config):
    """
    Improved end-to-end training with adaptive scheduling and stagnation recovery.
    
    Key improvements:
    1. Adaptive learning rate scheduling
    2. Improved regularization decay
    3. Gradient health monitoring
    4. Loss stagnation detection
    5. Better data augmentation
    """

    writer = SummaryWriter(log_dir='./LatentConditionerE2EImprovedRuns', comment='LatentConditioner_E2E_Improved')
    loss = 0
    latent_conditioner, device = setup_device_and_model(latent_conditioner)

    LC_alpha = float(config.get('LC_alpha')) if config else 1.0

    # Load VAE model for end-to-end training
    vae_model_path = config.get('e2e_vae_model_path', 'model_save/SimulGen-VAE') if config else 'model_save/SimulGen-VAE'
    vae_model = load_vae_model(vae_model_path, device)
    
    # Load scalers
    print("Loading scalers for latent prediction descaling...")
    latent_vectors_scaler = load_scaler('./model_save/latent_vectors_scaler.pkl')
    xs_scaler = load_scaler('./model_save/xs_scaler.pkl')
    
    if latent_vectors_scaler is None or xs_scaler is None:
        # raise error
        raise ValueError("Could not load scalers. E2E training will use raw latent predictions.")
    else:
        print("✅ Scalers loaded successfully for latent prediction descaling")
    
    # Setup improved loss function
    loss_function_type = config.get('e2e_loss_function', 'MSE') if config else 'MSE'
    if loss_function_type == 'MSE':
        reconstruction_loss_fn = nn.MSELoss()
    elif loss_function_type == 'MAE':
        reconstruction_loss_fn = nn.L1Loss()
    elif loss_function_type == 'Huber':
        reconstruction_loss_fn = nn.HuberLoss(delta=0.1)  # Smaller delta for smoother gradients
    elif loss_function_type == 'SmoothL1':
        reconstruction_loss_fn = nn.SmoothL1Loss(beta=0.1)  # Smaller beta
    else:
        reconstruction_loss_fn = nn.MSELoss()
        print(f"Unknown loss function {loss_function_type}, using MSE")

    # Configuration for regularization
    use_latent_regularization = config.get('use_latent_regularization') == 1 if config else False
    latent_reg_weight = float(config.get('latent_reg_weight')) if config else 0.001

    # Setup improved optimizer and scheduler
    optimizer, lr_scheduler = setup_improved_optimizer_and_scheduler(
        latent_conditioner, latent_conditioner_lr, weight_decay, latent_conditioner_epoch
    )
    
    latent_conditioner = latent_conditioner.to(device)
    
    # Proper weight initialization for SiLU activation
    def init_weights(m):
        if isinstance(m, nn.Conv2d):
            # He initialization for conv layers with SiLU activation
            nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            # For final prediction layers (typically small output dims), use smaller initialization
            if m.out_features <= 64:  # Final prediction layers
                nn.init.normal_(m.weight, mean=0, std=0.1)
            else:
                nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (nn.GroupNorm, nn.LayerNorm, nn.BatchNorm2d)):
            if hasattr(m, 'weight') and m.weight is not None:
                nn.init.ones_(m.weight)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.zeros_(m.bias)
    
    latent_conditioner.apply(init_weights)
    model_summary_shown = False

    print(f"🚀 Starting improved end-to-end latent conditioner training for {latent_conditioner_epoch} epochs")
    print(f"📊 Reconstruction loss function: {loss_function_type}")
    print(f"🎯 Latent regularization: {'Enabled' if use_latent_regularization else 'Disabled'}")
    
    # Initialize comprehensive loss tracking
    train_losses = [];val_losses = [];train_recon_losses = [];val_recon_losses = [];train_latent_reg_losses = [];val_latent_reg_losses = [];learning_rates = [];regularization_weights = [];gradient_norms = []    
    # Best model tracking
    best_val_loss = float('inf');best_model_state = None;patience_counter = 0;
    
    for epoch in range(latent_conditioner_epoch):
        epoch_start_time = time.time()
        latent_conditioner.train(True)
        
        # Initialize per-epoch variables
        epoch_loss = 0;epoch_recon_loss = 0;epoch_latent_reg_loss = 0;num_batches = 0
        epoch_gradient_sum = 0.0;gradient_count = 0
        
        for i, (x, y1, y2, target_data) in enumerate(e2e_dataloader):
            # Move to device
            x, y1, y2 = x.to(device, non_blocking=True), y1.to(device, non_blocking=True), y2.to(device, non_blocking=True)
            target_data = target_data.to(device, non_blocking=True)
            
            if not model_summary_shown:
                batch_size = x.shape[0]
                input_features = x.shape[-1]
                img_size = int(math.sqrt(input_features))
                
                print(f"🔍 Model Input Analysis:")
                print(f"   Input shape: {x.shape}, range: [{x.min():.4f}, {x.max():.4f}]")
                print(f"   Target shape: {target_data.shape}, range: [{target_data.min():.4f}, {target_data.max():.4f}]")
                print(f"   Image size: {img_size}x{img_size} ({input_features} pixels)")
                
                summary(latent_conditioner, (batch_size, 1, input_features))
                model_summary_shown = True
            
            # Apply improved data augmentation
            x, target_data, y1, y2 = data_augmentation(x, target_data, y1, y2, is_image_data, device, use_latent_regularization)
            
            # Forward pass
            optimizer.zero_grad(set_to_none=True)
            
            y_pred1, y_pred2 = latent_conditioner(x)
            
            # Diagnostic prints (every 100 epochs, first batch only)
            if i == 0 and epoch % 100 == 0:
                print(f"🔧 Epoch {epoch} - CNN vs Target Latents:")
                print(f"   y1 - CNN: [{y_pred1.min():.4f}, {y_pred1.max():.4f}], Target: [{y1.min():.4f}, {y1.max():.4f}]")
                print(f"   y2 - CNN: [{y_pred2.min():.4f}, {y_pred2.max():.4f}], Target: [{y2.min():.4f}, {y2.max():.4f}]")
            
            # Descale predictions for VAE decoder
            y_pred1_descaled, y_pred2_descaled = descale_latent_predictions(y_pred1, y_pred2, latent_vectors_scaler, xs_scaler)
            
            # Diagnostic prints (every 100 epochs, first batch only)
            if i == 0 and epoch % 100 == 0:
                # Get target descaled latents for comparison
                y1_target_descaled, y2_target_descaled = descale_latent_predictions(y1, y2, latent_vectors_scaler, xs_scaler)
                print(f"🔧 Epoch {epoch} - Descaled Latents Comparison:")
                print(f"   y1 - CNN: [{y_pred1_descaled.min():.4f}, {y_pred1_descaled.max():.4f}], Target: [{y1_target_descaled.min():.4f}, {y1_target_descaled.max():.4f}]")
                print(f"   y2 - CNN: [{y_pred2_descaled.min():.4f}, {y_pred2_descaled.max():.4f}], Target: [{y2_target_descaled.min():.4f}, {y2_target_descaled.max():.4f}]")
            
            # Prepare hierarchical latents for VAE decoder
            y_pred2_tensor = y_pred2  # Keep original for regularization
            y_pred2_for_decoder = y_pred2_descaled
            
            if torch.is_tensor(y_pred2_for_decoder):
                if y_pred2_for_decoder.dim() == 3 and y_pred2_for_decoder.shape[1] == 3:
                    y_pred2_for_decoder = [y_pred2_for_decoder[:, i, :] for i in range(y_pred2_for_decoder.shape[1])]
                elif y_pred2_for_decoder.dim() == 2:
                    num_layers = 3
                    latent_dim = y_pred2_for_decoder.shape[1] // num_layers
                    y_pred2_for_decoder = y_pred2_for_decoder.view(y_pred2_for_decoder.shape[0], num_layers, latent_dim)
                    y_pred2_for_decoder = [y_pred2_for_decoder[:, i, :] for i in range(num_layers)]
            
            # VAE decoder forward pass
            reconstructed_data, _ = vae_model.decoder(y_pred1_descaled, y_pred2_for_decoder)
            
            # Diagnostic prints (every 100 epochs, first batch only)
            if i == 0 and epoch % 100 == 0:
                print(f"🔧 Epoch {epoch} - Reconstructed Data - range: [{reconstructed_data.min():.4f}, {reconstructed_data.max():.4f}]")
                print(f"🔧 Epoch {epoch} - Target vs Reconstructed - Target: [{target_data.min():.4f}, {target_data.max():.4f}], Recon: [{reconstructed_data.min():.4f}, {reconstructed_data.max():.4f}]")
            
            # Compute reconstruction loss
            recon_loss = reconstruction_loss_fn(reconstructed_data, target_data)
            
            # Fixed regularization with label smoothing
            if use_latent_regularization:
                current_reg_weight = latent_reg_weight
                
                # # Apply label smoothing to latent targets (0.05 factor)
                # smoothing_factor = 0.05
                # y1_smooth = y1 + smoothing_factor * torch.randn_like(y1)
                # y2_smooth = y2 + smoothing_factor * torch.randn_like(y2)
                
                # Deleted label smoothing
                y1_smooth = y1
                y2_smooth = y2
                
                # Improved regularization with better weighting
                latent_reg_main = nn.MSELoss()(y_pred1, y1_smooth)
                latent_reg_hier = nn.MSELoss()(y_pred2_tensor.reshape(-1), y2_smooth.reshape(-1))

                # Weight main regularization more heavily (it's more important)
                latent_reg_total = 0.9 * latent_reg_main + 0.1 * latent_reg_hier
                
                loss = LC_alpha * recon_loss + current_reg_weight * latent_reg_total
                epoch_latent_reg_loss += (current_reg_weight * latent_reg_total).item()
            else:
                loss = recon_loss
                current_reg_weight = 0.0

            epoch_loss += loss.item()
            epoch_recon_loss += recon_loss.item()
            num_batches += 1
            
            # Backward pass
            loss.backward()
            
            # Hybrid gradient clipping with minimum and maximum bounds
            min_grad_norm = 1e-5  # Minimum gradient norm
            max_grad_norm = 10   # Maximum gradient norm
            
            # Calculate original gradient norm
            total_norm = 0.0
            for p in latent_conditioner.parameters():
                if p.grad is not None:
                    param_norm = p.grad.data.norm(2)
                    total_norm += param_norm.item() ** 2
            original_gradient_norm = total_norm ** (1. / 2)
            
            # Apply hybrid gradient clipping
            if original_gradient_norm > 0:
                if original_gradient_norm < min_grad_norm:
                    # Scale up small gradients
                    scale_factor = min_grad_norm / original_gradient_norm
                    torch.nn.utils.clip_grad_norm_(latent_conditioner.parameters(), min_grad_norm)
                    final_gradient_norm = min_grad_norm
                    if i % 10 == 0:  # Print occasionally to avoid spam
                        print(f"  Batch {i}: Scaled up gradients by {scale_factor:.2f} ({original_gradient_norm:.2E} -> {final_gradient_norm:.2E})")
                elif original_gradient_norm > max_grad_norm:
                    # Scale down large gradients
                    torch.nn.utils.clip_grad_norm_(latent_conditioner.parameters(), max_grad_norm)
                    final_gradient_norm = max_grad_norm
                    if i % 10 == 0:
                        print(f"  Batch {i}: Scaled down gradients ({original_gradient_norm:.2E} -> {final_gradient_norm:.2E})")
                else:
                    # Gradients are in acceptable range
                    final_gradient_norm = original_gradient_norm
            else:
                final_gradient_norm = 0.0
            
            gradient_norms.append(final_gradient_norm)
            
            # Accumulate for epoch averaging (use final gradient norm)
            epoch_gradient_sum += final_gradient_norm
            gradient_count += 1
            
            # Optimizer step
            optimizer.step()
        
        # Calculate training averages
        avg_train_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        avg_train_recon_loss = epoch_recon_loss / num_batches if num_batches > 0 else 0.0
        avg_train_latent_reg_loss = epoch_latent_reg_loss/current_reg_weight / num_batches if num_batches > 0 else 0.0

        # Validation
        latent_conditioner.eval()
        val_loss = 0
        val_recon_loss = 0
        val_latent_reg_loss = 0
        val_batches = 0
        
        with torch.no_grad():
            for x_val, y1_val, y2_val, target_val_data in e2e_validation_dataloader:
                x_val, y1_val, y2_val = x_val.to(device), y1_val.to(device), y2_val.to(device)
                target_val_data = target_val_data.to(device)
                
                y_pred1_val, y_pred2_val = latent_conditioner(x_val)
                
                # Descale validation predictions
                y_pred1_val_descaled, y_pred2_val_descaled = descale_latent_predictions(
                    y_pred1_val, y_pred2_val, latent_vectors_scaler, xs_scaler
                )
                
                # Prepare for VAE decoder
                y_pred2_val_tensor = y_pred2_val
                y_pred2_val_for_decoder = y_pred2_val_descaled
                
                if torch.is_tensor(y_pred2_val_for_decoder):
                    if y_pred2_val_for_decoder.dim() == 3 and y_pred2_val_for_decoder.shape[1] == 3:
                        y_pred2_val_for_decoder = [y_pred2_val_for_decoder[:, i, :] for i in range(y_pred2_val_for_decoder.shape[1])]
                    elif y_pred2_val_for_decoder.dim() == 2:
                        num_layers = 3
                        latent_dim = y_pred2_val_for_decoder.shape[1] // num_layers
                        y_pred2_val_for_decoder = y_pred2_val_for_decoder.view(y_pred2_val_for_decoder.shape[0], num_layers, latent_dim)
                        y_pred2_val_for_decoder = [y_pred2_val_for_decoder[:, i, :] for i in range(num_layers)]
                
                # VAE decoder
                reconstructed_val_data, _ = vae_model.decoder(y_pred1_val_descaled, y_pred2_val_for_decoder)
                recon_loss_val = reconstruction_loss_fn(reconstructed_val_data, target_val_data)
                
                if use_latent_regularization:
                    # NO label smoothing for validation - use clean targets
                    latent_reg_main_val = nn.MSELoss()(y_pred1_val, y1_val)
                    latent_reg_hier_val = nn.MSELoss()(y_pred2_val_tensor.reshape(-1), y2_val.reshape(-1))
                    latent_reg_total_val = 0.9 * latent_reg_main_val + 0.1 * latent_reg_hier_val
                    
                    total_val_loss = LC_alpha * recon_loss_val + current_reg_weight * latent_reg_total_val
                    val_latent_reg_loss += (current_reg_weight * latent_reg_total_val).item()
                else:
                    total_val_loss = recon_loss_val
                
                val_loss += total_val_loss.item()
                val_recon_loss += recon_loss_val.item()
                val_batches += 1

        # Calculate validation averages
        avg_val_loss = val_loss / val_batches if val_batches > 0 else 0.0
        avg_val_recon_loss = val_recon_loss / val_batches if val_batches > 0 else 0.0
        avg_val_latent_reg_loss = val_latent_reg_loss/current_reg_weight / val_batches if val_batches > 0 else 0.0
        
        # Update learning rate scheduler
        lr_scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        
        # Check for best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = latent_conditioner.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Simple progress tracking (removed stagnation detection)
        
        # Store metrics for plotting
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_recon_losses.append(avg_train_recon_loss)
        val_recon_losses.append(avg_val_recon_loss)
        train_latent_reg_losses.append(avg_train_latent_reg_loss)
        val_latent_reg_losses.append(avg_val_latent_reg_loss)
        learning_rates.append(current_lr)
        regularization_weights.append(current_reg_weight)
        
        # Enhanced progress reporting
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        
        # Calculate average gradient norm for the epoch
        avg_gradient_norm = epoch_gradient_sum / gradient_count if gradient_count > 0 else 0.0
        
        scheduler_info = "CosineAnnealing"
        reg_weight_info = f", RegW: {current_reg_weight:.4f}" if use_latent_regularization else ""
        
        print('[%d/%d]\tTrain: %.4E (recon:%.4E, reg:%.4E), Val: %.4E (recon:%.4E, reg:%.4E), LR: %.2E (%s)%s, AvgGrad: %.4E, Best: %.4E, ETA: %.2f h' % 
              (epoch, latent_conditioner_epoch, avg_train_loss, avg_train_recon_loss, avg_train_latent_reg_loss, 
               avg_val_loss, avg_val_recon_loss, avg_val_latent_reg_loss,
               current_lr, scheduler_info, reg_weight_info, avg_gradient_norm, best_val_loss,  
               (latent_conditioner_epoch-epoch)*epoch_duration/3600))
        
    # Save improved model
    torch.save(latent_conditioner.state_dict(), 'checkpoints/latent_conditioner_e2e_improved.pth')
    torch.save(latent_conditioner, 'model_save/LatentConditioner')
    
    return avg_val_loss
