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

# Import new adaptive scheduling tools
from modules.adaptive_training_scheduler import (
    AdaptiveLearningRateScheduler, AdaptiveRegularizationScheduler,
    GradientHealthMonitor, LossStagnationDetector, create_loss_stagnation_plot
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
    adaptive_scheduler = AdaptiveLearningRateScheduler(
        optimizer=optimizer,
        initial_lr=latent_conditioner_lr,
        patience=100,  # Wait 20 epochs before reducing
        factor=0.8,   # Reduce by 20% each time
        min_lr=1e-6,
        plateau_threshold=1e-6,
        warmup_epochs=30  # Longer warmup for E2E
    )
    
    return optimizer, adaptive_scheduler

def improved_data_augmentation(x, target_data, y1, y2, is_image_data, device, use_latent_regularization):
    """
    Improved data augmentation that's more conservative and physics-aware.
    """
    
    # 1. Very light Gaussian noise (reduced from 0.5% to 0.2%)
    if torch.rand(1, device=device) < 0.4:
        noise_std = 0.02  # 0.2% of input range
        noise = torch.randn_like(x) * noise_std
        x = x + noise
    
    # 2. Conservative dropout-style occlusion
    if torch.rand(1, device=device) < 0.3:
        mask = torch.ones_like(x)
        num_patches = int(0.01 * x.numel())  # Mask only 1% of pixels
        indices = torch.randperm(x.numel(), device=device)[:num_patches]
        mask.view(-1)[indices] = 0.95  # Very light masking
        x = x * mask
    
    # 3. Very mild Gaussian blur (only if image data)
    if is_image_data and torch.rand(1, device=device) < 0.2:
        im_size = int(math.sqrt(x.shape[-1]))
        x_2d = x.reshape(-1, im_size, im_size)
        # Extremely light smoothing
        x_smooth = 0.9 * x_2d + 0.05 * torch.roll(x_2d, 1, dims=1) + 0.05 * torch.roll(x_2d, 1, dims=2)
        x = x_smooth.reshape(x.shape[0], -1)
    
    # 4. Ultra-conservative mixup (reduced probability and strength)
    if torch.rand(1, device=device) < 0.1 and x.size(0) > 1:
        # Only mix very similar samples
        target_dists = torch.cdist(target_data.view(target_data.size(0), -1), 
                                 target_data.view(target_data.size(0), -1))
        similar_threshold = torch.quantile(target_dists, 0.2)  # Top 20% most similar
        similar_pairs = target_dists < similar_threshold
        
        if similar_pairs.sum() > x.size(0):
            alpha = 0.05  # Very conservative mixing (was 0.1)
            lam = torch.tensor(np.random.beta(alpha, alpha), device=device, dtype=x.dtype)
            batch_size = x.size(0)
            index = torch.randperm(batch_size, device=device)
            
            x = lam * x + (1 - lam) * x[index, :]
            target_data = lam * target_data + (1 - lam) * target_data[index, :]
            if use_latent_regularization:
                y1 = lam * y1 + (1 - lam) * y1[index, :]
                y2 = lam * y2 + (1 - lam) * y2[index, :]
    
    # 5. Very light brightness/contrast (reduced from ±10% to ±3%)
    if torch.rand(1, device=device) < 0.3:
        brightness_factor = 1.0 + (torch.rand(1, device=device) - 0.5) * 0.06  # ±3%
        contrast_factor = 1.0 + (torch.rand(1, device=device) - 0.5) * 0.06
        x = torch.clamp(x * contrast_factor + (brightness_factor - 1.0), 0, 1)
    
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

    # Load VAE model for end-to-end training
    vae_model_path = config.get('e2e_vae_model_path', 'model_save/SimulGen-VAE') if config else 'model_save/SimulGen-VAE'
    vae_model = load_vae_model(vae_model_path, device)
    
    # Load scalers
    print("Loading scalers for latent prediction descaling...")
    latent_vectors_scaler = load_scaler('./model_save/latent_vectors_scaler.pkl')
    xs_scaler = load_scaler('./model_save/xs_scaler.pkl')
    
    if latent_vectors_scaler is None or xs_scaler is None:
        print("⚠️ Warning: Could not load scalers. E2E training will use raw latent predictions.")
        latent_vectors_scaler = None
        xs_scaler = None
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
    use_latent_regularization = config.get('use_latent_regularization', 0) == 1 if config else False
    latent_reg_weight = float(config.get('latent_reg_weight', 0.001)) if config else 0.001

    # Setup improved optimizer and scheduler
    optimizer, adaptive_lr_scheduler = setup_improved_optimizer_and_scheduler(
        latent_conditioner, latent_conditioner_lr, weight_decay, latent_conditioner_epoch
    )
    
    # Setup improved regularization scheduler
    adaptive_reg_scheduler = AdaptiveRegularizationScheduler(
        initial_weight=latent_reg_weight,
        total_epochs=latent_conditioner_epoch,
        warmup_epochs=30,
        plateau_boost=1.5,  # More conservative boost
        min_weight_ratio=0.2  # Keep more regularization
    )
    
    # Setup gradient health monitor
    grad_monitor = GradientHealthMonitor(
        model=latent_conditioner,
        base_clip_norm=3.0,  # More conservative base clipping
        adaptive_factor=1.5
    )
    
    # Setup loss stagnation detector
    stagnation_detector = LossStagnationDetector(
        patience=25,  # Slightly more patient
        min_improvement=1e-7
    )
    
    latent_conditioner = latent_conditioner.to(device)
    latent_conditioner.apply(safe_initialize_weights_He)
    model_summary_shown = False

    print(f"🚀 Starting improved end-to-end latent conditioner training for {latent_conditioner_epoch} epochs")
    print(f"📊 Reconstruction loss function: {loss_function_type}")
    print(f"🎯 Latent regularization: {'Enabled' if use_latent_regularization else 'Disabled'}")
    
    # Initialize comprehensive loss tracking
    train_losses = []
    val_losses = []
    train_recon_losses = []
    val_recon_losses = []
    train_latent_reg_losses = []
    val_latent_reg_losses = []
    learning_rates = []
    regularization_weights = []
    gradient_norms = []
    
    # Best model tracking
    best_val_loss = float('inf')
    best_model_state = None
    patience_counter = 0
    
    for epoch in range(latent_conditioner_epoch):
        epoch_start_time = time.time()
        latent_conditioner.train(True)
        
        # Initialize per-epoch variables
        epoch_loss = 0
        epoch_recon_loss = 0
        epoch_latent_reg_loss = 0
        num_batches = 0
        
        for i, (x, y1, y2, target_data) in enumerate(e2e_dataloader):
            batch_start_time = time.time()
            
            # Move to device
            x, y1, y2 = x.to(device, non_blocking=True), y1.to(device, non_blocking=True), y2.to(device, non_blocking=True)
            target_data = target_data.to(device, non_blocking=True)
            
            if not model_summary_shown:
                batch_size = x.shape[0]
                input_features = x.shape[-1]
                img_size = int(math.sqrt(input_features))
                
                print(f"🔍 Model Input Analysis:")
                print(f"   Input shape: {x.shape}, range: [{x.min():.4f}, {x.max():.4f}]")
                print(f"   Target shape: {target_data.shape}")
                print(f"   Image size: {img_size}x{img_size} ({input_features} pixels)")
                
                try:
                    summary(latent_conditioner, (batch_size, 1, input_features))
                except Exception as e:
                    print(f"Could not display model summary: {e}")
                
                model_summary_shown = True
            
            # Apply improved data augmentation
            x, target_data, y1, y2 = improved_data_augmentation(
                x, target_data, y1, y2, is_image_data, device, use_latent_regularization
            )
            
            # Forward pass
            optimizer.zero_grad(set_to_none=True)
            
            y_pred1, y_pred2 = latent_conditioner(x)
            
            # Descale predictions for VAE decoder
            y_pred1_descaled, y_pred2_descaled = descale_latent_predictions(
                y_pred1, y_pred2, latent_vectors_scaler, xs_scaler
            )
            
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
            
            # Compute reconstruction loss
            recon_loss = reconstruction_loss_fn(reconstructed_data, target_data)
            
            # Adaptive regularization
            if use_latent_regularization:
                current_reg_weight = adaptive_reg_scheduler.get_weight(epoch, recon_loss.item())
                
                # Improved regularization with better weighting
                latent_reg_main = nn.MSELoss()(y_pred1, y1)
                latent_reg_hier = nn.MSELoss()(y_pred2_tensor.reshape(-1), y2.reshape(-1))
                
                # Weight hierarchical regularization more heavily (it's more important)
                latent_reg_total = 0.3 * latent_reg_main + 0.7 * latent_reg_hier
                
                loss = recon_loss + current_reg_weight * latent_reg_total
                epoch_latent_reg_loss += (current_reg_weight * latent_reg_total).item()
            else:
                loss = recon_loss
                current_reg_weight = 0.0

            epoch_loss += loss.item()
            epoch_recon_loss += recon_loss.item()
            num_batches += 1
            
            # Backward pass
            loss.backward()
            
            # Adaptive gradient clipping and monitoring
            clipped_grad_norm = grad_monitor.clip_and_monitor(epoch)
            gradient_norms.append(clipped_grad_norm)
            
            # Optimizer step
            optimizer.step()
            
            # Log detailed info periodically
            if epoch % 50 == 0 and i == 0:
                current_lr = optimizer.param_groups[0]['lr']
                print(f"🔍 Detailed Training Info (Epoch {epoch}):")
                print(f"   Learning Rate: {current_lr:.2e}")
                print(f"   Gradient Norm: {clipped_grad_norm:.4f}")
                print(f"   Recon Loss: {recon_loss.item():.6e}")
                print(f"   Reg Weight: {current_reg_weight:.6f}")
                print(f"   Total Loss: {loss.item():.6e}")
        
        # Calculate training averages
        avg_train_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        avg_train_recon_loss = epoch_recon_loss / num_batches if num_batches > 0 else 0.0
        avg_train_latent_reg_loss = epoch_latent_reg_loss / num_batches if num_batches > 0 else 0.0

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
                    y_pred2_val_for_decoder = [y_pred2_val_for_decoder[:, i, :] for i in range(y_pred2_val_for_decoder.shape[1])]
                
                # VAE decoder
                reconstructed_val_data, _ = vae_model.decoder(y_pred1_val_descaled, y_pred2_val_for_decoder)
                recon_loss_val = reconstruction_loss_fn(reconstructed_val_data, target_val_data)
                
                if use_latent_regularization:
                    latent_reg_main_val = nn.MSELoss()(y_pred1_val, y1_val)
                    latent_reg_hier_val = nn.MSELoss()(y_pred2_val_tensor.reshape(-1), y2_val.reshape(-1))
                    latent_reg_total_val = 0.3 * latent_reg_main_val + 0.7 * latent_reg_hier_val
                    
                    total_val_loss = recon_loss_val + current_reg_weight * latent_reg_total_val
                    val_latent_reg_loss += (current_reg_weight * latent_reg_total_val).item()
                else:
                    total_val_loss = recon_loss_val
                
                val_loss += total_val_loss.item()
                val_recon_loss += recon_loss_val.item()
                val_batches += 1

        # Calculate validation averages
        avg_val_loss = val_loss / val_batches if val_batches > 0 else 0.0
        avg_val_recon_loss = val_recon_loss / val_batches if val_batches > 0 else 0.0
        avg_val_latent_reg_loss = val_latent_reg_loss / val_batches if val_batches > 0 else 0.0
        
        # Update adaptive learning rate scheduler
        current_lr = adaptive_lr_scheduler.step(avg_val_loss)
        
        # Check for best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model_state = latent_conditioner.state_dict().copy()
            patience_counter = 0
        else:
            patience_counter += 1
        
        # Loss stagnation detection
        stagnation_info = stagnation_detector.update(avg_train_loss, avg_val_loss, epoch)
        if stagnation_info:
            print(f"🚨 Loss Stagnation Detected at Epoch {epoch}:")
            print(f"   Stagnant for {stagnation_info['stagnation_epochs']} epochs")
            print(f"   Issues: {', '.join(stagnation_info['issues'])}")
            
            # Implement recovery strategies
            if 'learning_rate_too_low' in stagnation_info['issues']:
                # Boost learning rate temporarily
                for param_group in optimizer.param_groups:
                    param_group['lr'] *= 2.0
                print(f"   Recovery: Boosted LR to {optimizer.param_groups[0]['lr']:.2e}")
        
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
        
        scheduler_info = f"Adaptive (patience: {adaptive_lr_scheduler.plateau_count}/{adaptive_lr_scheduler.patience})"
        reg_weight_info = f", RegW: {current_reg_weight:.4f}" if use_latent_regularization else ""
        
        print('[%d/%d]\t🎯 Train: %.4E (recon:%.4E, reg:%.4E), Val: %.4E (recon:%.4E, reg:%.4E), LR: %.2E (%s)%s, Best: %.4E, ETA: %.2f h' % 
              (epoch, latent_conditioner_epoch, avg_train_loss, avg_train_recon_loss, avg_train_latent_reg_loss, 
               avg_val_loss, avg_val_recon_loss, avg_val_latent_reg_loss,
               current_lr, scheduler_info, reg_weight_info, best_val_loss,
               (latent_conditioner_epoch-epoch)*epoch_duration/3600))
        
        # Early stopping check (more conservative)
        if patience_counter >= 200:  # Increased patience
            print(f'🛑 Early stopping at epoch {epoch}. Best val loss: {best_val_loss:.4E}')
            break

    # Load best model
    if best_model_state is not None:
        latent_conditioner.load_state_dict(best_model_state)
        print(f"✅ Loaded best model with validation loss: {best_val_loss:.4E}")

    # Save improved model
    torch.save(latent_conditioner.state_dict(), 'checkpoints/latent_conditioner_e2e_improved.pth')
    torch.save(latent_conditioner, 'model_save/LatentConditioner_Improved')
    
    # Generate comprehensive analysis plots
    print("📊 Generating improved training analysis plots...")
    
    os.makedirs('output', exist_ok=True)
    
    # Enhanced loss visualization
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Improved E2E Training Analysis\nFinal: Train={train_losses[-1]:.4E}, Val={val_losses[-1]:.4E}, Best={best_val_loss:.4E}', 
                 fontsize=16, fontweight='bold')
    
    epochs_range = range(len(train_losses))
    
    # Loss progression with best model marker
    ax1.plot(epochs_range, train_losses, 'b-', linewidth=2, label='Training Loss', alpha=0.8)
    ax1.plot(epochs_range, val_losses, 'r-', linewidth=2, label='Validation Loss', alpha=0.8)
    if best_val_loss < float('inf'):
        best_epoch = val_losses.index(min(val_losses))
        ax1.axvline(x=best_epoch, color='green', linestyle='--', alpha=0.7, label=f'Best Model (Epoch {best_epoch})')
    ax1.set_yscale('log')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Total Loss (log scale)')
    ax1.set_title('Loss Progression with Best Model')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Learning rate and regularization schedule
    ax2.plot(epochs_range, learning_rates, 'orange', linewidth=2, label='Learning Rate')
    ax2_twin = ax2.twinx()
    ax2_twin.plot(epochs_range, regularization_weights, 'purple', linewidth=2, label='Reg Weight')
    ax2.set_yscale('log')
    ax2_twin.set_yscale('log')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Learning Rate (log scale)', color='orange')
    ax2_twin.set_ylabel('Regularization Weight (log scale)', color='purple')
    ax2.set_title('Adaptive Scheduling')
    ax2.grid(True, alpha=0.3)
    
    # Gradient norms
    if gradient_norms:
        ax3.plot(gradient_norms, 'green', linewidth=1, alpha=0.7, label='Gradient Norm')
        ax3.axhline(y=3.0, color='red', linestyle='--', alpha=0.5, label='Base Clip Norm')
        ax3.set_xlabel('Training Step')
        ax3.set_ylabel('Gradient Norm')
        ax3.set_title('Gradient Health Monitoring')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
    
    # Reconstruction loss focus
    ax4.plot(epochs_range, train_recon_losses, 'g-', linewidth=2, label='Training Recon Loss', alpha=0.8)
    ax4.plot(epochs_range, val_recon_losses, 'm-', linewidth=2, label='Validation Recon Loss', alpha=0.8)
    ax4.set_yscale('log')
    ax4.set_xlabel('Epoch')
    ax4.set_ylabel('Reconstruction Loss (log scale)')
    ax4.set_title('Reconstruction Loss Detail')
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save with timestamp
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f'output/e2e_improved_training_{timestamp}.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"📊 Improved training plots saved to: {plot_filename}")
    
    plt.close()
    
    # Generate stagnation analysis plot
    stagnation_plot_path = f'output/loss_stagnation_analysis_{timestamp}.png'
    create_loss_stagnation_plot(train_losses, val_losses, stagnation_plot_path)
    
    print(f"✅ Improved E2E Training completed!")
    print(f"   Final validation loss: {avg_val_loss:.6E}")
    print(f"   Best validation loss: {best_val_loss:.6E}")
    print(f"   Training completed with adaptive scheduling")
    
    return avg_val_loss
