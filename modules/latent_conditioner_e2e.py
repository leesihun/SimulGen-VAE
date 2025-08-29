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
# Removed mixed precision as requested

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
    return True

# Import shared utilities from original module
from modules.latent_conditioner import (
    DEFAULT_IMAGE_SIZE, INTERPOLATION_METHOD,
    read_latent_conditioner_dataset_img, read_latent_conditioner_dataset_img_pca,
    read_latent_conditioner_dataset, apply_outline_preserving_augmentations,
    safe_cuda_initialization, safe_initialize_weights_He, setup_device_and_model
)

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

def setup_optimizer_and_scheduler_e2e(latent_conditioner, latent_conditioner_lr, weight_decay, latent_conditioner_epoch):
    """Setup optimizer and scheduler for end-to-end training with proper learning rate."""
    # Use learning rate from config (0.001) with proper weight decay
    optimizer = torch.optim.AdamW(latent_conditioner.parameters(), lr=latent_conditioner_lr, weight_decay=weight_decay)
    warmup_epochs = 10
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, 
        start_factor=0.1,  # Increased from 0.01 for better gradient flow
        total_iters=warmup_epochs
    )
    
    main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=latent_conditioner_epoch - warmup_epochs, 
        eta_min=1e-6  # Increased from 1e-8 to prevent vanishing gradients
    )
    
    return optimizer, warmup_scheduler, main_scheduler, warmup_epochs

def setup_latent_reg_scheduler(initial_weight, total_epochs, warmup_epochs, decay_rate=2.0):
    """
    Setup latent regularization weight scheduler using exponential decay.
    Simplified and fixed for proper E2E training.
    
    Args:
        initial_weight: Starting regularization weight from config (e.g., 0.001)
        total_epochs: Total number of training epochs
        warmup_epochs: Number of warmup epochs (maintain high weight)
        decay_rate: Exponential decay rate (higher = faster decay)
        
    Returns:
        Function that takes current epoch and returns regularization weight
    """
    # Reasonable decay to reduce overfitting over time
    final_weight = initial_weight / 10  # Target: 1/10 of original weight
    main_epochs = total_epochs - warmup_epochs
    
    def get_reg_weight(epoch):
        if epoch < warmup_epochs:
            # Maintain full regularization during warmup
            return initial_weight
        else:
            # Exponential decay for main training phase
            progress = (epoch - warmup_epochs) / main_epochs
            exponential_decay = math.exp(-decay_rate * progress)
            current_weight = final_weight + (initial_weight - final_weight) * exponential_decay
            # Clamp to reasonable minimum value
            return max(current_weight, initial_weight / 10)
    
    print(f"📉 Latent regularization scheduler: {initial_weight:.6f} → {final_weight:.6f} over {total_epochs} epochs (decay_rate={decay_rate})")
    return get_reg_weight

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

def train_latent_conditioner_e2e(latent_conditioner_epoch, e2e_dataloader, e2e_validation_dataloader, latent_conditioner, latent_conditioner_lr, weight_decay, is_image_data, image_size, config):
    """
    End-to-end training function with the same structure as the original train_latent_conditioner.
    
    Key difference: Loss is computed on reconstructed data instead of latent predictions.
    Architecture: Input Conditions → Latent Conditioner → VAE Decoder → Reconstructed Data
    """

    writer = SummaryWriter(log_dir = './LatentConditionerE2ERuns', comment = 'LatentConditioner_E2E')
    loss = 0
    latent_conditioner, device = setup_device_and_model(latent_conditioner)

    # Load VAE model for end-to-end training
    vae_model_path = config.get('e2e_vae_model_path', 'model_save/SimulGen-VAE') if config else 'model_save/SimulGen-VAE'
    vae_model = load_vae_model(vae_model_path, device)
    
    # Load scalers once at the beginning of training for efficiency
    print("Loading scalers for latent prediction descaling...")
    latent_vectors_scaler = load_scaler('./model_save/latent_vectors_scaler.pkl')
    xs_scaler = load_scaler('./model_save/xs_scaler.pkl')
    
    if latent_vectors_scaler is None or xs_scaler is None:
        print("⚠️ Warning: Could not load scalers. E2E training will use raw latent predictions.")
        print("   This may cause suboptimal results if VAE expects scaled latents.")
        latent_vectors_scaler = None
        xs_scaler = None
    else:
        print("✅ Scalers loaded successfully for latent prediction descaling")
    
    # Compile VAE decoder for 20-30% performance improvement
    #vae_model.decoder = torch.compile(vae_model.decoder)
    #print("✅ VAE decoder compiled for optimized performance")
    
    # Setup reconstruction loss function with label smoothing equivalent
    loss_function_type = config.get('e2e_loss_function', 'MSE') if config else 'MSE'
    if loss_function_type == 'MSE':
        reconstruction_loss_fn = nn.MSELoss()
    elif loss_function_type == 'MAE':
        reconstruction_loss_fn = nn.L1Loss()
    elif loss_function_type == 'Huber':
        reconstruction_loss_fn = nn.HuberLoss()
    elif loss_function_type == 'SmoothL1':
        reconstruction_loss_fn = nn.SmoothL1Loss()
    else:
        reconstruction_loss_fn = nn.MSELoss()
        print(f"Unknown loss function {loss_function_type}, using MSE")
    
    # Removed destructive label smoothing - was adding massive noise to targets
    # This was causing the model to learn corrupted targets instead of clean reconstruction

    # Configuration for end-to-end training - use proper regularization weight from config
    use_latent_regularization = config.get('use_latent_regularization', 0) == 1 if config else False
    latent_reg_weight = float(config.get('latent_reg_weight', 0.001)) if config else 0.001  # Use config value (0.001)

    latent_conditioner_optimized, warmup_scheduler, main_scheduler, warmup_epochs = setup_optimizer_and_scheduler_e2e(
        latent_conditioner, latent_conditioner_lr, weight_decay, latent_conditioner_epoch
    )
    
    # Setup latent regularization weight scheduler
    latent_reg_scheduler = setup_latent_reg_scheduler(latent_reg_weight, latent_conditioner_epoch, warmup_epochs)
    
    latent_conditioner = latent_conditioner.to(device)
    latent_conditioner.apply(safe_initialize_weights_He)
    model_summary_shown = False

    print(f"Starting end-to-end latent conditioner training for {latent_conditioner_epoch} epochs")
    print(f"Reconstruction loss function: {loss_function_type}")
    print(f"Latent regularization: {'Enabled' if use_latent_regularization else 'Disabled'}")
    
    # Initialize loss tracking arrays for plotting
    train_losses = []
    val_losses = []
    train_recon_losses = []
    val_recon_losses = []
    train_latent_reg_losses = []
    val_latent_reg_losses = []
    learning_rates = []
    regularization_weights = []
    
    # Display current VRAM usage at start of E2E training
    if torch.cuda.is_available():
        current_memory = torch.cuda.memory_allocated() / 1024**3
        reserved_memory = torch.cuda.memory_reserved() / 1024**3
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        free_memory = total_memory - reserved_memory
        print(f"=== VRAM Status at E2E Training Start ===")
        print(f"Used: {current_memory:.2f}GB | Reserved: {reserved_memory:.2f}GB | Free: {free_memory:.2f}GB | Total: {total_memory:.2f}GB")
        print(f"==========================================")

    for epoch in range(latent_conditioner_epoch):
        epoch_start_time = time.time()
        latent_conditioner.train(True)
        
        # Initialize per-epoch variables (both training and timing)
        epoch_loss = 0
        epoch_recon_loss = 0
        epoch_latent_reg_loss = 0
        num_batches = 0
        
        # Initialize timing variables per epoch (prevents cross-epoch accumulation)
        total_forward_time = 0
        total_lc_forward_time = 0
        total_tensor_prep_time = 0
        total_vae_decoder_time = 0
        total_loss_comp_time = 0
        total_backward_time = 0
        total_optimization_time = 0
        total_other_time = 0
        
        for i, (x, y1, y2, target_data) in enumerate(e2e_dataloader):
            batch_start_time = time.time()
            
            if not model_summary_shown:
                batch_size = x.shape[0]
                input_features = x.shape[-1]
                img_size = int(math.sqrt(input_features))
                
                print(f"DEBUG: Input shape: {x.shape}, Input range: [{x.min():.4f}, {x.max():.4f}]")
                print(f"DEBUG: Target shape: {target_data.shape}")
                print(f"DEBUG: Image size: {img_size}x{img_size} ({input_features} pixels)")
                
                # Check sample image statistics
                sample_img = x[0].reshape(img_size, img_size)
                print(f"DEBUG: Sample image - mean: {sample_img.mean():.4f}, std: {sample_img.std():.4f}")
                print(f"DEBUG: Non-zero pixels: {(sample_img > 0.01).sum()}/{sample_img.numel()}")
                
                # Check target value ranges
                print(f"DEBUG: Target y1 shape: {y1.shape}, range: [{y1.min():.4f}, {y1.max():.4f}], mean: {y1.mean():.4f}")
                print(f"DEBUG: Target y2 shape: {y2.shape}, range: [{y2.min():.4f}, {y2.max():.4f}], mean: {y2.mean():.4f}")
                
                summary(latent_conditioner, (batch_size, 1, input_features))
                model_summary_shown = True
            
            # # Further reduced data augmentation to prevent training instability
            # if is_image_data and torch.rand(1, device=x.device) < 0.0:
            #     im_size = int(math.sqrt(x.shape[-1]))
            #     x_2d = x.reshape(-1, im_size, im_size)
            #     x_2d = apply_outline_preserving_augmentations(x_2d, prob=0.0)
            #     x = x_2d.reshape(x.shape[0], -1)
                
            # # Further reduced mixup augmentation to prevent overfitting to augmented data
            # if torch.rand(1, device=x.device) < 0.0 and x.size(0) > 1:
            #     alpha = 0.2
            #     lam = torch.tensor(np.random.beta(alpha, alpha), device=x.device, dtype=x.dtype)
            #     batch_size = x.size(0)
            #     index = torch.randperm(batch_size, device=x.device)
                
            #     x = lam * x + (1 - lam) * x[index, :]
            #     target_data = lam * target_data + (1 - lam) * target_data[index, :]
            #     if use_latent_regularization:
            #         y1 = lam * y1 + (1 - lam) * y1[index, :]
            #         y2 = lam * y2 + (1 - lam) * y2[index, :]
            
            # # Further reduced noise augmentation for better generalization
            # if torch.rand(1, device=x.device) < 0.0:
            #     # Further reduced noise intensity to prevent training instability
            #     noise_intensity = 0.005 if epoch < latent_conditioner_epoch // 3 else 0.002
            #     noise = torch.randn_like(x) * noise_intensity
            #     x = x + noise
            
            # # Further reduced gaussian blur augmentation for images
            # if is_image_data and torch.rand(1, device=x.device) < 0.0:
            #     # Simple gaussian-like smoothing
            #     im_size = int(math.sqrt(x.shape[-1]))
            #     x_2d = x.reshape(-1, im_size, im_size)
            #     # Apply light smoothing by averaging with shifted versions
            #     x_smooth = 0.7 * x_2d + 0.15 * torch.roll(x_2d, 1, dims=1) + 0.15 * torch.roll(x_2d, 1, dims=2)
            #     x = x_smooth.reshape(x.shape[0], -1)
            
            other_ops_end_time = time.time()
            batch_other_time = other_ops_end_time - batch_start_time
            total_other_time += batch_other_time
            
            # === STAGE 2: FORWARD PASS (DETAILED PROFILING) ===
            forward_start_time = time.time()
            latent_conditioner_optimized.zero_grad(set_to_none=True)

            # === SUBSTAGE 2A: LATENT CONDITIONER FORWARD ===
            lc_forward_start = time.time()
            y_pred1, y_pred2 = latent_conditioner(x)
            
            # Descale predictions to match VAE decoder expectations
            y_pred1_descaled, y_pred2_descaled = descale_latent_predictions(
                y_pred1, y_pred2, latent_vectors_scaler, xs_scaler
            )
            
            lc_forward_end = time.time()
            lc_forward_time = lc_forward_end - lc_forward_start

            # === SUBSTAGE 2B: TENSOR PREPROCESSING ===
            tensor_prep_start = time.time()
            # Keep original tensor for latent regularization
            y_pred2_tensor = y_pred2
            
            # The VAE decoder expects xs as a list where each element corresponds to a decoder layer
            # Use descaled predictions for VAE decoder
            y_pred2_for_decoder = y_pred2_descaled
            if torch.is_tensor(y_pred2_for_decoder):
                # If y_pred2 is [batch_size, 3, 8] then split along dim 1
                if y_pred2_for_decoder.dim() == 3 and y_pred2_for_decoder.shape[1] == 3:
                    y_pred2_for_decoder = [y_pred2_for_decoder[:, i, :] for i in range(y_pred2_for_decoder.shape[1])]
                # If y_pred2 is [batch_size, num_layers * latent_dim], reshape and split
                elif y_pred2_for_decoder.dim() == 2:
                    # Assume it needs to be reshaped to [batch_size, num_layers, latent_dim]
                    num_layers = 3  # Based on decoder structure
                    latent_dim = y_pred2_for_decoder.shape[1] // num_layers
                    y_pred2_for_decoder = y_pred2_for_decoder.view(y_pred2_for_decoder.shape[0], num_layers, latent_dim)
                    y_pred2_for_decoder = [y_pred2_for_decoder[:, i, :] for i in range(num_layers)]
            elif isinstance(y_pred2_for_decoder, (list, tuple)):
                y_pred2_for_decoder = list(y_pred2_for_decoder)
            tensor_prep_end = time.time()
            tensor_prep_time = tensor_prep_end - tensor_prep_start
            
            # === SUBSTAGE 2C: VAE DECODER (MAIN BOTTLENECK SUSPECT) ===
            vae_decoder_start = time.time()
            
            # NOTE: VAE parameters are frozen (requires_grad=False) but we need gradient flow for E2E training
            # Use descaled predictions for VAE decoder
            reconstructed_data, _ = vae_model.decoder(y_pred1_descaled, y_pred2_for_decoder)
            
            vae_decoder_end = time.time()
            vae_decoder_time = vae_decoder_end - vae_decoder_start
            
            # === SUBSTAGE 2D: LOSS COMPUTATION ===
            loss_comp_start = time.time()
            # Use clean targets for proper reconstruction learning
            recon_loss = reconstruction_loss_fn(reconstructed_data, target_data)
            loss_comp_end = time.time()
            loss_comp_time = loss_comp_end - loss_comp_start
            
            # Optional latent regularization with proper weight scheduling
            if use_latent_regularization:
                # Get current regularization weight from scheduler
                current_reg_weight = latent_reg_scheduler(epoch)
                
                # Simple L2 regularization on latent predictions (avoid overfitting to targets)
                latent_reg_main = nn.MSELoss()(y_pred1, y1)
                latent_reg_hier = nn.MSELoss()(y_pred2_tensor.reshape(-1), y2.reshape(-1))
                
                latent_reg_total = latent_reg_main + latent_reg_hier
                
                # Combine reconstruction loss with scheduled latent regularization
                loss = recon_loss + current_reg_weight * latent_reg_total
                epoch_latent_reg_loss += (current_reg_weight * latent_reg_total).item()
            else:
                loss = recon_loss

            forward_end_time = time.time()
            batch_forward_time = forward_end_time - forward_start_time
            total_forward_time += batch_forward_time
            
            # Accumulate detailed forward pass timings
            total_lc_forward_time += lc_forward_time
            total_tensor_prep_time += tensor_prep_time
            total_vae_decoder_time += vae_decoder_time
            total_loss_comp_time += loss_comp_time

            epoch_loss += loss.item()
            epoch_recon_loss += recon_loss.item()
            num_batches += 1
                
            # === STAGE 3: BACKWARD PASS ===
            backward_start_time = time.time()
            loss.backward()
            backward_end_time = time.time()
            batch_backward_time = backward_end_time - backward_start_time
            total_backward_time += batch_backward_time
            
            # === STAGE 4: OPTIMIZATION ===
            optimization_start_time = time.time()
            
            # Standard gradient clipping for stability (slightly higher for E2E training)
            total_grad_norm = torch.nn.utils.clip_grad_norm_(latent_conditioner.parameters(), max_norm=3.0)
            
            # Removed destructive gradient noise - was preventing stable convergence
            
            # Gradient health monitoring for e2e training
            if total_grad_norm > 2.0:
                print(f"WARNING: Large gradient norm clipped: {total_grad_norm:.2f}")
            elif total_grad_norm < 1e-6:
                print(f"WARNING: Very small gradient norm detected: {total_grad_norm:.2E}")
                
            latent_conditioner_optimized.step()
            
            
            optimization_end_time = time.time()
            # cpu_after_opt = cpu_monitor.get_current_cpu_usage()  # Disabled for performance
            batch_optimization_time = optimization_end_time - optimization_start_time
            total_optimization_time += batch_optimization_time
            
            if epoch % 100 == 0 and i == 0:  # Log every 100 epochs, first batch
                current_lr = latent_conditioner_optimized.param_groups[0]['lr']
                print(f"DEBUG: Gradient norm: {total_grad_norm:.4f}, LR: {current_lr:.2E}, Recon Loss: {recon_loss.item():.4E}, Total Loss: {loss.item():.4E}")
                if total_grad_norm > 10.0:
                    print(f"WARNING: Large gradient norm detected: {total_grad_norm:.2f}")
                elif total_grad_norm < 1e-6:
                    print(f"WARNING: Very small gradient norm: {total_grad_norm:.2E}")
        
        
        avg_train_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        avg_train_recon_loss = epoch_recon_loss / num_batches if num_batches > 0 else 0.0
        avg_train_latent_reg_loss = epoch_latent_reg_loss / num_batches if num_batches > 0 else 0.0

        latent_conditioner.eval()
        val_loss = 0
        val_recon_loss = 0
        val_latent_reg_loss = 0
        val_batches = 0
        
        # Run validation every epoch for complete loss tracking
        validation_start_time = time.time()
        with torch.no_grad():
            for i, (x_val, y1_val, y2_val, target_val_data) in enumerate(e2e_validation_dataloader):
                    
                    x_val, y1_val, y2_val = x_val.to(device), y1_val.to(device), y2_val.to(device)
                    target_val_data = target_val_data.to(device)
                    
                    y_pred1_val, y_pred2_val = latent_conditioner(x_val)
                    
                    # Descale validation predictions to match VAE decoder expectations
                    y_pred1_val_descaled, y_pred2_val_descaled = descale_latent_predictions(
                        y_pred1_val, y_pred2_val, latent_vectors_scaler, xs_scaler
                    )
                    
                    # Keep original tensor for latent regularization
                    y_pred2_val_tensor = y_pred2_val
                    
                    # Convert y_pred2_val to proper list format for VAE decoder
                    y_pred2_val_for_decoder = y_pred2_val_descaled
                    if torch.is_tensor(y_pred2_val_for_decoder):
                        y_pred2_val_for_decoder = [y_pred2_val_for_decoder[:, i, :] for i in range(y_pred2_val_for_decoder.shape[1])]
                    elif isinstance(y_pred2_val_for_decoder, (list, tuple)):
                        y_pred2_val_for_decoder = list(y_pred2_val_for_decoder)
                    
                    # Validate with same end-to-end approach (gradients not needed for validation)
                    with torch.no_grad():
                        reconstructed_val_data, _ = vae_model.decoder(y_pred1_val_descaled, y_pred2_val_for_decoder)
                    recon_loss_val = reconstruction_loss_fn(reconstructed_val_data, target_val_data)
                    
                    if use_latent_regularization:
                        # Use same dynamic regularization weight for validation
                        current_reg_weight = latent_reg_scheduler(epoch)
                        
                        latent_reg_main_val = nn.MSELoss()(y_pred1_val, y1_val)
                        latent_reg_hier_val = nn.MSELoss()(y_pred2_val_tensor.reshape(-1), y2_val.reshape(-1))
                        latent_reg_total_val = latent_reg_main_val + latent_reg_hier_val
                        
                        total_val_loss = recon_loss_val + current_reg_weight * latent_reg_total_val
                        val_latent_reg_loss += (current_reg_weight * latent_reg_total_val).item()
                    else:
                        total_val_loss = recon_loss_val
                    
                    val_loss += total_val_loss.item()
                    val_recon_loss += recon_loss_val.item()
                    val_batches += 1

        # Process validation results (now runs every epoch)
        if val_batches > 0:
            avg_val_loss = val_loss / val_batches
            avg_val_recon_loss = val_recon_loss / val_batches
            avg_val_latent_reg_loss = val_latent_reg_loss / val_batches
            
            validation_end_time = time.time()
            validation_duration = validation_end_time - validation_start_time
            avg_val_batch_time = validation_duration / max(val_batches, 1)
        else:
            avg_val_loss = avg_val_recon_loss = avg_val_latent_reg_loss = 0.0
            validation_duration = 0.0
            avg_val_batch_time = 0.0
            
            
        if epoch < warmup_epochs:
            warmup_scheduler.step()
        else:
            main_scheduler.step()

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        
        # === ELAPSED TIME SUMMARY FOR THIS EPOCH ===
        avg_batch_time = epoch_duration / max(num_batches, 1)
        avg_other_time = total_other_time / max(num_batches, 1)
        avg_forward_time = total_forward_time / max(num_batches, 1)
        avg_backward_time = total_backward_time / max(num_batches, 1)
        avg_optimization_time = total_optimization_time / max(num_batches, 1)
        
        # Detailed forward pass averages
        avg_lc_forward_time = total_lc_forward_time / max(num_batches, 1)
        avg_tensor_prep_time = total_tensor_prep_time / max(num_batches, 1)
        avg_vae_decoder_time = total_vae_decoder_time / max(num_batches, 1)
        avg_loss_comp_time = total_loss_comp_time / max(num_batches, 1)
        
        # Calculate percentages based on actual measurements
        training_only_duration = epoch_duration - validation_duration if validation_duration > 0 else epoch_duration
        other_percent = (total_other_time / training_only_duration) * 100 if training_only_duration > 0 else 0
        forward_percent = (total_forward_time / training_only_duration) * 100 if training_only_duration > 0 else 0
        backward_percent = (total_backward_time / training_only_duration) * 100 if training_only_duration > 0 else 0
        optimization_percent = (total_optimization_time / training_only_duration) * 100 if training_only_duration > 0 else 0
        
        # Calculate remaining percentage (should be close to 0 if measurements are accurate)
        measured_total = other_percent + forward_percent + backward_percent + optimization_percent
        remaining_percent = 100 - measured_total
        
        # Detailed forward pass percentages (of total forward time)
        lc_forward_percent = (total_lc_forward_time / max(total_forward_time, 1e-8)) * 100
        tensor_prep_percent = (total_tensor_prep_time / max(total_forward_time, 1e-8)) * 100
        vae_decoder_percent = (total_vae_decoder_time / max(total_forward_time, 1e-8)) * 100
        loss_comp_percent = (total_loss_comp_time / max(total_forward_time, 1e-8)) * 100

        current_lr = latent_conditioner_optimized.param_groups[0]['lr']
        current_reg_weight = latent_reg_scheduler(epoch) if use_latent_regularization else 0.0
        scheduler_info = f"Warmup" if epoch < warmup_epochs else f"Cosine"
        
        # Store loss values for plotting
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_recon_losses.append(avg_train_recon_loss)
        val_recon_losses.append(avg_val_recon_loss)
        train_latent_reg_losses.append(avg_train_latent_reg_loss)
        val_latent_reg_losses.append(avg_val_latent_reg_loss)
        learning_rates.append(current_lr)
        regularization_weights.append(current_reg_weight)

        if epoch % 100 == 0:
            print(f'    ⏱️  TIMING BREAKDOWN - Epoch {epoch}:')
            print(f'        🔧 Data Prep/Aug:    {avg_other_time*1000:.1f}ms/batch ({other_percent:.1f}%)')
            print(f'        🔄 Forward Pass:     {avg_forward_time*1000:.1f}ms/batch ({forward_percent:.1f}%)')
            print(f'            🧠 Latent Cond:     {avg_lc_forward_time*1000:.1f}ms ({lc_forward_percent:.1f}% of forward)')
            print(f'            🔧 Tensor Prep:     {avg_tensor_prep_time*1000:.1f}ms ({tensor_prep_percent:.1f}% of forward)')
            print(f'            🏭 VAE Decoder:     {avg_vae_decoder_time*1000:.1f}ms ({vae_decoder_percent:.1f}% of forward) ⚠️')
            print(f'            📏 Loss Compute:    {avg_loss_comp_time*1000:.1f}ms ({loss_comp_percent:.1f}% of forward)')
            print(f'        ⬅️  Backward Pass:    {avg_backward_time*1000:.1f}ms/batch ({backward_percent:.1f}%)')
            print(f'        ⚡ Optimization:     {avg_optimization_time*1000:.1f}ms/batch ({optimization_percent:.1f}%)')
            print(f'        🔍 Unmeasured:       {remaining_percent:.1f}% (should be ~0%)')
            
            print(f'        📊 Training Only:    {training_only_duration:.2f}s ({num_batches} batches, {avg_batch_time*1000:.1f}ms/batch avg)')
            if validation_duration > 0:
                print(f'        ✅ Validation:       {validation_duration:.2f}s ({val_batches} batches, {avg_val_batch_time*1000:.1f}ms/batch avg)')
            print(f'        📊 Total Epoch:      {epoch_duration:.2f}s')
                
            
        
        # Enhanced progress display with detailed timing breakdown
        reg_weight_info = f", RegW: {current_reg_weight:.4f}" if use_latent_regularization else ""
        print('[%d/%d]\tTrain: %.4E (recon:%.4E, reg:%.4E), Val: %.4E (recon:%.4E, reg:%.4E), LR: %.2E (%s)%s, ETA: %.2f h' % 
              (epoch, latent_conditioner_epoch, avg_train_loss, avg_train_recon_loss, avg_train_latent_reg_loss, 
               avg_val_loss, avg_val_recon_loss, avg_val_latent_reg_loss,
               current_lr, scheduler_info, reg_weight_info,
               (latent_conditioner_epoch-epoch)*epoch_duration/3600))

    torch.save(latent_conditioner.state_dict(), 'checkpoints/latent_conditioner_e2e.pth')
    torch.save(latent_conditioner, 'model_save/LatentConditioner')
    
    # Save training/validation loss and reconstruction loss as graph in log scale
    print("📊 Generating training visualization plots...")
    
    # Ensure output directory exists
    os.makedirs('output', exist_ok=True)
    
    # Calculate final overfitting ratio for plot titles
    final_train_loss = train_losses[-1] if train_losses else 0
    final_val_loss = val_losses[-1] if val_losses else 0
    overfitting_ratio = final_val_loss / max(final_train_loss, 1e-8)
    
    # Create epoch array for x-axis
    epochs_range = range(len(train_losses))
    
    # Create comprehensive loss visualization
    plt.style.use('default')
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
    fig.suptitle(f'E2E Latent Conditioner Training Progress\nFinal Overfitting Ratio: {overfitting_ratio:.2f}x', fontsize=16, fontweight='bold')
    
    # Plot 1: Total Loss (Train vs Val) in Log Scale
    ax1.plot(epochs_range, train_losses, 'b-', linewidth=2, label='Training Loss', alpha=0.8)
    ax1.plot(epochs_range, val_losses, 'r-', linewidth=2, label='Validation Loss', alpha=0.8)
    ax1.set_yscale('log')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Total Loss (log scale)')
    ax1.set_title('Total Loss: Training vs Validation')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Reconstruction Loss (Train vs Val) in Log Scale
    ax2.plot(epochs_range, train_recon_losses, 'g-', linewidth=2, label='Training Recon Loss', alpha=0.8)
    ax2.plot(epochs_range, val_recon_losses, 'm-', linewidth=2, label='Validation Recon Loss', alpha=0.8)
    ax2.set_yscale('log')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Reconstruction Loss (log scale)')
    ax2.set_title('Reconstruction Loss: Training vs Validation')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Learning Rate Schedule
    ax3.plot(epochs_range, learning_rates, 'orange', linewidth=2, label='Learning Rate')
    ax3.set_yscale('log')
    ax3.set_xlabel('Epoch')
    ax3.set_ylabel('Learning Rate (log scale)')
    ax3.set_title('Learning Rate Schedule')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Regularization Components
    if use_latent_regularization:
        ax4.plot(epochs_range, train_latent_reg_losses, 'c-', linewidth=2, label='Train Latent Reg Loss', alpha=0.8)
        ax4.plot(epochs_range, val_latent_reg_losses, 'y-', linewidth=2, label='Val Latent Reg Loss', alpha=0.8)
        ax4.plot(epochs_range, regularization_weights, 'k--', linewidth=1, label='Reg Weight', alpha=0.6)
        ax4.set_yscale('log')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Regularization Loss & Weight (log scale)')
        ax4.set_title('Latent Regularization Components')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    else:
        # If no regularization, show overfitting ratio over time
        overfitting_ratios = [val/max(train, 1e-8) for train, val in zip(train_losses, val_losses)]
        ax4.plot(epochs_range, overfitting_ratios, 'purple', linewidth=2, label='Overfitting Ratio (Val/Train)')
        ax4.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Perfect Fit Line')
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Overfitting Ratio')
        ax4.set_title('Overfitting Ratio Over Time')
        ax4.legend()
        ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save with timestamp
    import datetime
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    plot_filename = f'output/e2e_training_losses_{timestamp}.png'
    plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
    print(f"📊 Training plots saved to: {plot_filename}")
    
    # Also save as latest for easy access
    latest_filename = 'output/e2e_training_losses_latest.png'
    plt.savefig(latest_filename, dpi=300, bbox_inches='tight')
    print(f"📊 Latest plots saved to: {latest_filename}")
    
    plt.close()
    
    # Save loss data as CSV for further analysis
    loss_data = {
        'epoch': list(epochs_range),
        'train_loss': train_losses,
        'val_loss': val_losses,
        'train_recon_loss': train_recon_losses,
        'val_recon_loss': val_recon_losses,
        'train_latent_reg_loss': train_latent_reg_losses,
        'val_latent_reg_loss': val_latent_reg_losses,
        'learning_rate': learning_rates,
        'regularization_weight': regularization_weights
    }
    
    import pandas as pd
    df = pd.DataFrame(loss_data)
    csv_filename = f'output/e2e_training_data_{timestamp}.csv'
    df.to_csv(csv_filename, index=False)
    print(f"📊 Training data saved to: {csv_filename}")
    
    print(f"✅ Final Results: Train Loss: {final_train_loss:.4E}, Val Loss: {final_val_loss:.4E}, Ratio: {overfitting_ratio:.2f}x")
    
    return avg_val_loss