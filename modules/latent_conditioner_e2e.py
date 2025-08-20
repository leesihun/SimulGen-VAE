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
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
from modules.pca_preprocessor import PCAPreprocessor
import matplotlib.pyplot as plt

# Import shared utilities from original module
from modules.latent_conditioner import (
    DEFAULT_IMAGE_SIZE, INTERPOLATION_METHOD,
    read_latent_conditioner_dataset_img, read_latent_conditioner_dataset_img_pca,
    read_latent_conditioner_dataset, apply_outline_preserving_augmentations,
    safe_cuda_initialization, safe_initialize_weights_He, setup_device_and_model
)

def setup_optimizer_and_scheduler_e2e(latent_conditioner, latent_conditioner_lr, weight_decay, latent_conditioner_epoch):
    """Setup optimizer and scheduler for end-to-end training - same as original but with E2E suffix."""
    optimizer = torch.optim.AdamW(latent_conditioner.parameters(), lr=latent_conditioner_lr, weight_decay=weight_decay)
    warmup_epochs = 100
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, 
        start_factor=0.01,
        total_iters=warmup_epochs
    )
    
    main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=latent_conditioner_epoch - warmup_epochs, 
        eta_min=1e-8
    )
    
    return optimizer, warmup_scheduler, main_scheduler, warmup_epochs

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
        print("Using dummy VAE model")
        class DummyVAE:
            def __init__(self):
                pass
            def decoder(self, latent_main, latent_hier):
                batch_size = latent_main.shape[0]
                return torch.randn(batch_size, 95008, 200, device=device), []
        return DummyVAE()

def train_latent_conditioner_e2e(latent_conditioner_epoch, 
latent_conditioner_dataloader, 
latent_conditioner_validation_dataloader, 
latent_conditioner, 
target_dataloader, 
latent_conditioner_lr, 
weight_decay, 
is_image_data, 
image_size, 
config):
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
    
    # Setup reconstruction loss function
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

    latent_conditioner_optimized, warmup_scheduler, main_scheduler, warmup_epochs = setup_optimizer_and_scheduler_e2e(
        latent_conditioner, latent_conditioner_lr, weight_decay, latent_conditioner_epoch
    )
    
    best_val_loss = float('inf')
    patience = 100000
    patience_counter = 0
    min_delta = 1e-8
    overfitting_threshold = 1000.0

    latent_conditioner = latent_conditioner.to(device)
    latent_conditioner.apply(safe_initialize_weights_He)
    model_summary_shown = False

    # Configuration for end-to-end training
    use_latent_regularization = config.get('use_latent_regularization', 0) == 1 if config else False
    latent_reg_weight = float(config.get('latent_reg_weight', 0.1)) if config else 0.1

    print(f"Starting end-to-end latent conditioner training for {latent_conditioner_epoch} epochs")
    print(f"Reconstruction loss function: {loss_function_type}")
    print(f"Latent regularization: {'Enabled' if use_latent_regularization else 'Disabled'}")
    
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
        start_time = time.time()
        latent_conditioner.train(True)
        
        epoch_loss = 0
        epoch_recon_loss = 0
        epoch_latent_reg_loss = 0
        num_batches = 0
        
        # Synchronize data loaders for target data
        target_iter = iter(target_dataloader)
        
        for i, (x, y1, y2) in enumerate(latent_conditioner_dataloader):
            # Get corresponding target data
            try:
                target_data = next(target_iter)
                if isinstance(target_data, (list, tuple)):
                    target_data = target_data[0]  # Extract data if wrapped in tuple
            except StopIteration:
                # Restart target iterator if we run out
                target_iter = iter(target_dataloader)
                target_data = next(target_iter)
                if isinstance(target_data, (list, tuple)):
                    target_data = target_data[0]
            
            if x.device != device:
                x, y1, y2 = x.to(device, non_blocking=True), y1.to(device, non_blocking=True), y2.to(device, non_blocking=True)
            target_data = target_data.to(device, non_blocking=True)
            
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
            
            # Data augmentation (same as original)
            if is_image_data and torch.rand(1, device=x.device) < 0.5:
                im_size = int(math.sqrt(x.shape[-1]))
                x_2d = x.reshape(-1, im_size, im_size)
                x_2d = apply_outline_preserving_augmentations(x_2d, prob=0.8)
                x = x_2d.reshape(x.shape[0], -1)
                
            # Mixup augmentation (applied to both input and target data)
            if torch.rand(1, device=x.device) < 0.02 and x.size(0) > 1:
                alpha = 0.2
                lam = torch.tensor(np.random.beta(alpha, alpha), device=x.device, dtype=x.dtype)
                batch_size = x.size(0)
                index = torch.randperm(batch_size, device=x.device)
                
                x = lam * x + (1 - lam) * x[index, :]
                target_data = lam * target_data + (1 - lam) * target_data[index, :]
                if use_latent_regularization:
                    y1 = lam * y1 + (1 - lam) * y1[index, :]
                    y2 = lam * y2 + (1 - lam) * y2[index, :]
            
            if torch.rand(1, device=x.device) < 0.05:
                noise = torch.randn_like(x) * 0.01
                x = x + noise
            
            latent_conditioner_optimized.zero_grad(set_to_none=True)

            try:
                # Forward pass through latent conditioner
                y_pred1, y_pred2 = latent_conditioner(x)

                y_pred2 = list(list(y_pred2))
                
                # ==== KEY DIFFERENCE: Use VAE decoder to reconstruct data ====
                with torch.no_grad():
                    reconstructed_data, _ = vae_model.decoder(y_pred1, y_pred2)
                
                # Primary loss: reconstruction quality
                recon_loss = reconstruction_loss_fn(reconstructed_data, target_data)
                
                # Optional latent regularization (same as original but weighted)
                if use_latent_regularization:
                    latent_reg_main = nn.MSELoss()(y_pred1, y1)
                    latent_reg_hier = nn.MSELoss()(y_pred2, y2)
                    latent_reg_total = latent_reg_main + latent_reg_hier
                    
                    # Combine reconstruction loss with latent regularization
                    loss = recon_loss + latent_reg_weight * latent_reg_total
                    epoch_latent_reg_loss += (latent_reg_weight * latent_reg_total).item()
                else:
                    # Pure end-to-end loss: only reconstruction quality matters
                    loss = recon_loss

                epoch_loss += loss.item()
                epoch_recon_loss += recon_loss.item()
                num_batches += 1
                
            except RuntimeError as e:
                if "out of memory" in str(e):
                    print(f"CUDA OOM at epoch {epoch+1}, batch {i+1}. Skipping batch.")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise
            
            loss.backward()
            
            # Check gradient norms before clipping
            total_grad_norm = torch.nn.utils.clip_grad_norm_(latent_conditioner.parameters(), max_norm=10.0)
            
            # Monitor gradient health
            if epoch % 100 == 0 and i == 0:  # Log every 100 epochs, first batch
                print(f"DEBUG: Gradient norm: {total_grad_norm:.4f}, Recon Loss: {recon_loss.item():.4E}, Total Loss: {loss.item():.4E}")
                if total_grad_norm > 10.0:
                    print(f"WARNING: Large gradient norm detected: {total_grad_norm:.2f}")
                elif total_grad_norm < 1e-4:
                    print(f"WARNING: Very small gradient norm: {total_grad_norm:.2E}")
            
            latent_conditioner_optimized.step()
        
        
        avg_train_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        avg_train_recon_loss = epoch_recon_loss / num_batches if num_batches > 0 else 0.0
        avg_train_latent_reg_loss = epoch_latent_reg_loss / num_batches if num_batches > 0 else 0.0

        latent_conditioner.eval()
        val_loss = 0
        val_recon_loss = 0
        val_latent_reg_loss = 0
        val_batches = 0
        
        if epoch % 10 == 0:
            with torch.no_grad():
                target_val_iter = iter(target_dataloader)  # Use same target data for validation
                
                for i, (x_val, y1_val, y2_val) in enumerate(latent_conditioner_validation_dataloader):
                    try:
                        target_val_data = next(target_val_iter)
                        if isinstance(target_val_data, (list, tuple)):
                            target_val_data = target_val_data[0]
                    except StopIteration:
                        break
                    
                    x_val, y1_val, y2_val = x_val.to(device), y1_val.to(device), y2_val.to(device)
                    target_val_data = target_val_data.to(device)
                    
                    y_pred1_val, y_pred2_val = latent_conditioner(x_val)
                    
                    # Validate with same end-to-end approach
                    reconstructed_val_data, _ = vae_model.decoder(y_pred1_val, y_pred2_val)
                    recon_loss_val = reconstruction_loss_fn(reconstructed_val_data, target_val_data)
                    
                    if use_latent_regularization:
                        latent_reg_main_val = nn.MSELoss()(y_pred1_val, y1_val)
                        latent_reg_hier_val = nn.MSELoss()(y_pred2_val, y2_val)
                        latent_reg_total_val = latent_reg_main_val + latent_reg_hier_val
                        
                        total_val_loss = recon_loss_val + latent_reg_weight * latent_reg_total_val
                        val_latent_reg_loss += (latent_reg_weight * latent_reg_total_val).item()
                    else:
                        total_val_loss = recon_loss_val
                    
                    val_loss += total_val_loss.item()
                    val_recon_loss += recon_loss_val.item()
                    val_batches += 1

            if val_batches > 0:
                avg_val_loss = val_loss / val_batches
                avg_val_recon_loss = val_recon_loss / val_batches
                avg_val_latent_reg_loss = val_latent_reg_loss / val_batches
            else:
                avg_val_loss = avg_val_recon_loss = avg_val_latent_reg_loss = 0.0

            overfitting_ratio = avg_val_loss / max(avg_train_loss, 1e-8)
            if overfitting_ratio > overfitting_threshold:
                print(f'Severe overfitting detected! Val/Train ratio: {overfitting_ratio:.1f}')
                print(f'Stopping early at epoch {epoch}')
                break
                
            if avg_val_loss < best_val_loss - min_delta:
                best_val_loss = avg_val_loss
                patience_counter = 0
            else:
                patience_counter += 1
        else:
            avg_val_loss = avg_val_recon_loss = avg_val_latent_reg_loss = 0.0
            
        if epoch < warmup_epochs:
            warmup_scheduler.step()
        else:
            main_scheduler.step()

        end_time = time.time()
        epoch_duration = end_time - start_time

        if epoch % 100 == 0:
            writer.add_scalar('LatentConditioner Loss/train_total', avg_train_loss, epoch)
            writer.add_scalar('LatentConditioner Loss/train_reconstruction', avg_train_recon_loss, epoch)
            writer.add_scalar('LatentConditioner Loss/val_total', avg_val_loss, epoch)
            writer.add_scalar('LatentConditioner Loss/val_reconstruction', avg_val_recon_loss, epoch)
            if use_latent_regularization:
                writer.add_scalar('LatentConditioner Loss/train_latent_reg', avg_train_latent_reg_loss, epoch)
                writer.add_scalar('LatentConditioner Loss/val_latent_reg', avg_val_latent_reg_loss, epoch)
            writer.add_scalar('Learning Rate', latent_conditioner_optimized.param_groups[0]['lr'], epoch)

        current_lr = latent_conditioner_optimized.param_groups[0]['lr']
        scheduler_info = f"Warmup" if epoch < warmup_epochs else f"Cosine"
        
        print('[%d/%d]\tTrain: %.4E (recon:%.4E, reg:%.4E), Val: %.4E (recon:%.4E, reg:%.4E), LR: %.2E (%s), ETA: %.2f h, Patience: %d/%d' % 
              (epoch, latent_conditioner_epoch, avg_train_loss, avg_train_recon_loss, avg_train_latent_reg_loss, 
               avg_val_loss, avg_val_recon_loss, avg_val_latent_reg_loss,
               current_lr, scheduler_info,
               (latent_conditioner_epoch-epoch)*epoch_duration/3600, patience_counter, patience))
               
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch}. Best validation loss: {best_val_loss:.4E}')
            break

    torch.save(latent_conditioner.state_dict(), 'checkpoints/latent_conditioner_e2e.pth')
    torch.save(latent_conditioner, 'model_save/LatentConditioner_E2E')

    writer.close()
    print(f"End-to-end training completed. Final validation loss: {best_val_loss:.4E}")

    return avg_val_loss