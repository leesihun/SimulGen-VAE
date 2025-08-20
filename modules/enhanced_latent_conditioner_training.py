import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import math
from torchinfo import summary
from modules.enhanced_loss_functions import compute_enhanced_loss, compute_perceptual_loss
from modules.latent_conditioner import (
    apply_outline_preserving_augmentations,
    setup_device_and_model,
    setup_optimizer_and_scheduler,
    safe_initialize_weights_He
)

def train_latent_conditioner_enhanced(latent_conditioner_epoch, latent_conditioner_dataloader, 
                                    latent_conditioner_validation_dataloader, latent_conditioner, 
                                    latent_conditioner_lr, weight_decay=1e-4, is_image_data=True, 
                                    image_size=256, config=None):
    
    loss = 0
    latent_conditioner, device = setup_device_and_model(latent_conditioner)
    
    optimizer, warmup_scheduler, main_scheduler, warmup_epochs = setup_optimizer_and_scheduler(
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
    
    for epoch in range(latent_conditioner_epoch):
        start_time = time.time()
        latent_conditioner.train(True)
        
        epoch_loss = 0
        epoch_loss_y1 = 0
        epoch_loss_y2 = 0
        num_batches = 0
        
        for i, (x, y1, y2) in enumerate(latent_conditioner_dataloader):
            if x.device != device:
                x, y1, y2 = x.to(device, non_blocking=True), y1.to(device, non_blocking=True), y2.to(device, non_blocking=True)
            
            if not model_summary_shown:
                batch_size = x.shape[0]
                input_features = x.shape[-1]
                img_size = int(math.sqrt(input_features))
                
                print(f"Enhanced training - Input shape: {x.shape}, range: [{x.min():.4f}, {x.max():.4f}]")
                print(f"Image size: {img_size}x{img_size}, Targets: y1={y1.shape}, y2={y2.shape}")
                
                try:
                    summary(latent_conditioner, (batch_size, 1, input_features))
                except Exception as e:
                    print(f"Could not display model summary: {e}")
                
                model_summary_shown = True
            
            # Apply augmentations (same as original)
            if is_image_data and torch.rand(1, device=x.device) < 0.5:
                im_size = int(math.sqrt(x.shape[-1]))
                x_2d = x.reshape(-1, im_size, im_size)
                x_2d = apply_outline_preserving_augmentations(x_2d, prob=0.8)
                x = x_2d.reshape(x.shape[0], -1)
            
            # Mixup augmentation (same as original)
            if torch.rand(1, device=x.device) < 0.02 and x.size(0) > 1:
                alpha = 0.2
                lam = torch.tensor(np.random.beta(alpha, alpha), device=x.device, dtype=x.dtype)
                batch_size = x.size(0)
                index = torch.randperm(batch_size, device=x.device)
                
                x = lam * x + (1 - lam) * x[index, :]
                y1 = lam * y1 + (1 - lam) * y1[index, :]
                y2 = lam * y2 + (1 - lam) * y2[index, :]
            
            # Noise injection (same as original)
            if torch.rand(1, device=x.device) < 0.05:
                noise = torch.randn_like(x) * 0.01
                x = x + noise
            
            optimizer.zero_grad(set_to_none=True)
            
            # Forward pass
            y_pred1, y_pred2 = latent_conditioner(x)
            
            # Use enhanced loss if config provided, otherwise fallback to original MSE
            if config and config.get('use_enhanced_loss', 0):
                loss = compute_enhanced_loss(y_pred1, y_pred2, y1, y2, config)
                
                # Add perceptual loss if enabled
                if config.get('use_perceptual_loss', 0):
                    loss += compute_perceptual_loss(y_pred1, y_pred2, y1, y2, config)
            else:
                A = nn.MSELoss()(y_pred1, y1)
                B = nn.MSELoss()(y_pred2, y2)
                loss = A * 10 + B
            
            # Track pure MSE for display (same as original)
            mae_main = F.L1Loss()(y_pred1, y1)
            mae_hier = F.L1Loss()(y_pred2, y2)
            epoch_loss += (mae_main * 0.9 + mae_hier * 0.1).item()
            epoch_loss_y1 += mae_main.item()
            epoch_loss_y2 += mae_hier.item()
            num_batches += 1
            
            loss.backward()
            
            # Gradient monitoring (same as original)
            total_grad_norm = torch.nn.utils.clip_grad_norm_(latent_conditioner.parameters(), max_norm=10.0)
            
            if epoch % 100 == 0 and i == 0:
                print(f"DEBUG: Gradient norm: {total_grad_norm:.4f}, Loss: {loss.item():.4E}")
                if total_grad_norm > 10.0:
                    print(f"WARNING: Large gradient norm detected: {total_grad_norm:.2f}")
                elif total_grad_norm < 1e-4:
                    print(f"WARNING: Very small gradient norm: {total_grad_norm:.2E}")
            
            optimizer.step()
        
        # Calculate averages (same as original)
        avg_train_loss = epoch_loss / num_batches
        avg_train_loss_y1 = epoch_loss_y1 / num_batches
        avg_train_loss_y2 = epoch_loss_y2 / num_batches
        
        # Validation (same as original)
        latent_conditioner.eval()
        val_loss = 0
        val_loss_y1 = 0
        val_loss_y2 = 0
        val_batches = 0
        
        if epoch % 10 == 0:
            with torch.no_grad():
                for i, (x_val, y1_val, y2_val) in enumerate(latent_conditioner_validation_dataloader):
                    x_val, y1_val, y2_val = x_val.to(device), y1_val.to(device), y2_val.to(device)
                    
                    y_pred1_val, y_pred2_val = latent_conditioner(x_val)
                    
                    A_val = nn.L1Loss()(y_pred1_val, y1_val)
                    B_val = nn.L1Loss()(y_pred2_val, y2_val)
                    
                    val_loss += (A_val * 0.9 + B_val * 0.1).item()
                    val_loss_y1 += A_val.item()
                    val_loss_y2 += B_val.item()
                    val_batches += 1
            
            avg_val_loss = val_loss / val_batches
            avg_val_loss_y1 = val_loss_y1 / val_batches
            avg_val_loss_y2 = val_loss_y2 / val_batches
            
            # Overfitting check (same as original)
            overfitting_ratio = avg_val_loss / max(avg_train_loss, 1e-8)
            if overfitting_ratio > overfitting_threshold:
                print(f'Enhanced training: Severe overfitting detected! Val/Train ratio: {overfitting_ratio:.1f}')
                print(f'Stopping early at epoch {epoch}')
                break
            
            # Early stopping logic (same as original)
            if avg_val_loss < best_val_loss - min_delta:
                best_val_loss = avg_val_loss
                patience_counter = 0
            else:
                patience_counter += 1
        
        # Learning rate scheduling (same as original)
        if epoch < warmup_epochs:
            warmup_scheduler.step()
        else:
            main_scheduler.step()
        
        # Progress output (same format as original)
        end_time = time.time()
        epoch_duration = end_time - start_time
        current_lr = optimizer.param_groups[0]['lr']
        scheduler_info = "Warmup" if epoch < warmup_epochs else "Cosine"
        
        print('[%d/%d]\t L1 losses: Train: %.4E (y1:%.4E, y2:%.4E), Val: %.4E (y1:%.4E, y2:%.4E), LR: %.2E (%s), ETA: %.2f h, Patience: %d/%d' % 
              (epoch, latent_conditioner_epoch, avg_train_loss, avg_train_loss_y1, avg_train_loss_y2,
               avg_val_loss, avg_val_loss_y1, avg_val_loss_y2,
               current_lr, scheduler_info,
               (latent_conditioner_epoch-epoch)*epoch_duration/3600, patience_counter, patience))
        
        # Early stopping check (same as original)
        if patience_counter >= patience:
            print(f'Enhanced training: Early stopping at epoch {epoch}. Best val loss: {best_val_loss:.4E}')
            break
    
    # Save model (same as original)
    torch.save(latent_conditioner.state_dict(), 'checkpoints/enhanced_latent_conditioner.pth')
    torch.save(latent_conditioner, 'model_save/LatentConditioner')
    
    print(f"Enhanced training completed. Final validation loss: {avg_val_loss:.6f}")
    
    return avg_val_loss


def train_latent_conditioner_with_enhancements(latent_conditioner_epoch, latent_conditioner_dataloader,
                                              latent_conditioner_validation_dataloader, latent_conditioner, 
                                              latent_conditioner_lr, weight_decay=1e-4, is_image_data=True,
                                              image_size=256, config=None, use_enhanced_loss=True):
    """Simple wrapper - use enhanced training if enabled, otherwise fallback to original."""
    
    if use_enhanced_loss and config and config.get('use_enhanced_loss', 0):
        print("Using enhanced training with configurable weights")
        return train_latent_conditioner_enhanced(
            latent_conditioner_epoch, latent_conditioner_dataloader, latent_conditioner_validation_dataloader,
            latent_conditioner, latent_conditioner_lr, weight_decay, is_image_data, image_size, config
        )
    else:
        print("Using original training (enhanced features disabled)")
        from modules.latent_conditioner import train_latent_conditioner
        return train_latent_conditioner(
            latent_conditioner_epoch, latent_conditioner_dataloader, latent_conditioner_validation_dataloader,
            latent_conditioner, latent_conditioner_lr, weight_decay, is_image_data, image_size
        )