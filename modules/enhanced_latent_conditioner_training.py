"""Enhanced Latent Conditioner Training Module

This module provides enhanced training functionality for the CNN-based latent 
conditioner with improved loss functions and training stability features.

Features:
- Multi-scale robust loss (MSE + MAE + Huber)
- Perceptual loss for semantic understanding
- Consistency regularization across augmentations
- Enhanced monitoring and logging
- Backward compatibility with original training

Author: SiHun Lee, Ph.D.
Email: kevin1007kr@gmail.com
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import math
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary

# Import enhanced loss functions
from modules.enhanced_loss_functions import (
    MultiScaleRobustLoss,
    PerceptualLatentLoss, 
    ConsistencyLoss,
    EnhancedLossConfig,
    create_enhanced_loss_system,
    get_preset_config
)

# Import existing utilities
from modules.latent_conditioner import (
    apply_outline_preserving_augmentations,
    setup_device_and_model,
    setup_optimizer_and_scheduler,
    safe_initialize_weights_He
)


def enhanced_train_latent_conditioner(
    latent_conditioner_epoch, 
    latent_conditioner_dataloader,
    latent_conditioner_validation_dataloader, 
    latent_conditioner, 
    latent_conditioner_lr,
    weight_decay=1e-4,
    is_image_data=True,
    image_size=256,
    enhancement_config=None,
    main_latent_dim=32,
    hier_latent_dim=8,
    size2=200
):
    """Enhanced training for latent conditioner with improved loss functions.
    
    Args:
        latent_conditioner_epoch (int): Number of training epochs
        latent_conditioner_dataloader: Training data loader
        latent_conditioner_validation_dataloader: Validation data loader
        latent_conditioner: Latent conditioner model
        latent_conditioner_lr (float): Learning rate
        weight_decay (float): Weight decay for optimizer (default: 1e-4)
        is_image_data (bool): Whether input data is images (default: True)
        image_size (int): Size of input images (default: 256)
        enhancement_config: EnhancedLossConfig instance (default: balanced_config)
        main_latent_dim (int): Dimension of main latent (default: 32)
        hier_latent_dim (int): Dimension of hierarchical latent (default: 8)
        size2 (int): Size2 parameter for hierarchical latent (default: 200)
    
    Returns:
        float: Final average validation loss
    """
    
    # Use balanced config as default if none provided
    if enhancement_config is None:
        enhancement_config = EnhancedLossConfig.balanced_config()
        print("Using default 'balanced' enhancement configuration")
    
    print(f"Enhanced training configuration:")
    print(f"  Multi-scale loss: {enhancement_config.use_multiscale_loss}")
    print(f"  Perceptual loss: {enhancement_config.use_perceptual_loss} (weight: {enhancement_config.perceptual_weight})")
    print(f"  Consistency loss: {enhancement_config.use_consistency_loss} (weight: {enhancement_config.consistency_weight})")
    
    # Initialize enhanced TensorBoard logging
    writer = SummaryWriter(log_dir='./EnhancedLatentConditionerRuns', 
                          comment='Enhanced_LatentConditioner')
    
    # Setup device and model 
    latent_conditioner, device = setup_device_and_model(latent_conditioner)
    
    print(f"Training on device: {device}")
    
    # Setup optimizer and scheduler
    optimizer, warmup_scheduler, main_scheduler, warmup_epochs = setup_optimizer_and_scheduler(
        latent_conditioner, latent_conditioner_lr, weight_decay, latent_conditioner_epoch
    )
    
    # Initialize enhanced loss system
    print("Initializing enhanced loss system...")
    loss_system = create_enhanced_loss_system(
        config=enhancement_config,
        main_latent_dim=main_latent_dim,
        hier_latent_dim=hier_latent_dim,
        device=device
    )
    
    print(f"Active loss components: {loss_system['active_components']}")
    
    # Training state tracking
    best_val_loss = float('inf')
    patience = 100000  # Effectively disabled
    patience_counter = 0
    min_delta = 1e-8
    overfitting_threshold = 1000.0
    
    # Move model to device and initialize weights
    latent_conditioner = latent_conditioner.to(device)
    latent_conditioner.apply(safe_initialize_weights_He)
    
    model_summary_shown = False
    
    # Training metrics tracking
    training_metrics = {
        'epoch_losses': [],
        'enhanced_loss_components': {comp: [] for comp in loss_system['active_components']},
        'validation_losses': [],
        'learning_rates': []
    }
    
    # Initialize validation MSE tracking
    last_mse_main_val = 0.0
    last_mse_hier_val = 0.0
    
    print(f"Starting enhanced training for {latent_conditioner_epoch} epochs...")
    
    for epoch in range(latent_conditioner_epoch):
        start_time = time.time()
        latent_conditioner.train(True)
        
        # Initialize epoch metrics
        epoch_metrics = {
            'total_loss': 0.0,
            'num_batches': 0,
            'enhanced_components': {comp: 0.0 for comp in loss_system['active_components']},
            # Track pure MSE losses for display (like original training)
            'mse_main': 0.0,
            'mse_hier': 0.0
        }
        
        for i, (x, y1, y2) in enumerate(latent_conditioner_dataloader):
            # Move data to device
            if x.device != device:
                x, y1, y2 = x.to(device, non_blocking=True), y1.to(device, non_blocking=True), y2.to(device, non_blocking=True)
            
            # Show model summary on first batch
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
            
            # Store original input for consistency loss
            x_original = None
            if enhancement_config.use_consistency_loss:
                x_original = x.clone()
            
            # Apply data augmentations (existing augmentation pipeline)
            if is_image_data and torch.rand(1, device=x.device) < 0.5:
                im_size = int(math.sqrt(x.shape[-1]))
                x_2d = x.reshape(-1, im_size, im_size)
                x_2d = apply_outline_preserving_augmentations(x_2d, prob=0.8)
                x = x_2d.reshape(x.shape[0], -1)
            
            # Mixup augmentation (existing)
            if torch.rand(1, device=x.device) < 0.02 and x.size(0) > 1:
                alpha = 0.2
                lam = torch.tensor(np.random.beta(alpha, alpha), device=x.device, dtype=x.dtype)
                batch_size = x.size(0)
                index = torch.randperm(batch_size, device=x.device)
                
                x = lam * x + (1 - lam) * x[index, :]
                y1 = lam * y1 + (1 - lam) * y1[index, :]
                y2 = lam * y2 + (1 - lam) * y2[index, :]
            
            # Noise injection (existing)
            if torch.rand(1, device=x.device) < 0.05:
                noise = torch.randn_like(x) * 0.01
                x = x + noise
            
            # Zero gradients
            optimizer.zero_grad(set_to_none=True)
            
            # Forward pass
            y_pred1, y_pred2 = latent_conditioner(x)
            
            # Track pure MSE losses for display (like original training)
            mse_main_train = F.mse_loss(y_pred1, y1)
            mse_hier_train = F.mse_loss(y_pred2, y2)
            epoch_metrics['mse_main'] += mse_main_train.item()
            epoch_metrics['mse_hier'] += mse_hier_train.item()
            
            # Enhanced loss computation
            losses = []
            all_loss_info = {}
            
            # 1. Multi-scale robust loss (replaces simple MSE)
            if 'multiscale_loss' in loss_system:
                multiscale_criterion = loss_system['multiscale_loss']
                robust_loss, robust_info = multiscale_criterion(y_pred1, y_pred2, y1, y2)
                losses.append(robust_loss)
                all_loss_info.update(robust_info)
                epoch_metrics['enhanced_components']['multiscale_loss'] += robust_loss.item()
            else:
                # Fallback to original MSE if multiscale is disabled
                A = nn.MSELoss()(y_pred1, y1)
                B = nn.MSELoss()(y_pred2, y2)
                fallback_loss = A * 10 + B
                losses.append(fallback_loss)
                all_loss_info.update({'loss_main': A.item(), 'loss_hier': B.item()})
            
            # 2. Perceptual loss (semantic understanding)
            if 'perceptual_loss' in loss_system:
                perceptual_criterion = loss_system['perceptual_loss']
                perceptual_loss, perceptual_info = perceptual_criterion(y_pred1, y_pred2, y1, y2)
                weighted_perceptual = enhancement_config.perceptual_weight * perceptual_loss
                losses.append(weighted_perceptual)
                all_loss_info.update(perceptual_info)
                epoch_metrics['enhanced_components']['perceptual_loss'] += weighted_perceptual.item()
            
            # 3. Consistency loss (augmentation stability)
            if 'consistency_loss' in loss_system and x_original is not None:
                consistency_criterion = loss_system['consistency_loss']
                consistency_loss, consistency_info = consistency_criterion(
                    latent_conditioner, x_original, x
                )
                losses.append(consistency_loss)
                all_loss_info.update(consistency_info)
                epoch_metrics['enhanced_components']['consistency_loss'] += consistency_loss.item()
            
            # Combine all losses (simple sum, no adaptive weighting)
            total_loss = sum(losses)
            
            # Backward pass and optimization
            total_loss.backward()
            
            # Gradient clipping for stability
            total_grad_norm = torch.nn.utils.clip_grad_norm_(
                latent_conditioner.parameters(), max_norm=10.0
            )
            
            # Gradient monitoring (every 100 epochs, first batch) - match original format
            if epoch % 100 == 0 and i == 0:
                print(f"DEBUG: Gradient norm: {total_grad_norm:.4f}, Loss: {total_loss.item():.4E}")
                if total_grad_norm > 10.0:
                    print(f"WARNING: Large gradient norm detected: {total_grad_norm:.2f}")
                elif total_grad_norm < 1e-4:
                    print(f"WARNING: Very small gradient norm: {total_grad_norm:.2E}")
            
            optimizer.step()
            
            # Accumulate epoch metrics
            epoch_metrics['total_loss'] += total_loss.item()
            epoch_metrics['num_batches'] += 1
            
            # Clean up large tensors
            del y_pred1, y_pred2, x
            if x_original is not None:
                del x_original
        
        # Calculate average training metrics
        if epoch_metrics['num_batches'] > 0:
            avg_train_loss = epoch_metrics['total_loss'] / epoch_metrics['num_batches']
            # Calculate pure MSE averages (like original training)
            avg_mse_main_train = epoch_metrics['mse_main'] / epoch_metrics['num_batches']
            avg_mse_hier_train = epoch_metrics['mse_hier'] / epoch_metrics['num_batches']
            for comp in epoch_metrics['enhanced_components']:
                epoch_metrics['enhanced_components'][comp] /= epoch_metrics['num_batches']
        else:
            avg_train_loss = 0.0
            avg_mse_main_train = 0.0
            avg_mse_hier_train = 0.0
        
        training_metrics['epoch_losses'].append(avg_train_loss)
        for comp in loss_system['active_components']:
            training_metrics['enhanced_loss_components'][comp].append(
                epoch_metrics['enhanced_components'].get(comp, 0.0)
            )
        
        # Validation every 10 epochs
        avg_val_loss = 0.0
        avg_mse_main_val = 0.0
        avg_mse_hier_val = 0.0
        if epoch % 10 == 0:
            latent_conditioner.eval()
            val_metrics = {'total_loss': 0.0, 'num_batches': 0, 'mse_main': 0.0, 'mse_hier': 0.0}
            
            try:
                with torch.no_grad():
                    for val_batch_idx, (x_val, y1_val, y2_val) in enumerate(latent_conditioner_validation_dataloader):
                        x_val, y1_val, y2_val = x_val.to(device), y1_val.to(device), y2_val.to(device)
                        
                        y_pred1_val, y_pred2_val = latent_conditioner(x_val)
                        
                        # Track pure MSE for validation (like original training)
                        mse_main_val = F.mse_loss(y_pred1_val, y1_val)
                        mse_hier_val = F.mse_loss(y_pred2_val, y2_val)
                        val_metrics['mse_main'] += mse_main_val.item()
                        val_metrics['mse_hier'] += mse_hier_val.item()
                        
                        # Use same enhanced loss for validation
                        val_losses = []
                        
                        if 'multiscale_loss' in loss_system:
                            val_robust_loss, _ = loss_system['multiscale_loss'](
                                y_pred1_val, y_pred2_val, y1_val, y2_val
                            )
                            val_losses.append(val_robust_loss)
                        else:
                            A_val = nn.MSELoss()(y_pred1_val, y1_val)
                            B_val = nn.MSELoss()(y_pred2_val, y2_val)
                            val_losses.append(A_val * 10 + B_val)
                        
                        if 'perceptual_loss' in loss_system:
                            val_perceptual_loss, _ = loss_system['perceptual_loss'](
                                y_pred1_val, y_pred2_val, y1_val, y2_val
                            )
                            val_losses.append(enhancement_config.perceptual_weight * val_perceptual_loss)
                        
                        # Note: Skip consistency loss in validation (no augmentation)
                        
                        val_total_loss = sum(val_losses)
                        val_metrics['total_loss'] += val_total_loss.item()
                        val_metrics['num_batches'] += 1
                        
                        del x_val, y_pred1_val, y_pred2_val
                
                if val_metrics['num_batches'] > 0:
                    avg_val_loss = val_metrics['total_loss'] / val_metrics['num_batches']
                    avg_mse_main_val = val_metrics['mse_main'] / val_metrics['num_batches']
                    avg_mse_hier_val = val_metrics['mse_hier'] / val_metrics['num_batches']
                
                # Store current validation MSE values for reuse
                last_mse_main_val = avg_mse_main_val
                last_mse_hier_val = avg_mse_hier_val
                
                # Check for overfitting
                overfitting_ratio = avg_val_loss / max(avg_train_loss, 1e-8)
                if overfitting_ratio > overfitting_threshold:
                    print(f'Enhanced training: Severe overfitting detected! Val/Train ratio: {overfitting_ratio:.1f}')
                    print(f'Stopping early at epoch {epoch}')
                    break
                
                # Track best validation loss
                if avg_val_loss < best_val_loss - min_delta:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
                    
            except Exception as e:
                print(f"Enhanced validation error at epoch {epoch}: {e}")
                if epoch > 0:
                    avg_val_loss = training_metrics['validation_losses'][-1] if training_metrics['validation_losses'] else 0.0
        else:
            # Use previous validation loss for non-validation epochs
            if training_metrics['validation_losses']:
                avg_val_loss = training_metrics['validation_losses'][-1]
                # Use last known validation MSE values
                avg_mse_main_val = last_mse_main_val
                avg_mse_hier_val = last_mse_hier_val
        
        training_metrics['validation_losses'].append(avg_val_loss)
        
        # Learning rate scheduling
        if epoch < warmup_epochs:
            warmup_scheduler.step()
        else:
            main_scheduler.step()
        
        current_lr = optimizer.param_groups[0]['lr']
        training_metrics['learning_rates'].append(current_lr)
        
        # Enhanced logging and monitoring
        end_time = time.time()
        epoch_duration = end_time - start_time
        
        # Console output - match original latent_conditioner.py format exactly
        scheduler_info = "Warmup" if epoch < warmup_epochs else "Cosine"
        
        # Calculate total MSE losses (weighted like original: main*10 + hier*1)
        total_mse_train = avg_mse_main_train * 10 + avg_mse_hier_train
        total_mse_val = avg_mse_main_val * 10 + avg_mse_hier_val
        
        print('[%d/%d]\tTrain: %.4E (y1:%.4E, y2:%.4E), Val: %.4E (y1:%.4E, y2:%.4E), LR: %.2E (%s), ETA: %.2f h, Patience: %d/%d' % 
              (epoch, latent_conditioner_epoch, 
               total_mse_train, avg_mse_main_train, avg_mse_hier_train,
               total_mse_val, avg_mse_main_val, avg_mse_hier_val,
               current_lr, scheduler_info,
               (latent_conditioner_epoch-epoch)*epoch_duration/3600, patience_counter, patience))
        
        # Enhanced debug output (every 100 epochs only)
        if epoch % 100 == 0:
            print(f'Enhanced DEBUG: Multi-scale total loss: {avg_train_loss:.4E}')
            enhanced_components = []
            for comp in loss_system['active_components']:
                comp_value = epoch_metrics['enhanced_components'].get(comp, 0.0)
                enhanced_components.append(f"{comp}: {comp_value:.4E}")
            if enhanced_components:
                print(f'Enhanced DEBUG: {" | ".join(enhanced_components)}')
        
        # Enhanced TensorBoard logging
        if epoch % 100 == 0:
            writer.add_scalar('Enhanced_Loss/Total_Train', avg_train_loss, epoch)
            writer.add_scalar('Enhanced_Loss/Total_Validation', avg_val_loss, epoch)
            writer.add_scalar('Enhanced_Training/Learning_Rate', current_lr, epoch)
            
            # Log individual loss components
            for comp in loss_system['active_components']:
                comp_value = epoch_metrics['enhanced_components'].get(comp, 0.0)
                writer.add_scalar(f'Enhanced_Loss_Components/{comp}', comp_value, epoch)
            
            # Log detailed loss info if available
            for key, value in all_loss_info.items():
                if isinstance(value, (int, float)):
                    writer.add_scalar(f'Enhanced_Loss_Details/{key}', value, epoch)
        
        # Early stopping check
        if patience_counter >= patience and epoch > latent_conditioner_epoch * 0.5:
            print(f'Enhanced training: Early stopping at epoch {epoch}. Best val loss: {best_val_loss:.4E}')
            break
    
    # Save enhanced model
    print("Saving enhanced latent conditioner model...")
    torch.save(latent_conditioner.state_dict(), 'checkpoints/enhanced_latent_conditioner.pth')
    torch.save(latent_conditioner, 'model_save/LatentConditioner')
    
    # Save training metrics
    np.savez('output/enhanced_training_metrics.npz', 
             epoch_losses=training_metrics['epoch_losses'],
             validation_losses=training_metrics['validation_losses'],
             learning_rates=training_metrics['learning_rates'],
             **training_metrics['enhanced_loss_components'])
    
    writer.close()
    
    print(f"Enhanced training completed. Final validation loss: {avg_val_loss:.6f}")
    print(f"Best validation loss achieved: {best_val_loss:.6f}")
    print(f"Active enhancement components: {loss_system['active_components']}")
    
    return avg_val_loss


def train_latent_conditioner_with_enhancements(
    latent_conditioner_epoch, 
    latent_conditioner_dataloader,
    latent_conditioner_validation_dataloader, 
    latent_conditioner, 
    latent_conditioner_lr,
    weight_decay=1e-4,
    is_image_data=True,
    image_size=256,
    enhancement_preset='balanced',
    use_enhanced_loss=True
):
    """Wrapper function that chooses between enhanced and original training.
    
    This function provides backward compatibility while allowing enhanced training.
    
    Args:
        All standard training arguments, plus:
        enhancement_preset (str): Preset configuration name ('balanced', 'robust', etc.)
        use_enhanced_loss (bool): Whether to use enhanced training (default: True)
    
    Returns:
        float: Final validation loss
    """
    
    if use_enhanced_loss:
        print(f"Using enhanced training with preset: {enhancement_preset}")
        
        try:
            # Get preset configuration
            enhancement_config = get_preset_config(enhancement_preset)
            
            # Use enhanced training
            return enhanced_train_latent_conditioner(
                latent_conditioner_epoch=latent_conditioner_epoch,
                latent_conditioner_dataloader=latent_conditioner_dataloader,
                latent_conditioner_validation_dataloader=latent_conditioner_validation_dataloader,
                latent_conditioner=latent_conditioner,
                latent_conditioner_lr=latent_conditioner_lr,
                weight_decay=weight_decay,
                is_image_data=is_image_data,
                image_size=image_size,
                enhancement_config=enhancement_config
            )
            
        except Exception as e:
            print(f"Enhanced training failed: {e}")
            print("Falling back to original training...")
            use_enhanced_loss = False
    
    if not use_enhanced_loss:
        print("Using original training (enhanced features disabled)")
        
        # Import and use original training function
        from modules.latent_conditioner import train_latent_conditioner
        
        return train_latent_conditioner(
            latent_conditioner_epoch=latent_conditioner_epoch,
            latent_conditioner_dataloader=latent_conditioner_dataloader,
            latent_conditioner_validation_dataloader=latent_conditioner_validation_dataloader,
            latent_conditioner=latent_conditioner,
            latent_conditioner_lr=latent_conditioner_lr,
            weight_decay=weight_decay,
            is_image_data=is_image_data,
            image_size=image_size
        )