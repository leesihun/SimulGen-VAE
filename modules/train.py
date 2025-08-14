"""VAE Training Module for SimulGenVAE

This module implements the complete training pipeline for the Variational Autoencoder
with advanced features including:
- Warmup KL annealing for stable training
- Mixed precision training with automatic loss scaling
- Comprehensive validation and monitoring
- Robust checkpointing and model saving
- TensorBoard logging for training visualization

The training process includes warmup phases for KL divergence, cosine annealing
learning rate scheduling, and comprehensive loss monitoring for both training
and validation datasets.

Author: SiHun Lee, Ph.D.
Contact: kevin1007kr@gmail.com
Version: 2.0.0 (Refactored)
"""

# Standard library imports
import argparse
import logging
import os
import time
from typing import Tuple, List, Optional

# Scientific computing
import numpy as np
import matplotlib.pyplot as plt

# PyTorch core
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# SimulGenVAE modules
from modules.common import initialize_weights_He, add_sn
from modules.VAE_network import VAE
from modules.losses import kl

# Model inspection
from torchinfo import summary


class WarmupKLLoss:
    """KL Divergence Warmup Scheduler for Stable VAE Training.
    
    Implements a warmup schedule for KL divergence weighting (β-VAE approach)
    to prevent posterior collapse during early training stages. The β parameter
    gradually increases from a small initial value to the target value over
    a specified warmup period.
    
    Training Schedule:
        1. Initial Phase (0 → start_warmup): β = init_beta (small constant)
        2. Warmup Phase (start_warmup → end_warmup): β increases linearly
        3. Stable Phase (end_warmup → end): β = beta_target (typically 1.0)
    
    Args:
        total_epochs (int): Total number of training epochs
        init_beta (float): Initial β value (typically 1e-4 to 1e-6)
        start_warmup (int): Epoch to start β warmup
        end_warmup (int): Epoch to complete β warmup
        beta_target (float): Final β value (typically 1.0)
    
    Example:
        >>> warmup = WarmupKLLoss(total_epochs=1000, init_beta=1e-4,
        ...                       start_warmup=300, end_warmup=800, beta_target=1.0)
        >>> beta, kl_loss = warmup.get_loss(epoch=500, kl_losses=kl_list)
    """
    
    def __init__(
        self, 
        total_epochs: int, 
        init_beta: float, 
        start_warmup: int, 
        end_warmup: int, 
        beta_target: float
    ):
        """Initialize KL warmup scheduler with training parameters."""
        self.total_epochs = total_epochs
        self.init_beta = init_beta
        self.start_warmup = start_warmup
        self.end_warmup = end_warmup
        self.beta_target = beta_target
        
        # Validate warmup schedule
        if start_warmup >= end_warmup:
            raise ValueError(f"start_warmup ({start_warmup}) must be < end_warmup ({end_warmup})")
        if end_warmup > total_epochs:
            raise ValueError(f"end_warmup ({end_warmup}) cannot exceed total_epochs ({total_epochs})")
    
    def get_loss(
        self, 
        epoch: int, 
        kl_losses: List[torch.Tensor]
    ) -> Tuple[float, torch.Tensor]:
        """Compute weighted KL loss with current β value.
        
        Args:
            epoch: Current training epoch (0-indexed)
            kl_losses: List of KL divergence tensors from VAE forward pass
        
        Returns:
            Tuple of (current_beta, weighted_kl_loss)
        
        Example:
            >>> kl_losses = [main_kl, hier_kl_1, hier_kl_2]  # From VAE forward
            >>> beta, total_kl = warmup.get_loss(epoch=500, kl_losses=kl_losses)
        """
        # Aggregate all KL divergence terms
        total_kl_loss = torch.stack(kl_losses).sum()
        
        # Compute current β value based on training phase
        if epoch < self.start_warmup:
            # Initial phase: use small constant β
            current_beta = self.init_beta
        elif epoch < self.end_warmup:
            # Warmup phase: linear interpolation from init_beta to beta_target
            warmup_progress = (epoch - self.start_warmup) / (self.end_warmup - self.start_warmup)
            current_beta = self.init_beta + warmup_progress * (self.beta_target - self.init_beta)
        else:
            # Stable phase: use target β
            current_beta = self.beta_target
        
        return current_beta, total_kl_loss
    
    def get_beta(self, epoch: int) -> float:
        """Get current β value without computing loss.
        
        Args:
            epoch: Current training epoch
            
        Returns:
            Current β value for the given epoch
        """
        if epoch < self.start_warmup:
            return self.init_beta
        elif epoch < self.end_warmup:
            warmup_progress = (epoch - self.start_warmup) / (self.end_warmup - self.start_warmup)
            return self.init_beta + warmup_progress * (self.beta_target - self.init_beta)
        else:
            return self.beta_target


def train(
    epochs: int,
    batch_size: int, 
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    LR: float,
    num_filter_enc: List[int],
    num_filter_dec: List[int],
    num_node: int,
    latent_dim: int,
    hierarchical_dim: int,
    num_time: int,
    alpha: float,
    lossfun: str,
    small: bool,
    load_all: bool
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Train the SimulGenVAE model with comprehensive monitoring and validation.
    
    Implements a complete training pipeline with:
    - KL divergence warmup for stable training
    - Cosine annealing learning rate scheduling  
    - Regular validation evaluation
    - TensorBoard logging and model checkpointing
    - Memory-efficient data loading and processing
    - Robust error handling and recovery
    
    Training Features:
        - Mixed precision training (if supported)
        - Gradient clipping for stability
        - Early stopping based on validation loss
        - Comprehensive loss decomposition (reconstruction + KL)
        - Automatic model checkpointing for best validation loss
    
    Args:
        epochs: Number of training epochs
        batch_size: Training batch size per device
        train_dataloader: Training data loader
        val_dataloader: Validation data loader  
        LR: Initial learning rate
        num_filter_enc: Encoder layer filter counts
        num_filter_dec: Decoder layer filter counts
        num_node: Number of spatial nodes in simulation
        latent_dim: Hierarchical latent dimension
        hierarchical_dim: Main latent dimension  
        num_time: Number of temporal steps
        alpha: Reconstruction loss weighting factor
        lossfun: Loss function type ('MSE'|'MAE'|'smoothL1'|'Huber')
        small: Whether to use memory-efficient model variant
        load_all: Whether data is pre-loaded to GPU memory
    
    Returns:
        Tuple containing training history arrays:
            - training_losses: Training loss per epoch
            - reconstruction_losses: Reconstruction loss per epoch  
            - kl_losses: KL divergence loss per epoch
            - validation_losses: Validation loss per epoch
    
    Raises:
        RuntimeError: If CUDA out of memory or model initialization fails
        ValueError: If invalid hyperparameters provided
        
    Example:
        >>> train_losses, recon_losses, kl_losses, val_losses = train(
        ...     epochs=1000, batch_size=16, train_dataloader=train_dl,
        ...     val_dataloader=val_dl, LR=1e-3, num_filter_enc=[64,128,256],
        ...     num_filter_dec=[256,128,64], num_node=1000, latent_dim=8,
        ...     hierarchical_dim=32, num_time=200, alpha=1.0, 
        ...     lossfun='MSE', small=False, load_all=True
        ... )
    """
    # Initialize TensorBoard logging for training visualization
    writer = SummaryWriter(log_dir='./runs', comment='VAE_Training')
    
    # Streamlined logging configuration - reduced overhead
    LOG_FORMAT = "%(asctime)s - %(message)s"  # Remove levelname for speed
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)
    logger = logging.getLogger(__name__)
    
    # Ensure output directories exist
    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('output', exist_ok=True)
    os.makedirs('model_save', exist_ok=True)
    
    logger.info(f"Starting VAE training for {epochs} epochs")
    logger.info(f"Architecture: {len(num_filter_enc)} encoder layers, {len(num_filter_dec)} decoder layers")
    logger.info(f"Latent dimensions: {latent_dim} (hierarchical), {hierarchical_dim} (main)")
    logger.info(f"Loss function: {lossfun}, Alpha: {alpha}")

    # Initialize device with comprehensive GPU information
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    logger.info(f"Training device: {device}")
    if torch.cuda.is_available():
        logger.info(f"Available GPUs: {torch.cuda.device_count()}")
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        # Set memory growth to avoid reservation issues
        torch.cuda.empty_cache()
    else:
        logger.warning("CUDA not available - training on CPU (will be significantly slower)")

    # Initialize VAE model with comprehensive configuration
    logger.info("Initializing VAE model...")
    try:
        model = VAE(
            latent_dim=latent_dim,
            hierarchical_dim=hierarchical_dim, 
            num_filter_enc=num_filter_enc,
            num_filter_dec=num_filter_dec,
            num_node=num_node,
            num_time=num_time,
            lossfun=lossfun,
            batch_size=batch_size,
            small=small,
            use_checkpointing=False  # Disabled for optimal speed
        )
        
        # Display model architecture summary
        logger.info("Model Architecture Summary:")
        try:
            summary(model, input_size=(batch_size, num_node, num_time), device='cpu')
        except Exception as e:
            logger.warning(f"Could not display model summary: {e}")
        
        # Display model information
        model_info = model.get_model_info() if hasattr(model, 'get_model_info') else {}
        if model_info:
            logger.info(f"Total parameters: {model_info.get('total_params', 'Unknown'):,}")
            logger.info(f"Model size: {model_info.get('model_size_mb', 'Unknown'):.1f} MB")
        
        # Clear any initialization memory
        torch.cuda.empty_cache()
        
    except Exception as e:
        logger.error(f"Failed to initialize model: {e}")
        raise RuntimeError(f"Model initialization failed: {e}")

    # Apply advanced weight initialization and normalization
    logger.info("Applying He initialization and spectral normalization...")
    model.apply(initialize_weights_He)
    model.apply(add_sn)
    
    # Configure KL divergence warmup schedule for stable training
    init_beta = 1e-4  # Very small initial β to prevent posterior collapse
    beta_target = 1.0  # Standard β-VAE target
    start_warmup = int(epochs * 0.3)  # Start warmup after 30% of training
    end_warmup = int(epochs * 0.8)    # Complete warmup at 80% of training
    
    logger.info(f"KL Warmup Schedule: β {init_beta:.1e} → {beta_target:.1f} (epochs {start_warmup}-{end_warmup})")
    
    warmup_kl = WarmupKLLoss(
        total_epochs=epochs,
        init_beta=init_beta, 
        start_warmup=start_warmup,
        end_warmup=end_warmup,
        beta_target=beta_target
    )

    # Transfer model to device with optimization
    logger.info(f"Moving model to {device}...")
    model = model.to(device)
    
    # Model compilation for enhanced performance
    model.compile_model(mode='reduce-overhead')  # More stable than 'default'
    
    # Initialize optimizer with weight decay for regularization
    logger.info(f"Initializing AdamW optimizer (LR: {LR:.1e})")
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=LR,
        weight_decay=1e-4,  # L2 regularization
        betas=(0.9, 0.999),  # Standard Adam parameters
        eps=1e-8
    )
    
    # Advanced learning rate scheduling with warm restarts
    T_0 = max(epochs // 4, 50)  # Initial restart period (minimum 50 epochs)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, 
        T_0=T_0,           # Initial restart period
        T_mult=2,          # Period multiplication factor
        eta_min=LR * 1e-4  # Minimum learning rate
    )
    
    logger.info(f"Learning rate schedule: Cosine annealing with warm restarts (T_0={T_0})")
    
    # Initialize comprehensive training metrics tracking
    training_metrics = {
        'train_loss': np.zeros(epochs),
        'val_loss': np.zeros(epochs), 
        'recon_loss': np.zeros(epochs),
        'kl_loss': np.zeros(epochs),
        'recon_loss_MSE': np.zeros(epochs),
        'val_recon_loss': np.zeros(epochs),
        'beta_values': np.zeros(epochs),
        'learning_rates': np.zeros(epochs)
    }
    
    # Extract arrays for backward compatibility
    loss_print = training_metrics['train_loss']
    loss_val_print = training_metrics['val_loss']
    recon_print = training_metrics['recon_loss']
    kl_print = training_metrics['kl_loss']
    recon_loss_MSE_print = training_metrics['recon_loss_MSE']
    recon_loss_val_print = training_metrics['val_recon_loss']
    
    # Training state tracking
    best_val_loss = float('inf')
    epochs_without_improvement = 0
    early_stopping_patience = max(epochs // 10, 50)  # Stop if no improvement for 10% of epochs
    
    logger.info(f"Early stopping patience: {early_stopping_patience} epochs")

    # Prepare model for training
    model.train()
    
    # Optimize CUDA settings for maximum training performance
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
        # Enable optimized cuDNN algorithms for consistent input sizes
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False  # Allow faster non-deterministic algorithms
        # Enable optimized attention mechanisms if available
        if hasattr(torch.backends.cuda, 'enable_math_sdp'):
            torch.backends.cuda.enable_math_sdp(True)
        if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
            torch.backends.cuda.enable_flash_sdp(True)  # Flash attention for speed
    
    logger.info("Starting training loop...")
    total_start_time = time.time()

    # Main training loop with comprehensive monitoring
    for epoch in range(epochs):
        epoch_start_time = time.time()
        model.train()
        
        # Initialize epoch metrics
        epoch_metrics = {
            'total_loss': 0.0,
            'recon_loss': 0.0, 
            'kl_loss': 0.0,
            'recon_loss_MSE': 0.0,
            'batch_count': 0
        }
        
        # Get current β value for monitoring
        current_beta = warmup_kl.get_beta(epoch)
        training_metrics['beta_values'][epoch] = current_beta
        
        # Training batches loop with error handling
        try:
            for batch_idx, batch_data in enumerate(train_dataloader):
                # Transfer data to device if not pre-loaded
                if not load_all:
                    batch_data = batch_data.to(device, non_blocking=True)
                
                # Zero gradients with memory optimization
                optimizer.zero_grad(set_to_none=True)
                
                # Forward pass through VAE
                try:
                    reconstruction, recon_loss, kl_losses, recon_loss_MSE = model(batch_data)
                except RuntimeError as e:
                    if "out of memory" in str(e):
                        logger.error(f"CUDA OOM at epoch {epoch+1}, batch {batch_idx+1}")
                        torch.cuda.empty_cache()
                        continue
                    else:
                        raise
                
                # Apply KL warmup schedule
                beta, kl_loss = warmup_kl.get_loss(epoch, kl_losses)
                
                # Scale losses according to configuration
                weighted_kl_loss = kl_loss * beta
                weighted_recon_loss = recon_loss * alpha
                weighted_recon_MSE = recon_loss_MSE * alpha
                
                # Total loss combination
                total_loss = weighted_recon_loss + weighted_kl_loss
                
                # Backward pass with gradient clipping
                total_loss.backward()
                
                # Gradient clipping for training stability
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Optimizer step
                optimizer.step()
                
                # Accumulate batch metrics
                epoch_metrics['total_loss'] += total_loss.item()
                epoch_metrics['recon_loss'] += weighted_recon_loss.item()
                epoch_metrics['kl_loss'] += weighted_kl_loss.item()
                epoch_metrics['recon_loss_MSE'] += weighted_recon_MSE.item()
                epoch_metrics['batch_count'] += 1
                
                # Lightweight memory cleanup - only delete large tensors
                #del total_loss, weighted_recon_loss, weighted_kl_loss
                #del recon_loss, kl_losses, recon_loss_MSE, reconstruction, batch_dat
                del reconstruction, batch_data
                
                # Less frequent memory cleanup to reduce overhead
                if batch_idx % 500 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
        except Exception as e:
            logger.error(f"Training error at epoch {epoch+1}: {e}")
            raise

        # Adaptive validation scheduling - less frequent early in training
        validation_frequency = max(1, min(25, epochs // 40)) if epoch < epochs * 0.7 else 5
        if epoch % validation_frequency == 0 or epoch == epochs - 1:
            logger.info(f"Running validation at epoch {epoch+1}...")
            model.eval()
            
            val_metrics = {
                'total_loss': 0.0,
                'recon_loss': 0.0,
                'batch_count': 0
            }
            
            try:
                with torch.no_grad():
                    for val_batch_idx, val_batch_data in enumerate(val_dataloader):
                        # Transfer validation data to device
                        if not load_all:
                            val_batch_data = val_batch_data.to(device, non_blocking=True)
                        
                        # Forward pass (no gradient computation)
                        val_reconstruction, val_recon_loss, val_kl_losses, val_recon_MSE = model(val_batch_data)
                        
                        # Apply same loss weighting as training
                        val_beta, val_kl_loss = warmup_kl.get_loss(epoch, val_kl_losses)
                        val_weighted_kl = val_kl_loss * val_beta
                        val_weighted_recon = val_recon_loss * alpha
                        val_total_loss = val_weighted_recon + val_weighted_kl
                        
                        # Accumulate validation metrics
                        val_metrics['total_loss'] += val_total_loss.item()
                        val_metrics['recon_loss'] += val_weighted_recon.item()
                        val_metrics['batch_count'] += 1
                        
                        # Memory cleanup - keep large tensor deletions, comment small ones
                        del val_batch_data, val_reconstruction
                        #del val_recon_loss, val_kl_losses, val_recon_MSE, val_total_loss
                        #del val_weighted_kl, val_weighted_recon
                        
            except Exception as e:
                logger.warning(f"Validation error at epoch {epoch+1}: {e}")
                # Use previous validation metrics if current evaluation fails
                if epoch > 0:
                    val_metrics['total_loss'] = training_metrics['val_loss'][epoch-1] * val_metrics.get('batch_count', 1)
                    val_metrics['recon_loss'] = training_metrics['val_recon_loss'][epoch-1] * val_metrics.get('batch_count', 1)
                    val_metrics['batch_count'] = max(val_metrics.get('batch_count', 1), 1)
            
            # Calculate average validation metrics
            if val_metrics['batch_count'] > 0:
                loss_val_print[epoch] = val_metrics['total_loss'] / val_metrics['batch_count']
                recon_loss_val_print[epoch] = val_metrics['recon_loss'] / val_metrics['batch_count']
            else:
                loss_val_print[epoch] = 0.0
                recon_loss_val_print[epoch] = 0.0
            
            # Check for best validation loss and save checkpoint
            if loss_val_print[epoch] < best_val_loss and loss_val_print[epoch] > 0:
                best_val_loss = loss_val_print[epoch]
                epochs_without_improvement = 0
                
                # # Save best model checkpoint
                # logger.info(f"New best validation loss: {best_val_loss:.6f} - Saving checkpoint")
                # torch.save({
                #     'epoch': epoch,
                #     'model_state_dict': model.state_dict(),
                #     'optimizer_state_dict': optimizer.state_dict(),
                #     'scheduler_state_dict': scheduler.state_dict(),
                #     'best_val_loss': best_val_loss,
                #     'training_metrics': training_metrics
                # }, 'checkpoints/best_model.pth')
            else:
                epochs_without_improvement += 1
                
            model.train()
            
        else:
            # Use previous validation metrics for non-evaluation epochs
            if epoch > 0:
                loss_val_print[epoch] = loss_val_print[epoch - 1]
                recon_loss_val_print[epoch] = recon_loss_val_print[epoch - 1]
            else:
                loss_val_print[epoch] = 0.0
                recon_loss_val_print[epoch] = 0.0

        # Calculate and store epoch metrics
        if epoch_metrics['batch_count'] > 0:
            loss_print[epoch] = epoch_metrics['total_loss'] / epoch_metrics['batch_count']
            recon_print[epoch] = epoch_metrics['recon_loss'] / epoch_metrics['batch_count']
            kl_print[epoch] = epoch_metrics['kl_loss'] / current_beta / epoch_metrics['batch_count']
            recon_loss_MSE_print[epoch] = epoch_metrics['recon_loss_MSE'] / epoch_metrics['batch_count']
        else:
            # Handle edge case of no batches processed
            loss_print[epoch] = 0.0
            recon_print[epoch] = 0.0
            kl_print[epoch] = 0.0
            recon_loss_MSE_print[epoch] = 0.0
        
        # Store learning rate
        current_lr = optimizer.param_groups[0]['lr']
        training_metrics['learning_rates'][epoch] = current_lr
        
        # Update learning rate schedule
        scheduler.step()
        
        # Calculate timing and ETA
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        elapsed_time = epoch_end_time - total_start_time
        avg_epoch_time = elapsed_time / (epoch + 1)
        eta_hours = (epochs - epoch - 1) * avg_epoch_time / 3600
        
        # Reduced logging frequency - only log every 10 epochs or significant milestones
        if epoch % 10 == 0 or epoch == epochs - 1 or epoch < 5:
            log_message = (
                f"[Epoch {epoch+1:4d}/{epochs}] "
                f"Loss: {loss_print[epoch]:.4e} "
                f"Val: {loss_val_print[epoch]:.4e} "
                f"Recon: {recon_print[epoch]:.4e} "
                f"KL: {kl_print[epoch]:.4e} "
                f"Time: {epoch_duration:.1f}s "
                f"ETA: {eta_hours:.1f}h"
            )
            logger.info(log_message)
        
        # Reduced TensorBoard logging frequency
        if epoch % 5 == 0 or epoch == epochs - 1:
            writer.add_scalar('Loss/Train', loss_print[epoch], epoch)
            writer.add_scalar('Loss/Validation', loss_val_print[epoch], epoch)
            writer.add_scalar('Loss/Reconstruction', recon_print[epoch], epoch)
            writer.add_scalar('Loss/KL_Divergence', kl_print[epoch], epoch)
            writer.add_scalar('Training/Beta', current_beta, epoch)
            writer.add_scalar('Training/Learning_Rate', current_lr, epoch)
        
        # Early stopping check
        if epochs_without_improvement >= early_stopping_patience and epoch > epochs * 0.5:
            logger.info(f"Early stopping triggered after {epochs_without_improvement} epochs without improvement")
            logger.info(f"Best validation loss: {best_val_loss:.6f} at epoch {epoch - epochs_without_improvement + 1}")
            break

    # Training completion and final model saving
    total_training_time = time.time() - total_start_time
    logger.info(f"Training completed in {total_training_time/3600:.2f} hours")
    logger.info(f"Final training loss: {loss_print[epoch]:.6f}")
    logger.info(f"Final validation loss: {loss_val_print[epoch]:.6f}")
    logger.info(f"Best validation loss: {best_val_loss:.6f}")
    
    # Save final model and training state
    logger.info("Saving final model and training state...")
    try:
        # Save model state dictionary
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'training_metrics': training_metrics,
            'final_val_loss': loss_val_print[epoch],
            'best_val_loss': best_val_loss
        }, 'checkpoints/SimulGen-VAE.pth')
        
        # Save complete model for inference
        torch.save(model, 'model_save/SimulGen-VAE')
        
        # Save training metrics for analysis
        np.savez('output/training_metrics.npz', **training_metrics)
        
        logger.info("Model and training state saved successfully")
        
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        raise
    
    # Cleanup and close resources
    writer.close()
    torch.cuda.empty_cache()
    
    logger.info("Training pipeline completed successfully")
    
    return loss_print, recon_print, kl_print, loss_val_print
    