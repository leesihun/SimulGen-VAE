"""
SimulGenVAE Training Module - Refactored and Clean

This module provides a comprehensive training pipeline for the SimulGenVAE model 
with Variational Autoencoder architecture. The training includes advanced features
such as KL divergence warmup, adaptive learning rate scheduling, and robust 
validation monitoring.

Key Features:
    - β-VAE training with KL warmup scheduling
    - Cosine annealing learning rate with warm restarts
    - Comprehensive validation and early stopping
    - Memory-efficient data handling
    - Detailed logging and monitoring
    - TensorBoard integration
    - Robust error handling and recovery

Architecture:
    The training pipeline is organized into several key components:
    1. Configuration Management: TrainingConfig class
    2. State Tracking: TrainingState class  
    3. KL Warmup: WarmupKLScheduler class
    4. Training Logic: Modular functions for each training phase
    5. Validation: Separate validation pipeline
    6. Monitoring: Logging and metrics collection

Author: SiHun Lee, Ph.D.
Contact: kevin1007kr@gmail.com
Version: 3.0.0 (Refactored)
"""

import os
import time
import logging
from typing import Tuple, List, Dict, Optional, Any
from dataclasses import dataclass, field

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary

from modules.common import initialize_weights_He, add_sn
from modules.VAE_network import VAE
from modules.losses import kl


# =============================================================================
# CONFIGURATION AND CONSTANTS
# =============================================================================

@dataclass
class TrainingConfig:
    """
    Configuration class for VAE training parameters.
    
    This class centralizes all training hyperparameters and settings,
    making the training pipeline more maintainable and configurable.
    
    Attributes:
        Model Architecture:
            epochs: Number of training epochs
            batch_size: Training batch size per device
            learning_rate: Initial learning rate for optimizer
            num_filter_enc: Encoder layer filter counts
            num_filter_dec: Decoder layer filter counts
            
        VAE Parameters:
            num_node: Number of spatial nodes in simulation
            latent_dim: Hierarchical latent dimension
            hierarchical_dim: Main latent dimension
            num_time: Number of temporal steps
            alpha: Reconstruction loss weighting factor
            lossfun: Loss function type ('MSE'|'MAE'|'smoothL1'|'Huber')
            
        Training Control:
            small: Whether to use memory-efficient model variant
            load_all: Whether data is pre-loaded to GPU memory
            early_stopping_patience: Epochs to wait for improvement
            gradient_clip_norm: Maximum gradient norm for clipping
            
        KL Warmup:
            kl_init_beta: Initial β value for KL warmup
            kl_target_beta: Final β value for KL warmup
            kl_warmup_start_ratio: When to start warmup (fraction of total epochs)
            kl_warmup_end_ratio: When to end warmup (fraction of total epochs)
            
        Learning Rate:
            lr_restart_period_ratio: T_0 as fraction of total epochs
            lr_restart_multiplier: T_mult for cosine annealing
            lr_min_factor: Minimum LR as factor of initial LR
            
        Validation:
            validation_frequency_early: Validation frequency in early training
            validation_frequency_late: Validation frequency in late training
            validation_transition_ratio: When to switch frequencies
            
        Logging:
            log_epoch_frequency: How often to log detailed epoch info
            tensorboard_frequency: How often to log to TensorBoard
    """
    
    # Model Architecture
    epochs: int
    batch_size: int
    learning_rate: float
    num_filter_enc: List[int]
    num_filter_dec: List[int]
    
    # VAE Parameters  
    num_node: int
    latent_dim: int
    hierarchical_dim: int
    num_time: int
    alpha: float
    lossfun: str
    
    # Training Control
    small: bool = False
    load_all: bool = True
    early_stopping_patience: Optional[int] = None
    gradient_clip_norm: float = 1.0
    
    # KL Warmup Settings
    kl_init_beta: float = 1e-4
    kl_target_beta: float = 1.0
    kl_warmup_start_ratio: float = 0.3
    kl_warmup_end_ratio: float = 0.8
    
    # Learning Rate Settings
    lr_restart_period_ratio: float = 0.25
    lr_restart_multiplier: int = 2
    lr_min_factor: float = 1e-4
    
    # Validation Settings
    validation_frequency_early: int = 25
    validation_frequency_late: int = 5
    validation_transition_ratio: float = 0.7
    
    # Logging Settings
    log_epoch_frequency: int = 1  # Log every epoch
    tensorboard_frequency: int = 5
    
    def __post_init__(self):
        """Validate configuration and set derived parameters."""
        if self.early_stopping_patience is None:
            self.early_stopping_patience = max(self.epochs // 10, 50)
        
        # Validate ratios
        if not 0 <= self.kl_warmup_start_ratio < self.kl_warmup_end_ratio <= 1:
            raise ValueError("KL warmup ratios must satisfy: 0 ≤ start < end ≤ 1")
        
        if not 0 < self.validation_transition_ratio < 1:
            raise ValueError("Validation transition ratio must be between 0 and 1")


@dataclass  
class TrainingState:
    """
    Tracks the current state of training.
    
    This class maintains all stateful information during training,
    including metrics, best models, and progress tracking.
    
    Attributes:
        Current State:
            current_epoch: Current training epoch (0-indexed)
            best_val_loss: Best validation loss seen so far
            epochs_without_improvement: Consecutive epochs without improvement
            
        Loss Tracking:
            train_losses: Training loss per epoch
            val_losses: Validation loss per epoch
            recon_losses: Reconstruction loss per epoch
            kl_losses: KL divergence loss per epoch
            recon_mse_losses: MSE reconstruction loss per epoch
            val_recon_losses: Validation reconstruction loss per epoch
            
        Training Metadata:
            beta_values: β values used per epoch
            learning_rates: Learning rates per epoch
            epoch_durations: Time taken per epoch
            
        Flags:
            should_stop_early: Whether early stopping should trigger
    """
    
    # Training progress
    current_epoch: int = 0
    best_val_loss: float = float('inf')
    epochs_without_improvement: int = 0
    
    # Loss arrays - initialized lazily
    train_losses: np.ndarray = field(default=None)
    val_losses: np.ndarray = field(default=None)
    recon_losses: np.ndarray = field(default=None)
    kl_losses: np.ndarray = field(default=None)
    recon_mse_losses: np.ndarray = field(default=None)
    val_recon_losses: np.ndarray = field(default=None)
    
    # Training metadata
    beta_values: np.ndarray = field(default=None)
    learning_rates: np.ndarray = field(default=None)
    epoch_durations: List[float] = field(default_factory=list)
    
    # Control flags
    should_stop_early: bool = False
    
    def initialize_arrays(self, num_epochs: int):
        """Initialize all tracking arrays."""
        self.train_losses = np.zeros(num_epochs)
        self.val_losses = np.zeros(num_epochs)
        self.recon_losses = np.zeros(num_epochs)
        self.kl_losses = np.zeros(num_epochs)
        self.recon_mse_losses = np.zeros(num_epochs)
        self.val_recon_losses = np.zeros(num_epochs)
        self.beta_values = np.zeros(num_epochs)
        self.learning_rates = np.zeros(num_epochs)
        self.epoch_durations = []
    
    def update_best_model(self, val_loss: float, min_delta: float = 1e-8) -> bool:
        """
        Update best model tracking.
        
        Args:
            val_loss: Current validation loss
            min_delta: Minimum improvement to count as better
            
        Returns:
            True if this is a new best model
        """
        if val_loss < self.best_val_loss - min_delta and val_loss > 0:
            self.best_val_loss = val_loss
            self.epochs_without_improvement = 0
            return True
        else:
            self.epochs_without_improvement += 1
            return False
    
    def check_early_stopping(self, patience: int, min_epochs_ratio: float = 0.5) -> bool:
        """
        Check if early stopping should trigger.
        
        Args:
            patience: Maximum epochs without improvement
            min_epochs_ratio: Minimum fraction of epochs to complete before early stopping
            
        Returns:
            True if training should stop early
        """
        min_epochs = int(len(self.train_losses) * min_epochs_ratio)
        return (self.epochs_without_improvement >= patience and 
                self.current_epoch > min_epochs)


# =============================================================================
# KL WARMUP SCHEDULER
# =============================================================================

class WarmupKLScheduler:
    """
    KL Divergence Warmup Scheduler for β-VAE Training.
    
    Implements a smooth warmup schedule for the KL divergence weighting parameter (β)
    to prevent posterior collapse during early training. The scheduler provides three
    distinct phases:
    
    1. Initial Phase: Small constant β to allow encoder learning
    2. Warmup Phase: Linear increase from initial to target β  
    3. Stable Phase: Constant target β for standard VAE training
    
    The warmup schedule helps balance reconstruction accuracy and latent space
    structure learning, leading to more stable and effective training.
    
    Args:
        total_epochs: Total number of training epochs
        init_beta: Initial β value (typically 1e-4 to 1e-6)
        target_beta: Final β value (typically 1.0)
        start_epoch: Epoch to begin warmup
        end_epoch: Epoch to complete warmup
        
    Example:
        >>> scheduler = WarmupKLScheduler(
        ...     total_epochs=1000, init_beta=1e-4, target_beta=1.0,
        ...     start_epoch=300, end_epoch=800
        ... )
        >>> beta = scheduler.get_beta(epoch=500)  # Returns interpolated value
        >>> weighted_kl = scheduler.apply_schedule(epoch=500, kl_losses=[kl1, kl2])
    """
    
    def __init__(
        self, 
        total_epochs: int,
        init_beta: float,
        target_beta: float, 
        start_epoch: int,
        end_epoch: int
    ):
        """Initialize KL warmup scheduler."""
        self.total_epochs = total_epochs
        self.init_beta = init_beta
        self.target_beta = target_beta
        self.start_epoch = start_epoch
        self.end_epoch = end_epoch
        
        # Validate parameters
        if start_epoch >= end_epoch:
            raise ValueError(f"start_epoch ({start_epoch}) must be < end_epoch ({end_epoch})")
        if end_epoch > total_epochs:
            raise ValueError(f"end_epoch ({end_epoch}) cannot exceed total_epochs ({total_epochs})")
        if init_beta <= 0 or target_beta <= 0:
            raise ValueError("Beta values must be positive")
    
    def get_beta(self, epoch: int) -> float:
        """
        Get the β value for a given epoch.
        
        Args:
            epoch: Current training epoch (0-indexed)
            
        Returns:
            Current β value based on warmup schedule
        """
        if epoch < self.start_epoch:
            return self.init_beta
        elif epoch < self.end_epoch:
            # Linear interpolation during warmup
            progress = (epoch - self.start_epoch) / (self.end_epoch - self.start_epoch)
            return self.init_beta + progress * (self.target_beta - self.init_beta)
        else:
            return self.target_beta
    
    def apply_schedule(self, epoch: int, kl_losses: List[torch.Tensor]) -> Tuple[float, torch.Tensor]:
        """
        Apply warmup schedule to KL losses.
        
        Args:
            epoch: Current training epoch
            kl_losses: List of KL divergence tensors from VAE
            
        Returns:
            Tuple of (current_beta, weighted_kl_loss)
        """
        current_beta = self.get_beta(epoch)
        total_kl = torch.stack(kl_losses).sum() if kl_losses else torch.tensor(0.0)
        return current_beta, total_kl * current_beta


# =============================================================================
# TRAINING UTILITIES
# =============================================================================

def setup_logging() -> logging.Logger:
    """
    Configure logging for training.
    
    Returns:
        Configured logger instance
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(message)s",
        datefmt="%H:%M:%S"
    )
    return logging.getLogger(__name__)


def setup_device() -> torch.device:
    """
    Setup and configure training device.
    
    Returns:
        Configured torch device
    """
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
        # Optimize CUDA settings
        torch.cuda.set_device(device)
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
        
        # Enable optimized attention if available
        if hasattr(torch.backends.cuda, 'enable_math_sdp'):
            torch.backends.cuda.enable_math_sdp(True)
        if hasattr(torch.backends.cuda, 'enable_flash_sdp'):
            torch.backends.cuda.enable_flash_sdp(True)
        
        torch.cuda.empty_cache()
        return device
    else:
        return torch.device("cpu")


def create_directories():
    """Create necessary directories for training outputs."""
    for directory in ['checkpoints', 'output', 'model_save', 'runs']:
        os.makedirs(directory, exist_ok=True)


def initialize_model(config: TrainingConfig, device: torch.device) -> VAE:
    """
    Initialize and configure the VAE model.
    
    Args:
        config: Training configuration
        device: Target device for model
        
    Returns:
        Initialized VAE model
        
    Raises:
        RuntimeError: If model initialization fails
    """
    try:
        model = VAE(
            latent_dim=config.latent_dim,
            hierarchical_dim=config.hierarchical_dim,
            num_filter_enc=config.num_filter_enc,
            num_filter_dec=config.num_filter_dec,
            num_node=config.num_node,
            num_time=config.num_time,
            lossfun=config.lossfun,
            batch_size=config.batch_size,
            small=config.small,
            use_checkpointing=False
        )
        
        # Apply weight initialization and normalization
        model.apply(initialize_weights_He)
        model.apply(add_sn)
        
        # Move to device and ensure correct dtype
        model = model.to(device, dtype=torch.float32)
        
        return model
        
    except Exception as e:
        raise RuntimeError(f"Model initialization failed: {e}")


def create_optimizer_and_scheduler(
    model: nn.Module, 
    config: TrainingConfig
) -> Tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    """
    Create optimizer and learning rate scheduler.
    
    Args:
        model: Model to optimize
        config: Training configuration
        
    Returns:
        Tuple of (optimizer, scheduler)
    """
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        betas=(0.9, 0.999),
        eps=1e-8
    )
    
    # Cosine annealing with warm restarts
    T_0 = max(int(config.epochs * config.lr_restart_period_ratio), 50)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer,
        T_0=T_0,
        T_mult=config.lr_restart_multiplier,
        eta_min=config.learning_rate * config.lr_min_factor
    )
    
    return optimizer, scheduler


# =============================================================================
# TRAINING PHASES
# =============================================================================

def train_single_epoch(
    model: VAE,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    kl_scheduler: WarmupKLScheduler,
    config: TrainingConfig,
    epoch: int,
    device: torch.device,
    logger: logging.Logger
) -> Dict[str, float]:
    """
    Train model for a single epoch.
    
    Args:
        model: VAE model to train
        dataloader: Training data loader
        optimizer: Model optimizer
        kl_scheduler: KL warmup scheduler  
        config: Training configuration
        epoch: Current epoch number
        device: Training device
        logger: Logger instance
        
    Returns:
        Dictionary of epoch metrics
        
    Raises:
        RuntimeError: If training encounters unrecoverable errors
    """
    model.train()
    
    # Initialize epoch metrics
    metrics = {
        'total_loss': 0.0,
        'recon_loss': 0.0,
        'kl_loss': 0.0,
        'recon_mse_loss': 0.0,
        'batch_count': 0
    }
    
    try:
        for batch_idx, batch_data in enumerate(dataloader):
            # Move data to device if needed and ensure correct dtype
            if not config.load_all:
                batch_data = batch_data.to(device, dtype=torch.float32, non_blocking=True)
            else:
                # Ensure dtype consistency even for pre-loaded data
                batch_data = batch_data.to(dtype=torch.float32)
            
            # Zero gradients
            optimizer.zero_grad(set_to_none=True)
            
            # Forward pass
            try:
                reconstruction, recon_loss, kl_losses, recon_mse_loss = model(batch_data)
            except RuntimeError as e:
                if "out of memory" in str(e):
                    logger.error(f"CUDA OOM at epoch {epoch+1}, batch {batch_idx+1}")
                    torch.cuda.empty_cache()
                    continue
                else:
                    raise
            
            # Apply KL warmup schedule
            beta, weighted_kl_loss = kl_scheduler.apply_schedule(epoch, kl_losses)
            
            # Scale losses
            scaled_recon_loss = recon_loss * config.alpha
            scaled_mse_loss = recon_mse_loss * config.alpha
            
            # Total loss
            total_loss = scaled_recon_loss + weighted_kl_loss
            
            # Backward pass
            total_loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=config.gradient_clip_norm)
            
            # Optimizer step
            optimizer.step()
            
            # Accumulate metrics
            metrics['total_loss'] += total_loss.item()
            metrics['recon_loss'] += scaled_recon_loss.item()
            metrics['kl_loss'] += weighted_kl_loss.item()
            metrics['recon_mse_loss'] += scaled_mse_loss.item()
            metrics['batch_count'] += 1
            
            # Clean up
            del reconstruction, batch_data
            
            # Periodic memory cleanup
            if batch_idx % 500 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
                
    except Exception as e:
        logger.error(f"Training error at epoch {epoch+1}: {e}")
        raise
    
    # Average metrics over batches
    if metrics['batch_count'] > 0:
        for key in ['total_loss', 'recon_loss', 'kl_loss', 'recon_mse_loss']:
            metrics[key] /= metrics['batch_count']
    
    return metrics


def validate_model(
    model: VAE,
    dataloader: DataLoader,
    kl_scheduler: WarmupKLScheduler,
    config: TrainingConfig,
    epoch: int,
    device: torch.device,
    logger: logging.Logger
) -> Dict[str, float]:
    """
    Validate model performance.
    
    Args:
        model: VAE model to validate
        dataloader: Validation data loader
        kl_scheduler: KL warmup scheduler
        config: Training configuration
        epoch: Current epoch number
        device: Validation device
        logger: Logger instance
        
    Returns:
        Dictionary of validation metrics
        
    Raises:
        RuntimeError: If validation encounters errors
    """
    model.eval()
    
    metrics = {
        'total_loss': 0.0,
        'recon_loss': 0.0,
        'batch_count': 0
    }
    
    try:
        with torch.no_grad():
            for batch_idx, batch_data in enumerate(dataloader):
                # Move data to device if needed and ensure correct dtype
                if not config.load_all:
                    batch_data = batch_data.to(device, dtype=torch.float32, non_blocking=True)
                else:
                    # Ensure dtype consistency even for pre-loaded data
                    batch_data = batch_data.to(dtype=torch.float32)
                
                # Forward pass
                reconstruction, recon_loss, kl_losses, recon_mse_loss = model(batch_data)
                
                # Apply same loss weighting as training
                beta, weighted_kl_loss = kl_scheduler.apply_schedule(epoch, kl_losses)
                scaled_recon_loss = recon_loss * config.alpha
                total_loss = scaled_recon_loss + weighted_kl_loss
                
                # Accumulate metrics
                metrics['total_loss'] += total_loss.item()
                metrics['recon_loss'] += scaled_recon_loss.item()
                metrics['batch_count'] += 1
                
                # Clean up
                del batch_data, reconstruction
                
    except Exception as e:
        logger.warning(f"Validation error at epoch {epoch+1}: {e}")
        raise
    
    # Average metrics
    if metrics['batch_count'] > 0:
        for key in ['total_loss', 'recon_loss']:
            metrics[key] /= metrics['batch_count']
    
    return metrics


def should_run_validation(epoch: int, config: TrainingConfig) -> bool:
    """
    Determine if validation should run for this epoch.
    
    Args:
        epoch: Current epoch number
        config: Training configuration
        
    Returns:
        True if validation should run
    """
    transition_epoch = int(config.epochs * config.validation_transition_ratio)
    
    if epoch < transition_epoch:
        frequency = min(config.validation_frequency_early, config.epochs // 40)
        frequency = max(1, frequency)
    else:
        frequency = config.validation_frequency_late
    
    return epoch % frequency == 0 or epoch == config.epochs - 1


def log_epoch_results(
    epoch: int,
    config: TrainingConfig,
    state: TrainingState,
    train_metrics: Dict[str, float],
    kl_beta: float,
    learning_rate: float,
    epoch_duration: float,
    logger: logging.Logger
):
    """
    Log training results for an epoch.
    
    Args:
        epoch: Current epoch number (0-indexed)
        config: Training configuration
        state: Training state
        train_metrics: Training metrics for this epoch
        kl_beta: Current KL beta value
        learning_rate: Current learning rate
        epoch_duration: Time taken for this epoch
        logger: Logger instance
    """
    # Calculate ETA
    if len(state.epoch_durations) > 0:
        avg_epoch_time = np.mean(state.epoch_durations[-10:])  # Average of last 10 epochs
        remaining_epochs = config.epochs - epoch - 1
        eta_hours = remaining_epochs * avg_epoch_time / 3600
    else:
        eta_hours = 0.0
    
    # Get validation loss
    val_loss = state.val_losses[epoch] if state.val_losses is not None else 0.0
    
    # Format message
    message = (
        f"[Epoch {epoch+1:4d}/{config.epochs}] "
        f"Loss: {train_metrics['total_loss']:.4e} "
        f"Val: {val_loss:.4e} "
        f"Recon: {train_metrics['recon_loss']:.4e} "
        f"KL: {train_metrics['kl_loss']:.4e} "
        f"Time: {epoch_duration:.1f}s "
        f"ETA: {eta_hours:.1f}h"
    )
    
    logger.info(message)


def save_model_and_metrics(
    model: VAE,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    state: TrainingState,
    config: TrainingConfig,
    logger: logging.Logger
):
    """
    Save trained model and training metrics.
    
    Args:
        model: Trained VAE model
        optimizer: Model optimizer
        scheduler: Learning rate scheduler
        state: Training state with metrics
        config: Training configuration
        logger: Logger instance
        
    Raises:
        RuntimeError: If saving fails
    """
    try:
        # Save model state dictionary
        torch.save({
            'epoch': state.current_epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'final_val_loss': state.val_losses[state.current_epoch],
            'best_val_loss': state.best_val_loss,
            'config': config
        }, 'checkpoints/SimulGen-VAE.pth')
        
        # Save complete model
        torch.save(model, 'model_save/SimulGen-VAE')
        
        # Save training metrics
        metrics_dict = {
            'train_losses': state.train_losses,
            'val_losses': state.val_losses,
            'recon_losses': state.recon_losses,
            'kl_losses': state.kl_losses,
            'recon_mse_losses': state.recon_mse_losses,
            'val_recon_losses': state.val_recon_losses,
            'beta_values': state.beta_values,
            'learning_rates': state.learning_rates
        }
        np.savez('output/training_metrics.npz', **metrics_dict)
        
        logger.info("Model and training metrics saved successfully")
        
    except Exception as e:
        logger.error(f"Error saving model: {e}")
        raise RuntimeError(f"Model saving failed: {e}")


# =============================================================================
# MAIN TRAINING FUNCTION
# =============================================================================

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
    """
    Train SimulGenVAE model with comprehensive monitoring and validation.
    
    This is the main training function that orchestrates the entire training
    pipeline. It sets up all components, runs the training loop, and returns
    the training history.
    
    The training process includes:
    - Model initialization and configuration
    - KL divergence warmup scheduling
    - Adaptive learning rate scheduling
    - Regular validation and early stopping
    - Comprehensive logging and monitoring
    - Model checkpointing and saving
    
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
            - train_losses: Training loss per epoch
            - recon_losses: Reconstruction loss per epoch
            - kl_losses: KL divergence loss per epoch  
            - val_losses: Validation loss per epoch
            
    Raises:
        RuntimeError: If training setup or execution fails
        ValueError: If invalid parameters are provided
        
    Example:
        >>> train_losses, recon_losses, kl_losses, val_losses = train(
        ...     epochs=1000, batch_size=16, train_dataloader=train_dl,
        ...     val_dataloader=val_dl, LR=1e-3, num_filter_enc=[64,128,256],
        ...     num_filter_dec=[256,128,64], num_node=1000, latent_dim=8,
        ...     hierarchical_dim=32, num_time=200, alpha=1.0,
        ...     lossfun='MSE', small=False, load_all=True
        ... )
    """
    
    # =========================================================================
    # SETUP PHASE
    # =========================================================================
    
    # Initialize configuration
    config = TrainingConfig(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=LR,
        num_filter_enc=num_filter_enc,
        num_filter_dec=num_filter_dec,
        num_node=num_node,
        latent_dim=latent_dim,
        hierarchical_dim=hierarchical_dim,
        num_time=num_time,
        alpha=alpha,
        lossfun=lossfun,
        small=small,
        load_all=load_all
    )
    
    # Setup logging and directories
    logger = setup_logging()
    create_directories()
    
    # Setup device
    device = setup_device()
    
    # Initialize TensorBoard
    writer = SummaryWriter(log_dir='./runs', comment='VAE_Training')
    
    # Log training configuration
    logger.info(f"Starting VAE training for {config.epochs} epochs")
    logger.info(f"Architecture: {len(config.num_filter_enc)} encoder layers, {len(config.num_filter_dec)} decoder layers")
    logger.info(f"Latent dimensions: {config.latent_dim} (hierarchical), {config.hierarchical_dim} (main)")
    logger.info(f"Loss function: {config.lossfun}, Alpha: {config.alpha}")
    logger.info(f"Training device: {device}")
    
    if torch.cuda.is_available():
        logger.info(f"Available GPUs: {torch.cuda.device_count()}")
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    
    # =========================================================================
    # MODEL INITIALIZATION
    # =========================================================================
    
    logger.info("Initializing VAE model...")
    model = initialize_model(config, device)
    
    # Display model summary
    try:
        logger.info("Model Architecture Summary:")
        summary(model, input_size=(config.batch_size, config.num_node, config.num_time), device='cpu')
    except Exception as e:
        logger.warning(f"Could not display model summary: {e}")
    
    # =========================================================================
    # TRAINING COMPONENTS SETUP
    # =========================================================================
    
    # Create optimizer and scheduler
    optimizer, lr_scheduler = create_optimizer_and_scheduler(model, config)
    
    # Setup KL warmup scheduler
    kl_start_epoch = int(config.epochs * config.kl_warmup_start_ratio)
    kl_end_epoch = int(config.epochs * config.kl_warmup_end_ratio)
    
    kl_scheduler = WarmupKLScheduler(
        total_epochs=config.epochs,
        init_beta=config.kl_init_beta,
        target_beta=config.kl_target_beta,
        start_epoch=kl_start_epoch,
        end_epoch=kl_end_epoch
    )
    
    logger.info(f"KL Warmup Schedule: β {config.kl_init_beta:.1e} → {config.kl_target_beta:.1f} "
                f"(epochs {kl_start_epoch}-{kl_end_epoch})")
    
    # Initialize training state
    state = TrainingState()
    state.initialize_arrays(config.epochs)
    
    logger.info(f"Early stopping patience: {config.early_stopping_patience} epochs")
    logger.info(f"Learning rate schedule: Cosine annealing with warm restarts")
    
    # =========================================================================
    # MAIN TRAINING LOOP
    # =========================================================================
    
    logger.info("Starting training loop...")
    total_start_time = time.time()
    
    for epoch in range(config.epochs):
        epoch_start_time = time.time()
        state.current_epoch = epoch
        
        # Get current KL beta and learning rate
        current_beta = kl_scheduler.get_beta(epoch)
        current_lr = optimizer.param_groups[0]['lr']
        
        # Store values
        state.beta_values[epoch] = current_beta
        state.learning_rates[epoch] = current_lr
        
        # =====================================================================
        # TRAINING PHASE
        # =====================================================================
        
        train_metrics = train_single_epoch(
            model=model,
            dataloader=train_dataloader,
            optimizer=optimizer,
            kl_scheduler=kl_scheduler,
            config=config,
            epoch=epoch,
            device=device,
            logger=logger
        )
        
        # Store training metrics
        state.train_losses[epoch] = train_metrics['total_loss']
        state.recon_losses[epoch] = train_metrics['recon_loss']
        state.kl_losses[epoch] = train_metrics['kl_loss'] / current_beta  # Unweighted KL
        state.recon_mse_losses[epoch] = train_metrics['recon_mse_loss']
        
        # =====================================================================
        # VALIDATION PHASE
        # =====================================================================
        
        if should_run_validation(epoch, config):
            val_metrics = validate_model(
                model=model,
                dataloader=val_dataloader,
                kl_scheduler=kl_scheduler,
                config=config,
                epoch=epoch,
                device=device,
                logger=logger
            )
            
            state.val_losses[epoch] = val_metrics['total_loss']
            state.val_recon_losses[epoch] = val_metrics['recon_loss']
            
            # Check for best model
            is_best = state.update_best_model(val_metrics['total_loss'])
            
        else:
            # Use previous validation metrics
            if epoch > 0:
                state.val_losses[epoch] = state.val_losses[epoch - 1]
                state.val_recon_losses[epoch] = state.val_recon_losses[epoch - 1]
        
        # =====================================================================
        # LEARNING RATE UPDATE
        # =====================================================================
        
        lr_scheduler.step()
        
        # =====================================================================
        # LOGGING AND MONITORING
        # =====================================================================
        
        # Calculate epoch duration
        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        state.epoch_durations.append(epoch_duration)
        
        # Log epoch results
        log_epoch_results(
            epoch=epoch,
            config=config,
            state=state,
            train_metrics=train_metrics,
            kl_beta=current_beta,
            learning_rate=current_lr,
            epoch_duration=epoch_duration,
            logger=logger
        )
        
        # TensorBoard logging
        if epoch % config.tensorboard_frequency == 0 or epoch == config.epochs - 1:
            writer.add_scalar('Loss/Train', state.train_losses[epoch], epoch)
            writer.add_scalar('Loss/Validation', state.val_losses[epoch], epoch)
            writer.add_scalar('Loss/Reconstruction', state.recon_losses[epoch], epoch)
            writer.add_scalar('Loss/KL_Divergence', state.kl_losses[epoch], epoch)
            writer.add_scalar('Training/Beta', current_beta, epoch)
            writer.add_scalar('Training/Learning_Rate', current_lr, epoch)
        
        # =====================================================================
        # EARLY STOPPING CHECK
        # =====================================================================
        
        if state.check_early_stopping(config.early_stopping_patience):
            logger.info(f"Early stopping triggered after {state.epochs_without_improvement} epochs without improvement")
            logger.info(f"Best validation loss: {state.best_val_loss:.6f} at epoch {epoch - state.epochs_without_improvement + 1}")
            break
    
    # =========================================================================
    # TRAINING COMPLETION
    # =========================================================================
    
    total_training_time = time.time() - total_start_time
    logger.info(f"Training completed in {total_training_time/3600:.2f} hours")
    logger.info(f"Final training loss: {state.train_losses[epoch]:.6f}")
    logger.info(f"Final validation loss: {state.val_losses[epoch]:.6f}")
    logger.info(f"Best validation loss: {state.best_val_loss:.6f}")
    
    # Save model and metrics
    save_model_and_metrics(model, optimizer, lr_scheduler, state, config, logger)
    
    # Cleanup
    writer.close()
    torch.cuda.empty_cache()
    
    logger.info("Training pipeline completed successfully")
    
    # Return training history for backward compatibility
    return (
        state.train_losses,
        state.recon_losses,
        state.kl_losses,
        state.val_losses
    )