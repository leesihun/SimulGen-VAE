"""VAE Network Module

This module implements the main Variational Autoencoder (VAE) architecture for SimulGenVAE.
It combines hierarchical encoder-decoder networks with multiple loss functions and advanced
optimization features for simulation data processing.

Author: SiHun Lee, Ph.D.
Email: kevin1007kr@gmail.com
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from modules.encoder import Encoder
from modules.decoder import Decoder, reparameterize
from modules.losses import kl
from modules.common import add_sn
import matplotlib.pyplot as plt
from torchinfo import summary

class VAE(nn.Module):
    """Main Variational Autoencoder class for simulation data processing.
    
    This VAE implements a hierarchical latent space architecture with encoder-decoder
    networks optimized for transient and static simulation data. Features include
    multiple loss functions, gradient checkpointing support, and advanced memory management.
    
    Args:
        latent_dim (int): Main latent space dimension (typically 32)
        hierarchical_dim (int): Hierarchical latent dimension for multi-scale representation (typically 8)
        num_filter_enc (list): Number of filters for each encoder layer
        num_filter_dec (list): Number of filters for each decoder layer  
        num_node (int): Number of nodes in the simulation data
        num_time (int): Number of time steps in the simulation data
        lossfun (str): Loss function type. Options: 'MSE', 'MAE', 'smoothL1', 'Huber'
        batch_size (int): Batch size for training (used for decoder initialization)
        small (bool): Whether to use smaller model variant for memory efficiency
        use_checkpointing (bool): Enable gradient checkpointing (disabled for speed)
    
    Attributes:
        latent_dim (int): Dimension of the main latent space
        encoder (Encoder): Hierarchical encoder network
        decoder (Decoder): Hierarchical decoder network with skip connections
        lossfun (str): Selected loss function type
        loss_functions (dict): Pre-compiled loss functions for efficiency
        mse_loss (nn.MSELoss): MSE loss function (always available for metrics)
    """
    def __init__(self, latent_dim, hierarchical_dim, num_filter_enc, num_filter_dec, num_node, num_time, lossfun='MSE', batch_size=1, small=False, use_checkpointing=False):
        super().__init__()

        self.latent_dim = latent_dim
        self.encoder = Encoder(latent_dim, hierarchical_dim, num_filter_enc, num_node, num_time, small)
        self.decoder = Decoder(latent_dim, hierarchical_dim, num_filter_dec, num_node, num_time, batch_size, small)
        self.lossfun = lossfun
        # Checkpointing disabled for speed (user preference)
        self.use_checkpointing = False
        
        # Pre-compile loss functions for efficiency
        self.loss_functions = {
            'MSE': nn.MSELoss(),
            'MAE': nn.L1Loss(), 
            'smoothL1': nn.SmoothL1Loss(),
            'Huber': nn.HuberLoss()
        }
        self.mse_loss = nn.MSELoss()  # Always needed for recon_loss_MSE

    def forward(self, x):
        """Forward pass through the VAE.
        
        Performs encoding to latent space, reparameterization, and decoding back to
        reconstruction space. Includes comprehensive error handling for CUDA operations.
        
        Args:
            x (torch.Tensor): Input tensor of shape [batch, num_node, num_time]
        
        Returns:
            tuple: (decoder_output, recon_loss, kl_losses, recon_loss_MSE)
                - decoder_output (torch.Tensor): Reconstructed data
                - recon_loss (torch.Tensor): Reconstruction loss using specified loss function
                - kl_losses (list): KL divergence losses [main_kl, hierarchical_kl_losses...]
                - recon_loss_MSE (torch.Tensor): MSE reconstruction loss for monitoring
        
        Raises:
            RuntimeError: If CUDA out of memory or other CUDA errors occur
        """
        try:
            # Always use regular forward pass - no speed trade-offs
            mu, log_var, xs = self.encoder(x)
            
            # Clamp log_var to prevent numerical instability before computing std
            log_var = torch.clamp(log_var, min=-30, max=30)
            std = torch.exp(0.5*log_var)
            z = reparameterize(mu, std)
            
            decoder_output, kl_losses = self.decoder(z, xs)

            # Use pre-compiled loss functions for efficiency
            recon_loss = self.loss_functions.get(self.lossfun, self.mse_loss)(decoder_output, x)
            recon_loss_MSE = self.mse_loss(decoder_output, x)

            kl_loss = kl(mu, log_var)

            # Clean up intermediate variables to free memory faster
            del mu, log_var, xs, z
            return decoder_output, recon_loss, [kl_loss]+kl_losses, recon_loss_MSE
            
        except RuntimeError as e:
            if "CUDA out of memory" in str(e):
                print(f"CUDA out of memory in VAE forward pass: {e}")
                print(f"Input tensor shape: {x.shape}, device: {x.device}")
                # Try to recover by freeing memory
                torch.cuda.empty_cache()
                # Re-raise to let caller handle it
                raise
            elif "CUDA error" in str(e):
                print(f"CUDA error in VAE forward pass: {e}")
                # Try to recover
                torch.cuda.empty_cache()
                raise
            else:
                print(f"Unexpected error in VAE forward pass: {e}")
                raise
    
    def compile_model(self, mode='default'):
        """Compile the model for better performance using torch.compile.
        
        Should be called after moving to GPU and before training. Compiles both
        encoder and decoder separately for optimal performance on consistent input sizes.
        Includes fallback handling if compilation fails.
        
        Args:
            mode (str or bool): Compilation mode. Options:
                - 'default': Conservative compilation (recommended)
                - 'reduce-overhead': Faster compilation time
                - 'max-autotune': Most aggressive optimization (may fail)
                - False or 'none': Skip compilation entirely
        
        Note:
            If compilation fails, the model falls back to uncompiled mode without
            raising an exception to ensure training can continue.
        """
        if mode is False or mode == 'none':
            print("Model compilation disabled")
            return
            
        if not hasattr(torch, 'compile'):
            print("torch.compile not available, skipping model compilation")
            return
            
        print(f"Compiling VAE model with mode='{mode}'...")
        try:
            print("  Compiling encoder...")
            self.encoder = torch.compile(self.encoder, mode=mode)
            print("  Compiling decoder...")
            self.decoder = torch.compile(self.decoder, mode=mode)
            print("✓ Model compilation successful")
        except Exception as e:
            print(f"Model compilation failed: {e}")
            print("Falling back to uncompiled model...")
            # Don't re-raise the exception, just continue with uncompiled model
    
    def to(self, *args, **kwargs):
        """Override to method to optimize memory format for better performance.
        
        Automatically converts the model to channels_last memory format when moving
        to GPU, which can provide significant performance improvements on modern GPUs.
        
        Args:
            *args: Positional arguments passed to parent to() method
            **kwargs: Keyword arguments passed to parent to() method
            
        Returns:
            VAE: The model moved to the specified device with optimized memory format
        """
        device = args[0] if args else kwargs.get('device', None)
        if device:
            # Convert to channels_last memory format for better performance
            print("Converting model to channels_last memory format for better performance")
            self = super().to(*args, **kwargs)
            self = self.to(memory_format=torch.channels_last)
            return self
        return super().to(*args, **kwargs)