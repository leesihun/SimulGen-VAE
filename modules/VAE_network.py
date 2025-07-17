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
        """
        Compile the model for better performance on consistent input sizes.
        Call this after moving to GPU and before training.
        
        Args:
            mode: Compilation mode. Can be:
                - 'default': Conservative compilation (recommended)
                - 'reduce-overhead': Faster compilation time
                - 'max-autotune': Most aggressive (may fail on some models)
                - False: Skip compilation entirely
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
        """Override to method to convert to channels_last memory format"""
        device = args[0] if args else kwargs.get('device', None)
        if device:
            # Convert to channels_last memory format for better performance
            print("Converting model to channels_last memory format for better performance")
            self = super().to(*args, **kwargs)
            self = self.to(memory_format=torch.channels_last)
            return self
        return super().to(*args, **kwargs)