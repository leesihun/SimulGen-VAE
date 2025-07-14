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

    def forward(self, x):
        # Always use regular forward pass - no speed trade-offs
        mu, log_var, xs = self.encoder(x)
        
        # Clamp log_var to prevent numerical instability before computing std
        log_var = torch.clamp(log_var, min=-20, max=20)
        std = torch.exp(0.5*log_var)
        z = reparameterize(mu, std)
        
        decoder_output, kl_losses = self.decoder(z, xs)

        if self.lossfun == 'MSE':
            recon_loss = nn.MSELoss()(decoder_output, x)
        elif self.lossfun == 'MAE':
            recon_loss = nn.L1Loss()(decoder_output, x)
        elif self.lossfun == 'smoothL1':
            recon_loss = nn.SmoothL1Loss()(decoder_output, x)
        elif self.lossfun == 'Huber':
            recon_loss = nn.HuberLoss()(decoder_output, x)

        recon_loss_MSE = nn.MSELoss()(decoder_output, x)

        kl_loss = kl(mu, log_var)

        # Clean up intermediate variables to free memory faster
        del mu, log_var, xs, z
        return decoder_output, recon_loss, [kl_loss]+kl_losses, recon_loss_MSE