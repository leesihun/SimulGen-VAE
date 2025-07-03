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
    def __init__(self, latent_dim, hierarchical_dim, num_filter_enc, num_filter_dec, num_node, num_time, lossfun='MSE', batch_size=1, small=False):
        super().__init__()

        self.latent_dim = latent_dim
        self.encoder = Encoder(latent_dim, hierarchical_dim, num_filter_enc, num_node, num_time, small)
        self.decoder = Decoder(latent_dim, hierarchical_dim, num_filter_dec, num_node, num_time, batch_size, small)
        self.lossfun = lossfun

    def forward(self, x):
        mu, log_var, xs=self.encoder(x)
        z = reparameterize(mu, torch.exp(0.5*log_var))

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

        del mu, log_var, xs, z
        return decoder_output, recon_loss, [kl_loss]+kl_losses, recon_loss_MSE