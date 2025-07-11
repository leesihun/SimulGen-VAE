import torch.nn as nn
from modules.common import *
from torch.nn import functional as F
from modules.losses import kl, kl_2
import torch

class UpsampleBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()

        self._seq = nn.Sequential(
            nn.ConvTranspose1d(in_channel, out_channel, kernel_size=3, padding=1),
            nn.GELU(),
        )

    def forward(self, x):
        return self._seq(x)

class DecoderBlock(nn.Module):
    def __init__(self, channels, small):
        super().__init__()
        self.channels = channels
        modules = []

        for i in range(len(channels) - 1):
            modules.append(UpsampleBlock(channels[i], channels[i + 1]))
        self.module_list = nn.ModuleList(modules)

    def forward(self, x):
        for module in self.module_list:
            x = module(x)

        return x

class Decoder(nn.Module):
    def __init__(self, z_dim, hierarchical_dim, num_filter_dec, num_node, num_time, batch_size, small):
        super().__init__()

        self.decoder_blocks = nn.ModuleList([])
        for i in range(len(num_filter_dec)-1):
            self.decoder_blocks.append(DecoderBlock([num_filter_dec[i], num_filter_dec[i+1]], small))

        self.decoder_residual_blocks = nn.ModuleList([])
        for i in range(len(num_filter_dec)-1):
            self.decoder_residual_blocks.append(DecoderResidualBlock(num_filter_dec[i+1], small))

        self.recon = nn.Sequential(
            nn.Conv1d(num_filter_dec[-1], num_node, kernel_size=1),
            nn.BatchNorm1d(num_node),
            nn.Tanh()
        )

        self.zs = []
        self.num_filter_dec = num_filter_dec
        self.num_time = num_time

        latent_dim = z_dim
        hierarchical_dim = hierarchical_dim
        batch_size = batch_size

        self.sequence_start = nn.ModuleList([])
        self.sequence_start.append(nn.Sequential(
            nn.Linear(latent_dim, latent_dim*num_time),
            nn.Unflatten(1, (latent_dim, num_time)),
            nn.Conv1d((latent_dim), self.num_filter_dec[0], kernel_size=5, padding=2),
            nn.BatchNorm1d(self.num_filter_dec[0]),
            nn.GELU(),
        ))

        self.xs_sequence = nn.ModuleList([])
        for i in range(len(num_filter_dec)-1):
            self.xs_sequence.append(nn.Sequential(
                nn.Linear(hierarchical_dim, hierarchical_dim*num_time),
                nn.Unflatten(1, (hierarchical_dim, num_time)),
                nn.Conv1d(hierarchical_dim, self.num_filter_dec[i+1], kernel_size=5, padding=2),
                nn.BatchNorm1d(self.num_filter_dec[i+1]),
                nn.GELU(),
            ))

        self.condition_z = nn.ModuleList([])
        for i in range(len(num_filter_dec)-1):
            self.condition_z.append(nn.Sequential(
                ResidualBlock(num_filter_dec[i+1], small),
                nn.GELU(),
                nn.Conv1d(num_filter_dec[i+1], 2*num_filter_dec[i+1], kernel_size=3, padding=1),
                # nn.BatchNorm1d(2*num_filter_dec[i+1]),
            ))

        self.condition_xz = nn.ModuleList([])
        for i in range(len(num_filter_dec)-1):
            self.condition_xz.append(nn.Sequential(
                ResidualBlock(2*num_filter_dec[i+1], small),
                nn.GELU(),
                nn.Conv1d(2*num_filter_dec[i+1], 2*num_filter_dec[i+1], kernel_size=3, padding=1),
                # nn.BatchNorm1d(2*num_filter_dec[i+1]),
            ))

        self.small = small

    def forward(self, z, xs=None, mode = "random", freeze_level = -1):

        kl_losses = []

        for i in range(len(self.decoder_residual_blocks)):
            if i==0:
                z_sample = self.sequence_start[0](z)

            else:
                z_sample = torch.add(decoder_out, z)

            decoder_out = self.decoder_blocks[i](z_sample)
            decoder_out = self.decoder_residual_blocks[i](decoder_out)

            if i==len(self.decoder_residual_blocks)-1:
                break

            mu, log_var = self.condition_z[i](decoder_out).chunk(2, dim=1)

            if xs is not None:
                xs_sample = self.xs_sequence[i](xs[i])

                delta_mu, delta_log_var = self.condition_xz[i](torch.cat([xs_sample, decoder_out], dim=1)).chunk(2, dim=1)
                kl_losses.append(kl_2(delta_mu, delta_log_var, mu, log_var))

                mu = mu + delta_mu
                log_var = log_var + delta_log_var

                if mode=="fix" and i<freeze_level:
                    if len(self.zs) < freeze_level+1:
                        z = reparameterize(mu, 1e-10*torch.exp(0.5*log_var))
                        self.zs.append(z)
                    else:
                        z = self.zs[i+1]

                elif mode== "fix":
                    z= reparameterize(mu, 1e-10*torch.exp(0.5*log_var))
                else:
                    z=reparameterize(mu, torch.exp(0.5*log_var))

        x_hat = self.recon(decoder_out)

        return x_hat, kl_losses

def reparameterize(mu, std):
    # Clamp mu and std to prevent extreme values
    # mu = torch.clamp(mu, min=-10, max=10)
    # std = torch.clamp(std, min=1e-8, max=10)
    
    eps = torch.randn_like(std)
    z = eps.mul(std).add_(mu)
    
    # Additional safety check
    # z = torch.clamp(z, min=-10, max=10)
    
    return z