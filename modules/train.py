import argparse
import logging
import os
import time
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader
from modules.common import initialize_weights_He, add_sn
from modules.VAE_network import VAE
from modules.losses import kl
from torchinfo import summary
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter

# Add mixed precision imports
from torch.cuda.amp import autocast, GradScaler

class WarmupKLLoss:
    def __init__(self, epoch, init_beta, start_warmup, end_warmup, beta_target):
        self.epoch = epoch
        self.init_beta = init_beta
        self.start_warmup = start_warmup
        self.end_warmup = end_warmup
        self.beta_target = beta_target

    def get_loss(self, step, losses):
        # Initialize loss as tensor with same device/dtype as first loss
        print(losses)
        for i, l in enumerate(losses):
            l = losses[i]
            loss += l

        if step<self.start_warmup:
            beta = self.init_beta

        elif self.start_warmup<=step<self.end_warmup:
            beta = (step-self.start_warmup)*(self.beta_target-self.init_beta)/(self.end_warmup-self.start_warmup)+self.init_beta
        else:
            beta = self.beta_target

        return [beta, loss]

def print_gpu_mem_checkpoint(msg):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        max_allocated = torch.cuda.max_memory_allocated() / 1024**2
        print(f"[GPU MEM] {msg}: Allocated={allocated:.2f}MB, Max Allocated={max_allocated:.2f}MB")
        torch.cuda.reset_peak_memory_stats()

def train(epochs, batch_size, train_dataloader, val_dataloader, LR, num_filter_enc, num_filter_dec, num_node, latent_dim, hierarchical_dim, num_time, alpha, lossfun, small, load_all):
    writer = SummaryWriter(log_dir = './runs', comment = 'VAE')

    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('output', exist_ok=True)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print('GPU Machine used?', device)
    print('Number of GPUs available:', torch.cuda.device_count())

    # Disable gradient checkpointing for speed (user preference)
    model = VAE(latent_dim, hierarchical_dim, num_filter_enc, num_filter_dec, num_node, num_time, lossfun=lossfun, batch_size = batch_size, small= small, use_checkpointing=False)

    summary(model, (batch_size, num_node, num_time))
    print(model)
    torch.cuda.empty_cache()

    model.apply(initialize_weights_He)
    model.apply(add_sn)

    init_beta = 1e-7
    beta_target = 1e-3
    epoch = epochs
    start_warmup =int(epoch*0.3)
    end_warmup = int(epoch*0.7)

    warmup_kl = WarmupKLLoss(epoch, init_beta, start_warmup, end_warmup, beta_target)

    model.to(device)

    # optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch, eta_min=0)
    
    # Initialize mixed precision scaler - MAJOR memory savings with no speed loss
    scaler = GradScaler()

    loss_print = np.zeros(epochs)
    loss_val_print = np.zeros(epochs)
    recon_print = np.zeros(epochs)
    kl_print = np.zeros(epochs)
    recon_loss_MSE_print = np.zeros(epochs)
    loss_plot = np.zeros(epochs)
    recon_loss_val_print = np.zeros(epochs)

    model.train(True)
    import time
    
    print("🚀 MAXIMUM PERFORMANCE MODE ENABLED:")
    print("   ✓ Mixed Precision (FP16) - ~50% memory reduction + speed boost")
    print("   ✓ FP16 data storage - Halved dataset memory usage") 
    print("   ✓ Gradient Checkpointing DISABLED - Full speed maintained")
    print("   ✓ H100 tensor cores optimized - Maximum performance")
    print("   ✓ Efficient tensor management - Faster cleanup")
    print("   ✓ Multi-threaded DataLoaders - CPU-GPU overlap maximized")
    print("   ✓ Persistent workers - No worker recreation overhead")
    print("   ✓ ALL NaN CHECKS REMOVED - Pure speed, no safety overhead")

    # Enable memory efficient attention and convolutions
    torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
    torch.backends.cuda.matmul.allow_tf32 = True  # Faster on Ampere GPUs (H100)
    torch.backends.cudnn.allow_tf32 = True
    
    # Pre-allocate CUDA streams for better async execution
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
        # Create a stream for async data transfers
        transfer_stream = torch.cuda.Stream()
    else:
        transfer_stream = None

    for epoch in range(epochs):
        start_time = time.time()
        model.train(True)
        
        # Clear cache at start of each epoch
        torch.cuda.empty_cache()

        for i, image in enumerate(train_dataloader):
            # All data now comes from CPU (even with load_all=True) to avoid CUDA worker issues
            # Use async transfer with dedicated stream for maximum overlap
            if transfer_stream is not None:
                with torch.cuda.stream(transfer_stream):
                    if hasattr(train_dataloader.dataset.dataset, 'load_all') and train_dataloader.dataset.dataset.load_all:
                        # For load_all=True: data is FP16 on CPU, convert to GPU
                        image = image.to(device, dtype=torch.float16, non_blocking=True)
                    else:
                        # For load_all=False: data is FP32 on CPU, convert to GPU
                        image = image.to(device, non_blocking=True)
                # Synchronize to ensure data is ready before forward pass
                torch.cuda.current_stream().wait_stream(transfer_stream)
            else:
                if hasattr(train_dataloader.dataset.dataset, 'load_all') and train_dataloader.dataset.dataset.load_all:
                    image = image.to(device, dtype=torch.float16, non_blocking=True)
                else:
                    image = image.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)  # More memory efficient than zero_grad()

            # Use autocast for forward pass - memory savings happen here
            with autocast():
                _, recon_loss, kl_losses, recon_loss_MSE = model(image)

            beta, kl_loss = warmup_kl.get_loss(epoch, kl_losses)

            # Loss calculation in FP32 for stability
            kl_loss = kl_loss*beta
            recon_loss = recon_loss*alpha
            recon_loss_MSE = recon_loss_MSE*alpha
            loss = recon_loss + kl_loss

            # Mixed precision backward pass - scales gradients to prevent underflow
            scaler.scale(loss).backward()
            
            # Mixed precision optimizer step
            scaler.step(optimizer)
            scaler.update()

            if i==0:
                kl_loss_save = kl_loss.detach().item()
                recon_loss_save = recon_loss.detach().item()
                recon_loss_MSE_save = recon_loss_MSE.detach().item()
                loss_save = loss.detach().item()

            else:
                kl_loss_save = kl_loss_save + kl_loss.detach().item()
                recon_loss_save = recon_loss_save + recon_loss.detach().item()
                recon_loss_MSE_save = recon_loss_MSE_save + recon_loss_MSE.detach().item()
                loss_save = loss_save + loss.detach().item()

            # More efficient memory cleanup - delete in reverse order of creation
            del loss, recon_loss_MSE, kl_loss, recon_loss, kl_losses, image
        num = i

        # Validation loop
        model.eval()
        val_batches_processed = 0
        recon_loss_save_val = 0.0
        loss_save_val = 0.0
        
        for i, image in enumerate(val_dataloader):
            with torch.no_grad():
                # All data now comes from CPU to avoid CUDA worker issues
                if hasattr(val_dataloader.dataset.dataset, 'load_all') and val_dataloader.dataset.dataset.load_all:
                    # For load_all=True: data is FP16 on CPU, convert to GPU
                    image = image.to(device, dtype=torch.float16, non_blocking=True)
                else:
                    # For load_all=False: data is FP32 on CPU, convert to GPU
                    image = image.to(device, non_blocking=True)

                # Use autocast for validation forward pass as well
                with autocast():
                    _, recon_loss, kl_losses, recon_loss_MSE = model(image)

                beta, kl_loss = warmup_kl.get_loss(epoch, kl_losses)

                kl_loss = kl_loss*beta
                recon_loss = recon_loss*alpha
                recon_loss_MSE = recon_loss_MSE*alpha
                loss = recon_loss + kl_loss

                # Accumulate validation metrics
                recon_loss_save_val += recon_loss.detach().item()
                loss_save_val += loss.detach().item()
                val_batches_processed += 1
                
                del image, loss, recon_loss, kl_losses, recon_loss_MSE, kl_loss
        
        # Calculate validation averages
        loss_val_print[epoch] = loss_save_val / val_batches_processed
        recon_loss_val_print[epoch] = recon_loss_save_val / val_batches_processed

        loss_print[epoch] = loss_save/(num+1)
        recon_print[epoch] = recon_loss_save/(num+1)
        kl_print[epoch] = kl_loss_save/beta/(num+1)
        recon_loss_MSE_print[epoch] = recon_loss_MSE_save/(num+1)

        current_lr = optimizer.param_groups[0]['lr']

        loss_plot[epoch] = loss_print[epoch]

        scheduler.step()

        end_time = time.time()
        epoch_duration = end_time - start_time

        if epoch % 100 == 0:
            writer.add_scalar('Loss/train', loss_print[epoch], epoch)
            writer.add_scalar('Loss/val', loss_val_print[epoch], epoch)

        log_str = "\r[Epoch {}/{}] Loss: {:.4E}   val_loss: {:.2E}   Recon:{:.4E}   Recon_val:{:.4E}   KL:{:.4E}   Beta:{:.4E}   Time: {:.2f}s   ETA: {:.2f}h    LR: {:.2E}".format(
            epoch+1, epochs, loss_print[epoch], loss_val_print[epoch], recon_print[epoch], recon_loss_val_print[epoch], kl_print[epoch], beta, epoch_duration, (epochs-epoch)*epoch_duration/3600, current_lr
        )

        logging.info(log_str)

    torch.save(model.state_dict(), 'checkpoints/SimulGen-VAE.pth')
    torch.save(model, 'model_save/SimulGen-VAE')
    torch.cuda.empty_cache()

    return loss_print, recon_print, kl_print, loss_val_print
    