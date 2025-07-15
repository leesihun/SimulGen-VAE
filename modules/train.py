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
        loss = 0
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

    # Improved beta scheduling for better VAE training
    init_beta = 1e-8  # Start even lower
    beta_target = 5e-4  # Lower target for better reconstruction
    epoch = epochs
    start_warmup = int(epoch*0.1)  # Start warmup earlier
    end_warmup = int(epoch*0.8)    # End warmup later for more gradual increase

    warmup_kl = WarmupKLLoss(epoch, init_beta, start_warmup, end_warmup, beta_target)

    model.to(device)
    
    # Compile model for better performance on consistent input sizes
    # Use 'default' mode for safer compilation, or False to disable
    model.compile_model(mode='default')

    # Add weight decay for better generalization
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-5)
    # Use a more sophisticated scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=epoch//4, T_mult=2, eta_min=LR*0.01
    )
    
    # Initialize GradScaler for mixed precision training
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
    
    # Enable memory efficient attention and convolutions
    torch.backends.cudnn.benchmark = True  # Optimize for consistent input sizes
    torch.backends.cuda.matmul.allow_tf32 = True  # Faster on Ampere GPUs (H100)
    torch.backends.cudnn.allow_tf32 = True
    
    # Additional optimizations for small variety datasets
    if torch.cuda.is_available():
        # Enable persistent RNN for consistent sequence lengths
        torch.backends.cudnn.deterministic = False  # Allow non-deterministic ops for speed
        torch.backends.cudnn.enabled = True
        
        # Set memory allocation strategy for better performance
        torch.cuda.set_per_process_memory_fraction(0.95)  # Use most GPU memory
    
    # Pre-allocate CUDA streams for better async execution
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
        # Create a stream for async data transfers
        transfer_stream = torch.cuda.Stream()
    else:
        transfer_stream = None
        
    # Pre-allocate tensors for loss accumulation (memory efficiency)
    loss_accumulator = torch.zeros(1, device=device)
    recon_accumulator = torch.zeros(1, device=device)
    kl_accumulator = torch.zeros(1, device=device)
    mse_accumulator = torch.zeros(1, device=device)
    
    # CUDA Graphs setup
    use_cuda_graphs = torch.cuda.is_available() and hasattr(torch.cuda, 'CUDAGraph')
    cuda_graph_batch = None
    cuda_graph = None
    static_input = None
    static_output = None
    static_recon_loss = None
    static_kl_losses = None
    static_recon_loss_MSE = None
    
    # Initialize CUDA graph after a few warm-up iterations
    cuda_graph_warmup = 10
    cuda_graph_initialized = False

    for epoch in range(epochs):
        start_time = time.time()
        model.train(True)
        
        # Reset accumulators
        loss_accumulator.zero_()
        recon_accumulator.zero_()
        kl_accumulator.zero_()
        mse_accumulator.zero_()
        
        # Clear cache at start of each epoch
        torch.cuda.empty_cache()

        for i, image in enumerate(train_dataloader):
            if load_all == False:
                # Use non_blocking=True for async GPU transfer when using pinned memory
                image = image.to(device, non_blocking=True)

            # Initialize CUDA graph after warm-up iterations
            if use_cuda_graphs and not cuda_graph_initialized and i >= cuda_graph_warmup:
                print("Initializing CUDA graph for faster training...")
                # Capture a static input with the same shape
                static_input = image.detach().clone()
                cuda_graph_batch = image.shape[0]  # Remember the batch size
                cuda_graph = torch.cuda.CUDAGraph()
                
                # Prepare for graph capture
                optimizer.zero_grad(set_to_none=True)
                
                with torch.cuda.graph(cuda_graph):
                    with autocast():
                        static_output, static_recon_loss, static_kl_losses, static_recon_loss_MSE = model(static_input)
                
                cuda_graph_initialized = True
                print("CUDA graph initialized successfully")

            # Use CUDA graph if initialized and batch size matches
            if use_cuda_graphs and cuda_graph_initialized and image.shape[0] == cuda_graph_batch:
                # Copy input data to static tensor
                static_input.copy_(image)
                
                # Execute the captured graph
                cuda_graph.replay()
                
                # Get results from static outputs
                recon_loss = static_recon_loss
                kl_losses = static_kl_losses
                recon_loss_MSE = static_recon_loss_MSE
            else:
                # Regular forward pass for variable batch sizes or before graph initialization
                optimizer.zero_grad(set_to_none=True)
                
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
            
            # Gradient clipping to prevent explosion
            scaler.unscale_(optimizer)
            # Start with 5.0, reduce to 1.0 only if you still get NaN
            # For VAEs, 2.0-5.0 is often a good balance
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            
            # Mixed precision optimizer step
            scaler.step(optimizer)
            scaler.update()

            # Accumulate losses efficiently
            loss_accumulator += loss.detach()
            recon_accumulator += recon_loss.detach()
            kl_accumulator += kl_loss.detach()
            mse_accumulator += recon_loss_MSE.detach()

            # More efficient memory cleanup - delete in reverse order of creation
            del loss, recon_loss_MSE, kl_loss, recon_loss, kl_losses, image
        num = i

        # Convert accumulated losses to final values
        loss_save = loss_accumulator.item()
        recon_loss_save = recon_accumulator.item()
        kl_loss_save = kl_accumulator.item()
        recon_loss_MSE_save = mse_accumulator.item()

        
        
        # Run validation every 100 epochs or on the last epoch
        if epoch % 100 == 0 or epoch == epochs - 1:

            # Validation loop - run only every 100 epochs
            model.eval()
            val_batches_processed = 0
            recon_loss_save_val = 0.0
            loss_save_val = 0.0
            
            for i, image in enumerate(val_dataloader):
                with torch.no_grad():
                    if load_all ==False:
                        # Use non_blocking=True for async GPU transfer when using pinned memory
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
        else:
            # For non-validation epochs, use previous validation values
            if epoch > 0:
                loss_val_print[epoch] = loss_val_print[epoch - 1]
                recon_loss_val_print[epoch] = recon_loss_val_print[epoch - 1]
            else:
                loss_val_print[epoch] = 0.0
                recon_loss_val_print[epoch] = 0.0

        loss_print[epoch] = loss_save/(num+1)
        recon_print[epoch] = recon_loss_save/(num+1)
        kl_print[epoch] = kl_loss_save/beta/(num+1)
        recon_loss_MSE_print[epoch] = recon_loss_MSE_save/(num+1)

        current_lr = optimizer.param_groups[0]['lr']

        loss_plot[epoch] = loss_print[epoch]

        scheduler.step()

        end_time = time.time()
        epoch_duration = end_time - start_time

        # if epoch % 100 == 0:
        #     # writer.add_scalar('Loss/train', loss_print[epoch], epoch)
        #     # Only log validation loss when it's actually computed
        #     if val_batches_processed > 0:
        #         # writer.add_scalar('Loss/val', loss_val_print[epoch], epoch)
        #         pass # Commented out as per edit hint

        log_str = "\r[Epoch {}/{}] Loss: {:.4E}   val_loss: {:.2E}   Recon:{:.4E}   Recon_val:{:.4E}   KL:{:.4E}   Beta:{:.4E}   Time: {:.2f}s   ETA: {:.2f}h    LR: {:.2E}".format(
            epoch+1, epochs, loss_print[epoch], loss_val_print[epoch], recon_print[epoch], recon_loss_val_print[epoch], kl_print[epoch], beta, epoch_duration, (epochs-epoch)*epoch_duration/3600, current_lr
        )

        logging.info(log_str)

    # Clean up CUDA graph resources
    if use_cuda_graphs and cuda_graph_initialized:
        del static_input, static_output, static_recon_loss, static_kl_losses, static_recon_loss_MSE
        del cuda_graph
        torch.cuda.empty_cache()

    torch.save(model.state_dict(), 'checkpoints/SimulGen-VAE.pth')
    torch.save(model, 'model_save/SimulGen-VAE')
    torch.cuda.empty_cache()

    return loss_print, recon_print, kl_print, loss_val_print
    