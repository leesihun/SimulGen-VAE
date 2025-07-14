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

class WarmupKLLoss:
    def __init__(self, epoch, init_beta, start_warmup, end_warmup, beta_target):
        self.epoch = epoch
        self.init_beta = init_beta
        self.start_warmup = start_warmup
        self.end_warmup = end_warmup
        self.beta_target = beta_target

    def get_loss(self, step, losses):
        loss = 0.
        for i, l in enumerate(losses):
            l = losses[i]
            loss+=l

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
        cached = torch.cuda.memory_reserved() / 1024**2
        print(f"[GPU MEM] {msg}")
        print(f"  Allocated: {allocated:.2f}MB")
        print(f"  Max Allocated: {max_allocated:.2f}MB") 
        print(f"  Cached: {cached:.2f}MB")
        print(f"  Free: {cached - allocated:.2f}MB")
        torch.cuda.reset_peak_memory_stats()

def detailed_memory_analysis(batch_data, model=None):
    """Detailed analysis of memory usage for input batch and model"""
    if not torch.cuda.is_available():
        print("[MEM] CUDA not available for memory analysis")
        return
        
    print(f"\n=== DETAILED MEMORY ANALYSIS ===")
    
    # Analyze input batch
    if isinstance(batch_data, torch.Tensor):
        batch_size_mb = batch_data.element_size() * batch_data.nelement() / 1024**2
        print(f"Input Batch:")
        print(f"  Shape: {batch_data.shape}")
        print(f"  Dtype: {batch_data.dtype}")
        print(f"  Size: {batch_size_mb:.2f}MB")
        print(f"  Elements: {batch_data.nelement():,}")
    
    # Current GPU memory state
    allocated = torch.cuda.memory_allocated() / 1024**2
    max_allocated = torch.cuda.max_memory_allocated() / 1024**2
    cached = torch.cuda.memory_reserved() / 1024**2
    
    print(f"GPU Memory State:")
    print(f"  Currently Allocated: {allocated:.2f}MB")
    print(f"  Peak Allocated: {max_allocated:.2f}MB")
    print(f"  Reserved: {cached:.2f}MB")
    print(f"  Free GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**2 - cached:.2f}MB")
    
    # Model parameter memory if model provided
    if model is not None:
        total_params = sum(p.numel() for p in model.parameters())
        param_memory = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        print(f"Model Parameters:")
        print(f"  Total Parameters: {total_params:,}")
        print(f"  Trainable Parameters: {trainable_params:,}")
        print(f"  Parameter Memory: {param_memory:.2f}MB")
    
    print(f"=================================\n")

def monitor_forward_pass_memory(model, image, step_name="Forward"):
    """Monitor memory usage during forward pass"""
    print(f"\n--- {step_name} Pass Memory Monitoring ---")
    
    print_gpu_mem_checkpoint(f"Before {step_name}")
    
    # Monitor input
    input_mem = image.element_size() * image.nelement() / 1024**2
    print(f"Input tensor: {image.shape} = {input_mem:.2f}MB")
    
    # Execute forward pass
    with torch.cuda.device(image.device):
        torch.cuda.synchronize()
        start_mem = torch.cuda.memory_allocated() / 1024**2
        
        output = model(image)
        
        torch.cuda.synchronize()
        end_mem = torch.cuda.memory_allocated() / 1024**2
        peak_mem = torch.cuda.max_memory_allocated() / 1024**2
    
    print(f"Memory change during {step_name}: {end_mem - start_mem:.2f}MB")
    print(f"Peak memory during {step_name}: {peak_mem:.2f}MB")
    
    # Monitor output
    if isinstance(output, tuple) and len(output) > 0:
        if isinstance(output[0], torch.Tensor):
            output_mem = output[0].element_size() * output[0].nelement() / 1024**2
            print(f"Output tensor: {output[0].shape} = {output_mem:.2f}MB")
    
    print_gpu_mem_checkpoint(f"After {step_name}")
    
    return output

def monitor_training_step_memory(model, image, optimizer, step_idx, epoch):
    """Comprehensive memory monitoring for a complete training step"""
    print(f"\n{'='*60}")
    print(f"MEMORY MONITORING - Epoch {epoch}, Step {step_idx}")
    print(f"{'='*60}")
    
    # Initial state
    detailed_memory_analysis(image, model)
    
    # Clear gradients
    optimizer.zero_grad()
    print_gpu_mem_checkpoint("After zero_grad()")
    
    # Forward pass with detailed monitoring
    _, recon_loss, kl_losses, recon_loss_MSE = monitor_forward_pass_memory(model, image, "Forward")
    
    # Loss computation monitoring
    print_gpu_mem_checkpoint("After loss computation")
    
    return recon_loss, kl_losses, recon_loss_MSE

def stabilize_batchnorm(model):
    """Stabilize BatchNorm running statistics to prevent NaN during evaluation"""
    for module in model.modules():
        if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            # Clamp running mean and var to prevent extreme values
            if module.running_mean is not None:
                module.running_mean.data = torch.clamp(module.running_mean.data, min=-10, max=10)
                # Replace NaN with zeros
                module.running_mean.data = torch.nan_to_num(module.running_mean.data, nan=0.0)
            
            if module.running_var is not None:
                module.running_var.data = torch.clamp(module.running_var.data, min=1e-8, max=100)
                # Replace NaN with small positive values
                module.running_var.data = torch.nan_to_num(module.running_var.data, nan=1.0)

def train(epochs, batch_size, train_dataloader, val_dataloader, LR, num_filter_enc, num_filter_dec, num_node, latent_dim, hierarchical_dim, num_time, alpha, lossfun, small, load_all):
    writer = SummaryWriter(log_dir = './runs', comment = 'VAE')

    LOG_FORMAT = "%(asctime)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=LOG_FORMAT)

    os.makedirs('checkpoints', exist_ok=True)
    os.makedirs('output', exist_ok=True)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print('GPU Machine used?', device)
    print('Number of GPUs available:', torch.cuda.device_count())

    model = VAE(latent_dim, hierarchical_dim, num_filter_enc, num_filter_dec, num_node, num_time, lossfun=lossfun, batch_size = batch_size, small= small)

    summary(model, (batch_size, num_node, num_time))
    print(model)
    torch.cuda.empty_cache()

    model.apply(initialize_weights_He)
    model.apply(add_sn)

    init_beta = 1e-8
    beta_target = 1e-4
    epoch = epochs
    start_warmup =int(epoch*0.3)
    end_warmup = int(epoch*0.7)

    warmup_kl = WarmupKLLoss(epoch, init_beta, start_warmup, end_warmup, beta_target)

    model.to(device)
    print_gpu_mem_checkpoint("After model.to(device)")
    
    # Analyze model memory usage
    print(f"\n=== MODEL MEMORY ANALYSIS ===")
    total_params = sum(p.numel() for p in model.parameters())
    param_memory = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024**2
    print(f"Total parameters: {total_params:,}")
    print(f"Parameter memory: {param_memory:.2f}MB")
    print(f"Model size estimation: ~{param_memory * 4:.2f}MB (with gradients + optimizer states)")
    print(f"==============================\n")

    # optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epoch, eta_min=0)
    print_gpu_mem_checkpoint("After optimizer creation")

    loss_print = np.zeros(epochs)
    loss_val_print = np.zeros(epochs)
    recon_print = np.zeros(epochs)
    kl_print = np.zeros(epochs)
    recon_loss_MSE_print = np.zeros(epochs)
    loss_plot = np.zeros(epochs)
    recon_loss_val_print = np.zeros(epochs)

    model.train(True)
    import time
    
    # Memory monitoring settings
    DETAILED_MONITORING_STEPS = 3  # Monitor first 3 steps in detail
    monitoring_step_count = 0

    for epoch in range(epochs):
        start_time = time.time()
        model.train(True)

        for i, image in enumerate(train_dataloader):
            if load_all==False:
                image = image.to(device)

            # Detailed memory monitoring for first few steps
            if monitoring_step_count < DETAILED_MONITORING_STEPS:
                _, recon_loss, kl_losses, recon_loss_MSE = monitor_training_step_memory(model, image, optimizer, i, epoch)
                monitoring_step_count += 1
            else:
                optimizer.zero_grad()
                _, recon_loss, kl_losses, recon_loss_MSE = model(image)
                
                # Quick memory check every 10 steps
                if i % 10 == 0:
                    print_gpu_mem_checkpoint(f"Epoch {epoch}, Step {i} - Quick check")

            # Check for NaN in model outputs
            if torch.isnan(recon_loss) or torch.isinf(recon_loss):
                print(f"Warning: NaN/Inf in recon_loss at epoch {epoch}, batch {i}")
                print(f"Input range: {image.min().item():.4f} to {image.max().item():.4f}")
                continue
                
            # Check KL losses for NaN
            for idx, kl_loss_item in enumerate(kl_losses):
                if torch.isnan(kl_loss_item) or torch.isinf(kl_loss_item):
                    print(f"Warning: NaN/Inf in kl_loss[{idx}] at epoch {epoch}, batch {i}")
                    continue

            beta, kl_loss = warmup_kl.get_loss(epoch, kl_losses)

            # Additional checks after scaling
            if torch.isnan(kl_loss) or torch.isinf(kl_loss):
                print(f"Warning: NaN/Inf in combined kl_loss at epoch {epoch}, batch {i}")
                continue

            kl_loss = kl_loss*beta
            recon_loss = recon_loss*alpha
            recon_loss_MSE = recon_loss_MSE*alpha
            loss = recon_loss + kl_loss

            # Check for NaN values before backpropagation
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: NaN or Inf detected in loss at epoch {epoch}, batch {i}")
                print(f"recon_loss: {recon_loss.item()}, kl_loss: {kl_loss.item()}")
                print(f"beta: {beta}, alpha: {alpha}")
                continue

            # Monitor backward pass for first few steps
            if monitoring_step_count <= DETAILED_MONITORING_STEPS:
                print_gpu_mem_checkpoint("Before backward()")
            
            loss.backward()
            
            if monitoring_step_count <= DETAILED_MONITORING_STEPS:
                print_gpu_mem_checkpoint("After backward()")
            
            # Add gradient clipping to prevent exploding gradients
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            if monitoring_step_count <= DETAILED_MONITORING_STEPS:
                print_gpu_mem_checkpoint("After optimizer.step()")
                print("--- End of detailed step monitoring ---\n")

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

            del image, loss
            del recon_loss, kl_losses, recon_loss_MSE, kl_loss
        num = i

        # Check model parameters for NaN before validation
        from modules.utils import check_model_for_nan
        if check_model_for_nan(model, f"Model at epoch {epoch}"):
            print(f"Critical: Model parameters contain NaN at epoch {epoch}. Stopping training.")
            break
        
        # Stabilize BatchNorm statistics before evaluation
        stabilize_batchnorm(model)
        
        # Validation loop with NaN checking
        model.eval()  # Move outside the loop for efficiency
        val_batches_processed = 0
        recon_loss_save_val = 0.0
        loss_save_val = 0.0
        
        for i, image in enumerate(val_dataloader):
            with torch.no_grad():
                if load_all==False:
                    image = image.to(device)

                _, recon_loss, kl_losses, recon_loss_MSE = model(image)

                # Check for NaN in validation outputs
                if torch.isnan(recon_loss) or torch.isinf(recon_loss):
                    print(f"Warning: NaN/Inf in validation recon_loss at epoch {epoch}, batch {i}")
                    print(f"Validation input range: {image.min().item():.4f} to {image.max().item():.4f}")
                    continue
                    
                # Check validation KL losses for NaN
                kl_has_nan = False
                for idx, kl_loss_item in enumerate(kl_losses):
                    if torch.isnan(kl_loss_item) or torch.isinf(kl_loss_item):
                        print(f"Warning: NaN/Inf in validation kl_loss[{idx}] at epoch {epoch}, batch {i}")
                        kl_has_nan = True
                        break
                
                if kl_has_nan:
                    continue

                beta, kl_loss = warmup_kl.get_loss(epoch, kl_losses)

                # Additional checks after scaling
                if torch.isnan(kl_loss) or torch.isinf(kl_loss):
                    print(f"Warning: NaN/Inf in validation combined kl_loss at epoch {epoch}, batch {i}")
                    continue

                kl_loss = kl_loss*beta
                recon_loss = recon_loss*alpha
                recon_loss_MSE = recon_loss_MSE*alpha
                loss = recon_loss + kl_loss

                # Check final validation loss for NaN
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Warning: NaN or Inf detected in validation loss at epoch {epoch}, batch {i}")
                    print(f"val_recon_loss: {recon_loss.item()}, val_kl_loss: {kl_loss.item()}")
                    print(f"beta: {beta}, alpha: {alpha}")
                    continue

                # Only accumulate if no NaN detected
                recon_loss_save_val += recon_loss.detach().item()
                loss_save_val += loss.detach().item()
                val_batches_processed += 1
                
                del image, loss
                del recon_loss, kl_losses, recon_loss_MSE, kl_loss
        
        # Handle case where all validation batches had NaN
        if val_batches_processed == 0:
            print(f"Warning: All validation batches had NaN at epoch {epoch}")
            # Use previous epoch's validation loss or set to a large value
            if epoch > 0:
                loss_val_print[epoch] = loss_val_print[epoch-1]
                recon_loss_val_print[epoch] = recon_loss_val_print[epoch-1]
            else:
                loss_val_print[epoch] = float('inf')
                recon_loss_val_print[epoch] = float('inf')
        else:
            loss_val_print[epoch] = loss_save_val / val_batches_processed
            recon_loss_val_print[epoch] = recon_loss_save_val / val_batches_processed

        loss_print[epoch] = loss_save/(num+1)
        recon_print[epoch] = recon_loss_save/(num+1)
        kl_print[epoch] = kl_loss_save/beta/(num+1)
        recon_loss_MSE_print[epoch] = recon_loss_MSE_save/(num+1)

        current_lr = optimizer.param_groups[0]['lr']

        loss_plot[epoch] = loss_print[epoch]

        scheduler.step()
        
        # Memory summary after first epoch
        if epoch == 0:
            print(f"\n{'='*80}")
            print(f"MEMORY SUMMARY AFTER FIRST EPOCH")
            print(f"{'='*80}")
            
            if torch.cuda.is_available():
                max_allocated = torch.cuda.max_memory_allocated() / 1024**2
                total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**2
                
                print(f"Peak memory usage: {max_allocated:.2f}MB")
                print(f"Total GPU memory: {total_memory:.2f}MB") 
                print(f"Memory utilization: {(max_allocated/total_memory)*100:.1f}%")
                
                if max_allocated > total_memory * 0.9:
                    print(f"⚠️  WARNING: Using >90% of GPU memory - at risk of OOM!")
                elif max_allocated > total_memory * 0.8:
                    print(f"⚠️  CAUTION: Using >80% of GPU memory")
                else:
                    print(f"✅ Memory usage looks safe")
                    
            print(f"{'='*80}\n")

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
    