import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import cv2
import os
import pandas as pd
import natsort
import time
import math
import psutil
import gc
import threading
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
from modules.pca_preprocessor import PCAPreprocessor
import matplotlib.pyplot as plt
# Removed mixed precision as requested

class CPUMonitor:
    """Real-time CPU monitoring for detecting spikes during training."""
    
    def __init__(self, spike_threshold=80.0, monitoring_interval=0.1):
        self.spike_threshold = spike_threshold
        self.monitoring_interval = monitoring_interval
        self.cpu_usage_history = []
        self.spike_events = []
        self.monitoring = False
        self.monitor_thread = None
        
    def start_monitoring(self):
        """Start background CPU monitoring."""
        self.monitoring = True
        self.monitor_thread = threading.Thread(target=self._monitor_cpu, daemon=True)
        self.monitor_thread.start()
        
    def stop_monitoring(self):
        """Stop background CPU monitoring."""
        self.monitoring = False
        if self.monitor_thread:
            self.monitor_thread.join(timeout=1.0)
            
    def _monitor_cpu(self):
        """Background thread for continuous CPU monitoring."""
        while self.monitoring:
            try:
                cpu_percent = psutil.cpu_percent(interval=self.monitoring_interval)
                timestamp = time.time()
                
                self.cpu_usage_history.append((timestamp, cpu_percent))
                
                # Detect CPU spikes
                if cpu_percent > self.spike_threshold:
                    self.spike_events.append({
                        'timestamp': timestamp,
                        'cpu_percent': cpu_percent,
                        'memory_percent': psutil.virtual_memory().percent,
                        'active_threads': threading.active_count()
                    })
                    
                # Keep only last 1000 measurements to prevent memory buildup
                if len(self.cpu_usage_history) > 1000:
                    self.cpu_usage_history = self.cpu_usage_history[-1000:]
                    
            except Exception as e:
                print(f"CPU monitoring error: {e}")
                break
                
    def get_current_cpu_usage(self):
        """Get instantaneous CPU usage."""
        return psutil.cpu_percent(interval=0.1)
        
    def get_cpu_spike_summary(self):
        """Get summary of CPU spikes detected."""
        if not self.spike_events:
            return "No CPU spikes detected"
            
        total_spikes = len(self.spike_events)
        max_spike = max(spike['cpu_percent'] for spike in self.spike_events)
        avg_spike = sum(spike['cpu_percent'] for spike in self.spike_events) / total_spikes
        
        return f"CPU Spikes: {total_spikes} events, Max: {max_spike:.1f}%, Avg: {avg_spike:.1f}%"
        
    def clear_history(self):
        """Clear monitoring history."""
        self.cpu_usage_history.clear()
        self.spike_events.clear()

def verify_tensor_devices(tensors_dict, expected_device):
    """Verify all tensors are on expected device and warn about mismatches."""
    device_issues = []
    
    for name, tensor in tensors_dict.items():
        if torch.is_tensor(tensor):
            if tensor.device != expected_device:
                device_issues.append(f"{name}: {tensor.device} (expected {expected_device})")
        elif isinstance(tensor, (list, tuple)):
            for i, t in enumerate(tensor):
                if torch.is_tensor(t) and t.device != expected_device:
                    device_issues.append(f"{name}[{i}]: {t.device} (expected {expected_device})")
    
    if device_issues:
        print(f"⚠️ DEVICE MISMATCH DETECTED:")
        for issue in device_issues:
            print(f"   • {issue}")
        return False
    return True

# Import shared utilities from original module
from modules.latent_conditioner import (
    DEFAULT_IMAGE_SIZE, INTERPOLATION_METHOD,
    read_latent_conditioner_dataset_img, read_latent_conditioner_dataset_img_pca,
    read_latent_conditioner_dataset, apply_outline_preserving_augmentations,
    safe_cuda_initialization, safe_initialize_weights_He, setup_device_and_model
)

def setup_optimizer_and_scheduler_e2e(latent_conditioner, latent_conditioner_lr, weight_decay, latent_conditioner_epoch):
    """Setup optimizer and scheduler for end-to-end training - same as original but with E2E suffix."""
    optimizer = torch.optim.AdamW(latent_conditioner.parameters(), lr=latent_conditioner_lr, weight_decay=weight_decay)
    warmup_epochs = 100
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, 
        start_factor=0.01,
        total_iters=warmup_epochs
    )
    
    main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, 
        T_max=latent_conditioner_epoch - warmup_epochs, 
        eta_min=1e-8
    )
    
    return optimizer, warmup_scheduler, main_scheduler, warmup_epochs

def load_vae_model(vae_model_path, device):
    """Load and prepare VAE model for end-to-end training."""
    try:
        # Try loading complete model first
        if os.path.exists(vae_model_path):
            vae_model = torch.load(vae_model_path, map_location=device, weights_only=False)
            print(f"Loaded VAE model from {vae_model_path}")
        else:
            # output error message
            print(f"VAE model not found at {vae_model_path}")
            raise FileNotFoundError(f"VAE model not found at {vae_model_path}")
        
        # Freeze VAE weights if it's a real model
        if hasattr(vae_model, 'parameters'):
            for param in vae_model.parameters():
                param.requires_grad = False
            vae_model.eval()
            print("VAE model weights frozen for end-to-end training")
        
        return vae_model
        
    except Exception as e:
        print(f"Error loading VAE model: {e}")

def train_latent_conditioner_e2e(latent_conditioner_epoch, e2e_dataloader, e2e_validation_dataloader, latent_conditioner, latent_conditioner_lr, weight_decay, is_image_data, image_size, config):
    """
    End-to-end training function with the same structure as the original train_latent_conditioner.
    
    Key difference: Loss is computed on reconstructed data instead of latent predictions.
    Architecture: Input Conditions → Latent Conditioner → VAE Decoder → Reconstructed Data
    """

    writer = SummaryWriter(log_dir = './LatentConditionerE2ERuns', comment = 'LatentConditioner_E2E')
    loss = 0
    latent_conditioner, device = setup_device_and_model(latent_conditioner)

    # Load VAE model for end-to-end training
    vae_model_path = config.get('e2e_vae_model_path', 'model_save/SimulGen-VAE') if config else 'model_save/SimulGen-VAE'
    vae_model = load_vae_model(vae_model_path, device)
    
    # Compile VAE decoder for 20-30% performance improvement
    vae_model.decoder = torch.compile(vae_model.decoder)
    print("✅ VAE decoder compiled for optimized performance")
    
    # Setup reconstruction loss function
    loss_function_type = config.get('e2e_loss_function', 'MSE') if config else 'MSE'
    if loss_function_type == 'MSE':
        reconstruction_loss_fn = nn.MSELoss()
    elif loss_function_type == 'MAE':
        reconstruction_loss_fn = nn.L1Loss()
    elif loss_function_type == 'Huber':
        reconstruction_loss_fn = nn.HuberLoss()
    elif loss_function_type == 'SmoothL1':
        reconstruction_loss_fn = nn.SmoothL1Loss()
    else:
        reconstruction_loss_fn = nn.MSELoss()
        print(f"Unknown loss function {loss_function_type}, using MSE")

    latent_conditioner_optimized, warmup_scheduler, main_scheduler, warmup_epochs = setup_optimizer_and_scheduler_e2e(
        latent_conditioner, latent_conditioner_lr, weight_decay, latent_conditioner_epoch
    )
    
    best_val_loss = float('inf')
    patience = 100000
    patience_counter = 0
    min_delta = 1e-8
    overfitting_threshold = 1000.0

    latent_conditioner = latent_conditioner.to(device)
    latent_conditioner.apply(safe_initialize_weights_He)
    model_summary_shown = False

    # Configuration for end-to-end training
    use_latent_regularization = config.get('use_latent_regularization', 0) == 1 if config else False
    latent_reg_weight = float(config.get('latent_reg_weight', 0.1)) if config else 0.1

    print(f"Starting end-to-end latent conditioner training for {latent_conditioner_epoch} epochs")
    print(f"Reconstruction loss function: {loss_function_type}")
    print(f"Latent regularization: {'Enabled' if use_latent_regularization else 'Disabled'}")
    
    # Display current VRAM usage at start of E2E training
    if torch.cuda.is_available():
        current_memory = torch.cuda.memory_allocated() / 1024**3
        reserved_memory = torch.cuda.memory_reserved() / 1024**3
        total_memory = torch.cuda.get_device_properties(0).total_memory / 1024**3
        free_memory = total_memory - reserved_memory
        print(f"=== VRAM Status at E2E Training Start ===")
        print(f"Used: {current_memory:.2f}GB | Reserved: {reserved_memory:.2f}GB | Free: {free_memory:.2f}GB | Total: {total_memory:.2f}GB")
        print(f"==========================================")

    for epoch in range(latent_conditioner_epoch):
        epoch_start_time = time.time()
        latent_conditioner.train(True)
        
        data_loader_start_time = time.time()
        # Use unified E2E dataloader
        for i, (x, y1, y2, target_data) in enumerate(e2e_dataloader):
            data_loader_end_time = time.time()
            
            dataloader_time = data_loader_end_time - data_loader_start_time
            
            if not model_summary_shown:
                batch_size = x.shape[0]
                input_features = x.shape[-1]
                img_size = int(math.sqrt(input_features))
                
                print(f"DEBUG: Input shape: {x.shape}, Input range: [{x.min():.4f}, {x.max():.4f}]")
                print(f"DEBUG: Target shape: {target_data.shape}")
                print(f"DEBUG: Image size: {img_size}x{img_size} ({input_features} pixels)")
                
                # Check sample image statistics
                sample_img = x[0].reshape(img_size, img_size)
                print(f"DEBUG: Sample image - mean: {sample_img.mean():.4f}, std: {sample_img.std():.4f}")
                print(f"DEBUG: Non-zero pixels: {(sample_img > 0.01).sum()}/{sample_img.numel()}")
                
                # Check target value ranges
                print(f"DEBUG: Target y1 shape: {y1.shape}, range: [{y1.min():.4f}, {y1.max():.4f}], mean: {y1.mean():.4f}")
                print(f"DEBUG: Target y2 shape: {y2.shape}, range: [{y2.min():.4f}, {y2.max():.4f}], mean: {y2.mean():.4f}")
                
                summary(latent_conditioner, (batch_size, 1, input_features))
                model_summary_shown = True
            
            # Data augmentation (same as original)
            if is_image_data and torch.rand(1, device=x.device) < 0.5:
                im_size = int(math.sqrt(x.shape[-1]))
                x_2d = x.reshape(-1, im_size, im_size)
                x_2d = apply_outline_preserving_augmentations(x_2d, prob=0.8)
                x = x_2d.reshape(x.shape[0], -1)
                
            # Mixup augmentation (applied to both input and target data)
            if torch.rand(1, device=x.device) < 0.02 and x.size(0) > 1:
                alpha = 0.2
                lam = torch.tensor(np.random.beta(alpha, alpha), device=x.device, dtype=x.dtype)
                batch_size = x.size(0)
                index = torch.randperm(batch_size, device=x.device)
                
                x = lam * x + (1 - lam) * x[index, :]
                target_data = lam * target_data + (1 - lam) * target_data[index, :]
                if use_latent_regularization:
                    y1 = lam * y1 + (1 - lam) * y1[index, :]
                    y2 = lam * y2 + (1 - lam) * y2[index, :]
            
            if torch.rand(1, device=x.device) < 0.2:
                noise = torch.randn_like(x) * 0.01
                x = x + noise
            
            # === STAGE 2: FORWARD PASS (DETAILED PROFILING) ===
            forward_start_time = time.time()
            latent_conditioner_optimized.zero_grad(set_to_none=True)

            try:
                # === SUBSTAGE 2A: LATENT CONDITIONER FORWARD ===
                lc_forward_start = time.time()
                y_pred1, y_pred2 = latent_conditioner(x)
                lc_forward_end = time.time()
                lc_forward_time = lc_forward_end - lc_forward_start

                # === SUBSTAGE 2B: TENSOR PREPROCESSING ===
                tensor_prep_start = time.time()
                # Keep original tensor for latent regularization
                y_pred2_tensor = y_pred2
                
                # The VAE decoder expects xs as a list where each element corresponds to a decoder layer
                # Based on the error, we need to restructure y_pred2 correctly
                if torch.is_tensor(y_pred2):
                    # If y_pred2 is [batch_size, 3, 8] then split along dim 1
                    if y_pred2.dim() == 3 and y_pred2.shape[1] == 3:
                        y_pred2 = [y_pred2[:, i, :] for i in range(y_pred2.shape[1])]
                    # If y_pred2 is [batch_size, num_layers * latent_dim], reshape and split
                    elif y_pred2.dim() == 2:
                        # Assume it needs to be reshaped to [batch_size, num_layers, latent_dim]
                        num_layers = 3  # Based on decoder structure
                        latent_dim = y_pred2.shape[1] // num_layers
                        y_pred2 = y_pred2.view(y_pred2.shape[0], num_layers, latent_dim)
                        y_pred2 = [y_pred2[:, i, :] for i in range(num_layers)]
                elif isinstance(y_pred2, (list, tuple)):
                    y_pred2 = list(y_pred2)
                tensor_prep_end = time.time()
                tensor_prep_time = tensor_prep_end - tensor_prep_start
                
                # === SUBSTAGE 2C: VAE DECODER (MAIN BOTTLENECK SUSPECT) ===
                vae_decoder_start = time.time()
                
                # NOTE: VAE parameters are frozen (requires_grad=False) but we need gradient flow for E2E training
                reconstructed_data, _ = vae_model.decoder(y_pred1, y_pred2)
                
                vae_decoder_end = time.time()
                vae_decoder_time = vae_decoder_end - vae_decoder_start
                
                # === SUBSTAGE 2D: LOSS COMPUTATION ===
                loss_comp_start = time.time()
                recon_loss = reconstruction_loss_fn(reconstructed_data, target_data)
                loss_comp_end = time.time()
                loss_comp_time = loss_comp_end - loss_comp_start
                
                # Optional latent regularization (same as original but weighted)
                if use_latent_regularization:
                    latent_reg_main = nn.MSELoss()(y_pred1, y1)
                    latent_reg_hier = nn.MSELoss()(y_pred2_tensor.reshape(-1), y2.reshape(-1))
                    latent_reg_total = latent_reg_main + latent_reg_hier
                    
                    # Combine reconstruction loss with latent regularization
                    loss = recon_loss + latent_reg_weight * latent_reg_total
                    epoch_latent_reg_loss += (latent_reg_weight * latent_reg_total).item()
                else:
                    loss = recon_loss

                forward_end_time = time.time()
                batch_forward_time = forward_end_time - forward_start_time
                total_forward_time += batch_forward_time
                
                # Accumulate detailed forward pass timings
                total_lc_forward_time += lc_forward_time
                total_tensor_prep_time += tensor_prep_time
                total_vae_decoder_time += vae_decoder_time
                total_loss_comp_time += loss_comp_time

                epoch_loss += loss.item()
                epoch_recon_loss += recon_loss.item()
                num_batches += 1
                
            except RuntimeError as e:
                print(f"Error during training at epoch {epoch+1}, batch {i+1}: {e}")
                raise
            
            # === STAGE 3: BACKWARD PASS ===
            backward_start_time = time.time()
            loss.backward()
            backward_end_time = time.time()
            batch_backward_time = backward_end_time - backward_start_time
            total_backward_time += batch_backward_time
            
            # === STAGE 4: OPTIMIZATION ===
            optimization_start_time = time.time()
            
            # Check gradient norms before clipping
            total_grad_norm = torch.nn.utils.clip_grad_norm_(latent_conditioner.parameters(), max_norm=10.0)
            latent_conditioner_optimized.step()
            
            
            optimization_end_time = time.time()
            # cpu_after_opt = cpu_monitor.get_current_cpu_usage()  # Disabled for performance
            batch_optimization_time = optimization_end_time - optimization_start_time
            total_optimization_time += batch_optimization_time
            
            if epoch % 100 == 0 and i == 0:  # Log every 100 epochs, first batch
                print(f"DEBUG: Gradient norm: {total_grad_norm:.4f}, Recon Loss: {recon_loss.item():.4E}, Total Loss: {loss.item():.4E}")
                if total_grad_norm > 10.0:
                    print(f"WARNING: Large gradient norm detected: {total_grad_norm:.2f}")
                elif total_grad_norm < 1e-4:
                    print(f"WARNING: Very small gradient norm: {total_grad_norm:.2E}")
        
        
        avg_train_loss = epoch_loss / num_batches if num_batches > 0 else 0.0
        avg_train_recon_loss = epoch_recon_loss / num_batches if num_batches > 0 else 0.0
        avg_train_latent_reg_loss = epoch_latent_reg_loss / num_batches if num_batches > 0 else 0.0

        latent_conditioner.eval()
        val_loss = 0
        val_recon_loss = 0
        val_latent_reg_loss = 0
        val_batches = 0
        
        # Run validation every epoch for complete loss tracking
        validation_start_time = time.time()
        with torch.no_grad():
            for i, (x_val, y1_val, y2_val, target_val_data) in enumerate(e2e_validation_dataloader):
                    
                    x_val, y1_val, y2_val = x_val.to(device), y1_val.to(device), y2_val.to(device)
                    target_val_data = target_val_data.to(device)
                    
                    y_pred1_val, y_pred2_val = latent_conditioner(x_val)
                    
                    # Keep original tensor for latent regularization
                    y_pred2_val_tensor = y_pred2_val
                    
                    # Convert y_pred2_val to proper list format for VAE decoder
                    if torch.is_tensor(y_pred2_val):
                        y_pred2_val = [y_pred2_val[:, i, :] for i in range(y_pred2_val.shape[1])]
                    elif isinstance(y_pred2_val, (list, tuple)):
                        y_pred2_val = list(y_pred2_val)
                    
                    # Validate with same end-to-end approach (gradients not needed for validation)
                    with torch.no_grad():
                        reconstructed_val_data, _ = vae_model.decoder(y_pred1_val, y_pred2_val)
                    recon_loss_val = reconstruction_loss_fn(reconstructed_val_data, target_val_data)
                    
                    if use_latent_regularization:
                        latent_reg_main_val = nn.MSELoss()(y_pred1_val, y1_val)
                        latent_reg_hier_val = nn.MSELoss()(y_pred2_val_tensor.reshape(-1), y2_val.reshape(-1))
                        latent_reg_total_val = latent_reg_main_val + latent_reg_hier_val
                        
                        total_val_loss = recon_loss_val + latent_reg_weight * latent_reg_total_val
                        val_latent_reg_loss += (latent_reg_weight * latent_reg_total_val).item()
                    else:
                        total_val_loss = recon_loss_val
                    
                    val_loss += total_val_loss.item()
                    val_recon_loss += recon_loss_val.item()
                    val_batches += 1

        # Process validation results (now runs every epoch)
        if val_batches > 0:
            avg_val_loss = val_loss / val_batches
            avg_val_recon_loss = val_recon_loss / val_batches
            avg_val_latent_reg_loss = val_latent_reg_loss / val_batches
            
            validation_end_time = time.time()
            validation_duration = validation_end_time - validation_start_time
            avg_val_batch_time = validation_duration / max(val_batches, 1)
        else:
            avg_val_loss = avg_val_recon_loss = avg_val_latent_reg_loss = 0.0
            validation_duration = 0.0
            avg_val_batch_time = 0.0
            
        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            patience_counter = 0
        else:
            patience_counter += 1
            
        if epoch < warmup_epochs:
            warmup_scheduler.step()
        else:
            main_scheduler.step()

        epoch_end_time = time.time()
        epoch_duration = epoch_end_time - epoch_start_time
        
        # === ELAPSED TIME SUMMARY FOR THIS EPOCH ===
        avg_batch_time = epoch_duration / max(num_batches, 1)
        avg_dataloader_time = dataloader_time
        avg_forward_time = total_forward_time / max(num_batches, 1)
        avg_backward_time = total_backward_time / max(num_batches, 1)
        avg_optimization_time = total_optimization_time / max(num_batches, 1)
        
        # Detailed forward pass averages
        avg_lc_forward_time = total_lc_forward_time / max(num_batches, 1)
        avg_tensor_prep_time = total_tensor_prep_time / max(num_batches, 1)
        avg_vae_decoder_time = total_vae_decoder_time / max(num_batches, 1)
        avg_loss_comp_time = total_loss_comp_time / max(num_batches, 1)
        
        # Calculate percentages
        dataloader_percent = (dataloader_time / epoch_duration) * 100 if epoch_duration > 0 else 0  # NEW
        forward_percent = (total_forward_time / epoch_duration) * 100 if epoch_duration > 0 else 0
        backward_percent = (total_backward_time / epoch_duration) * 100 if epoch_duration > 0 else 0
        optimization_percent = (total_optimization_time / epoch_duration) * 100 if epoch_duration > 0 else 0
        other_percent = 100 - (dataloader_percent + forward_percent + backward_percent + optimization_percent)
        
        # Detailed forward pass percentages (of total forward time)
        lc_forward_percent = (total_lc_forward_time / max(total_forward_time, 1e-8)) * 100
        tensor_prep_percent = (total_tensor_prep_time / max(total_forward_time, 1e-8)) * 100
        vae_decoder_percent = (total_vae_decoder_time / max(total_forward_time, 1e-8)) * 100
        loss_comp_percent = (total_loss_comp_time / max(total_forward_time, 1e-8)) * 100

        current_lr = latent_conditioner_optimized.param_groups[0]['lr']
        scheduler_info = f"Warmup" if epoch < warmup_epochs else f"Cosine"

        if epoch % 100 == 0:
            print(f'    ⏱️  TIMING BREAKDOWN - Epoch {epoch}:')
            print(f'        📡 DataLoader Fetch: {avg_dataloader_time*1000:.1f}ms/batch ({dataloader_percent:.1f}%) 🚨')
            print(f'        🔄 Forward Pass:     {avg_forward_time*1000:.1f}ms/batch ({forward_percent:.1f}%)')
            print(f'            🧠 Latent Cond:     {avg_lc_forward_time*1000:.1f}ms ({lc_forward_percent:.1f}% of forward)')
            print(f'            🔧 Tensor Prep:     {avg_tensor_prep_time*1000:.1f}ms ({tensor_prep_percent:.1f}% of forward)')
            print(f'            🏭 VAE Decoder:     {avg_vae_decoder_time*1000:.1f}ms ({vae_decoder_percent:.1f}% of forward) ⚠️')
            print(f'            📏 Loss Compute:    {avg_loss_comp_time*1000:.1f}ms ({loss_comp_percent:.1f}% of forward)')
            print(f'        ⬅️  Backward Pass:    {avg_backward_time*1000:.1f}ms/batch ({backward_percent:.1f}%)')
            print(f'        ⚡ Optimization:     {avg_optimization_time*1000:.1f}ms/batch ({optimization_percent:.1f}%)')
            print(f'        🔧 Other/Overhead:   {other_percent:.1f}%')
            
            print(f'        📊 Total Training:   {epoch_duration:.2f}s ({num_batches} batches, {avg_batch_time*1000:.1f}ms/batch avg)')
            if validation_duration > 0:
                print(f'        ✅ Validation:       {validation_duration:.2f}s ({val_batches} batches, {avg_val_batch_time*1000:.1f}ms/batch avg)')
        
        # Enhanced progress display with detailed timing breakdown
        print('[%d/%d]\tTrain: %.4E (recon:%.4E, reg:%.4E), Val: %.4E (recon:%.4E, reg:%.4E), LR: %.2E (%s), ETA: %.2f h, Patience: %d/%d' % 
              (epoch, latent_conditioner_epoch, avg_train_loss, avg_train_recon_loss, avg_train_latent_reg_loss, 
               avg_val_loss, avg_val_recon_loss, avg_val_latent_reg_loss,
               current_lr, scheduler_info,
               (latent_conditioner_epoch-epoch)*epoch_duration/3600, patience_counter, patience))
        
        data_loader_start_time = time.time()

    torch.save(latent_conditioner.state_dict(), 'checkpoints/latent_conditioner_e2e.pth')
    torch.save(latent_conditioner, 'model_save/LatentConditioner_E2E')
    
    return avg_val_loss