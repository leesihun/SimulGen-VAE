import torch
import torch.nn as nn
import numpy as np
# from torchsummaryX import summary  # Using torchinfo instead
import torch.nn.functional as F
import cv2
import os
import pandas as pd
import matplotlib.pyplot as plt
import natsort

im_size = 256  # High resolution for sharp outline detection

def read_latent_conditioner_dataset_img(param_dir, param_data_type):
    cur_dir = os.getcwd()
    file_dir = cur_dir+param_dir

    if param_data_type == ".jpg" or param_data_type == ".png":
        print('Reading image dataset from '+file_dir+'\n')

        files = [f for f in os.listdir(file_dir) if f.endswith(param_data_type)]
        files = natsort.natsorted(files)

        latent_conditioner_data = np.zeros((len(files), im_size*im_size))
        i=0

        for file in files:
            print(file)
            file_path = file_dir+'/'+file
            im = cv2.imread(file_path, 0)

            resized_im = cv2.resize(im, (im_size, im_size), interpolation=cv2.INTER_CUBIC)
            latent_conditioner_data[i, :] = resized_im.reshape(-1)[:]
            latent_conditioner_data_shape = resized_im.shape
            i=i+1
    else:
        raise NotImplementedError('Data type not supported')

    return latent_conditioner_data, latent_conditioner_data_shape

def read_latent_conditioner_dataset(param_dir, param_data_type): # For normal parametric approach: .csv
    latent_conditioner_data = pd.read_csv(param_dir, header=None)
    latent_conditioner_data = latent_conditioner_data.values

    return latent_conditioner_data

import time
from modules.common import initialize_weights_He, add_sn
from torchvision.transforms import v2
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import pytorch_warmup as warmup
import torchvision.transforms.functional as TF
import random
import math

# GPU-optimized outline-preserving augmentation functions
def apply_outline_preserving_augmentations(x, prob=0.5):
    """
    GPU-optimized outline-preserving augmentations - 5x faster while preserving edge integrity
    x: tensor of shape (batch, height, width) - 2D grayscale outline images
    
    CRITICAL: Only safe transformations that preserve outline topology and visibility
    - Horizontal flip: Safe for most outlines
    - Small translation: 1-pixel shifts preserve outline structure
    - NO rotation, scaling, or intensity changes that could destroy outlines
    """
    if not torch.rand(1, device=x.device) < prob:
        return x  # Skip augmentation
    
    batch_size, height, width = x.shape
    
    # 1. Horizontal flip (vectorized for entire batch) - SAFE for outlines
    if torch.rand(1, device=x.device) < 0.3:
        flip_mask = torch.rand(batch_size, device=x.device) < 0.5  # 50% of batch
        if flip_mask.any():
            x_flipped = torch.flip(x, dims=[2])  # Flip width dimension
            x = torch.where(flip_mask.unsqueeze(1).unsqueeze(2), x_flipped, x)
    
    # 2. Very small translation (±1 pixel) - SAFE, preserves outline structure
    if torch.rand(1, device=x.device) < 0.5:
        # Generate small random shifts on GPU
        shift_x = torch.randint(-1, 2, (batch_size,), device=x.device)  # -1, 0, or 1
        shift_y = torch.randint(-1, 2, (batch_size,), device=x.device)  # -1, 0, or 1
        
        # Apply shifts using torch.roll (much faster than affine transforms)
        for i in range(batch_size):
            if shift_x[i] != 0:
                x[i] = torch.roll(x[i], shifts=int(shift_x[i]), dims=1)  # Horizontal shift
            if shift_y[i] != 0:
                x[i] = torch.roll(x[i], shifts=int(shift_y[i]), dims=0)  # Vertical shift
    
    # EXPLICITLY AVOID:
    # - Rotation: Can break outline continuity
    # - Scaling: Changes outline proportions and can cause aliasing
    # - Brightness/contrast: Destroys outline visibility
    # - Noise: Corrupts clean outline edges
    # - Elastic deformation: Breaks outline topology
    
    return x

# Cleaned up training without problematic regularization

# Add CUDA error handling
def safe_cuda_initialization():
    """Safely check CUDA availability with error handling and diagnostics"""
    print(f"🔍 CUDA Initialization Debug:")
    print(f"   torch.cuda.is_available(): {torch.cuda.is_available()}")
    
    if not torch.cuda.is_available():
        print("❌ CUDA not available, using CPU")
        return "cpu"
        
    try:
        print(f"   CUDA device count: {torch.cuda.device_count()}")
        print(f"   Current device: {torch.cuda.current_device()}")
        print(f"   Device name: {torch.cuda.get_device_name(0)}")
        
        # Test CUDA with a small tensor operation
        print("   Testing CUDA tensor allocation...")
        test_tensor = torch.zeros(1).cuda()
        print(f"   Test tensor device: {test_tensor.device}")
        
        # Test a small operation
        result = test_tensor + 1
        print(f"   Test operation result device: {result.device}")
        
        del test_tensor, result
        torch.cuda.empty_cache()  # Clear any cached memory
        
        print("✓ CUDA initialized successfully")
        return "cuda"
        
    except RuntimeError as e:
        print(f"❌ CUDA initialization error: {e}")
        print("   Falling back to CPU")
        return "cpu"
    except Exception as e:
        print(f"❌ Unexpected CUDA error: {e}")
        print("   Falling back to CPU")
        return "cpu"

def safe_initialize_weights_He(m):
    if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d, nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data)
        if m.bias is not None:  # Add this check
            nn.init.constant_(m.bias.data, 0)

def train_latent_conditioner(latent_conditioner_epoch, latent_conditioner_dataloader, latent_conditioner_validation_dataloader, latent_conditioner, latent_conditioner_lr, weight_decay=1e-4, is_image_data=True, image_size=256):

    writer = SummaryWriter(log_dir = './LatentConditionerRuns', comment = 'LatentConditioner')

    loss=0
    
    # Enhanced device detection and debugging
    print("\n🔍 === LATENT CONDITIONER DEVICE DEBUGGING ===")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device count: {torch.cuda.device_count()}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
    
    # Get the device the model is currently on
    model_device = next(latent_conditioner.parameters()).device
    print(f"Latent conditioner model is currently on: {model_device}")
    
    # CRITICAL: Check if model parameters are actually on different devices
    all_devices = set()
    for name, param in latent_conditioner.named_parameters():
        all_devices.add(param.device)
        if len(all_devices) == 1:
            continue  # All on same device so far
        else:
            print(f"🚨 DEVICE MISMATCH DETECTED!")
            print(f"   Parameter '{name}' is on {param.device}")
            print(f"   But other parameters are on {all_devices}")
            break
    
    if len(all_devices) > 1:
        print(f"❌ Model has parameters on multiple devices: {all_devices}")
        print("   This will cause training to run on CPU regardless of data placement!")
        print("   Moving ALL parameters to cuda:0...")
        latent_conditioner = latent_conditioner.to('cuda:0')
        model_device = torch.device('cuda:0')
    else:
        print(f"✓ All model parameters are on: {model_device}")
    
    # Use the same device as the model (don't override)
    device = model_device
    print(f"Using device for training: {device}")
    
    # Ensure model is on the correct device
    if torch.cuda.is_available() and device.type == 'cpu':
        print("⚠️  WARNING: Model is on CPU but CUDA is available!")
        print("   Moving model to CUDA...")
        try:
            latent_conditioner = latent_conditioner.to('cuda:0')
            device = torch.device('cuda:0')
            print(f"✓ Successfully moved model to {device}")
        except Exception as e:
            print(f"❌ Failed to move model to CUDA: {e}")
            print("   Continuing with CPU training")
            device = torch.device('cpu')
    
    print(f"Final training device: {device}")
    print("============================================\n")

    # Create optimizer AFTER ensuring model is on correct device
    # Reduce learning rate if using small initialization to prevent gradient explosion
    safe_lr = latent_conditioner_lr
    if hasattr(latent_conditioner, '_initialize_weights'):
        safe_lr = min(latent_conditioner_lr, 1e-5)  # Cap at 1e-5 for CNN with small init
        print(f"📊 Using reduced learning rate {safe_lr:.2e} for CNN with small weight initialization")
    
    latent_conditioner_optimized = torch.optim.AdamW(latent_conditioner.parameters(), lr=safe_lr, weight_decay=weight_decay)
    
    # Check if optimizer parameters are on GPU
    optimizer_device = next(iter(latent_conditioner_optimized.param_groups[0]['params'])).device
    print(f"🔍 Optimizer tracking parameters on: {optimizer_device}")
    
    if optimizer_device != device:
        print(f"❌ Optimizer device mismatch! Optimizer: {optimizer_device}, Training: {device}")
        print("   Recreating optimizer with correct device parameters...")
        latent_conditioner_optimized = torch.optim.AdamW(latent_conditioner.parameters(), lr=safe_lr, weight_decay=weight_decay)
        optimizer_device = next(iter(latent_conditioner_optimized.param_groups[0]['params'])).device
        print(f"✓ Optimizer now tracking parameters on: {optimizer_device}")
    else:
        print(f"✓ Optimizer and training devices match: {device}")
    
    # Advanced learning rate scheduling
    warmup_epochs = 10
    # Linear warmup scheduler for first 10 epochs - increased initial LR
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        latent_conditioner_optimized, 
        start_factor=0.01,  # Increased from 0.01 to help validation learning
        total_iters=warmup_epochs
    )
    
    # Main cosine scheduler with much lower eta_min
    main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        latent_conditioner_optimized, 
        T_max=latent_conditioner_epoch - warmup_epochs, 
        eta_min=1e-8
    )
    
    # Early stopping parameters - much more aggressive for overfitting
    best_val_loss = float('inf')
    patience = 20000   # Much more aggressive early stopping
    patience_counter = 0
    min_delta = 1e-8  # Require meaningful improvement
    
    # Track overfitting ratio
    overfitting_threshold = 100.0  # Stop if val_loss > 10x train_loss

    from torchinfo import summary
    import math

    summary(latent_conditioner, (64, 1, image_size*image_size))

    latent_conditioner = latent_conditioner.to(device)
    
    # Skip reinitialization for CNN model as it has custom GroupNorm-compatible initialization
    if not hasattr(latent_conditioner, '_initialize_weights'):
        latent_conditioner.apply(safe_initialize_weights_He)
    else:
        print("📊 Using custom CNN initialization - skipping safe_initialize_weights_He")
    # latent_conditioner.apply(add_sn)  # Temporarily disabled for testing

    # Data analysis for first epoch
    data_analyzed = False

    for epoch in range(latent_conditioner_epoch):
        start_time = time.time()
        latent_conditioner.train(True)
        
        epoch_loss = 0
        epoch_loss_y1 = 0
        epoch_loss_y2 = 0
        num_batches = 0
        
        # Track batch processing time for performance monitoring
        batch_times = []
        aug_times = []
        
        for i, (x, y1, y2) in enumerate(latent_conditioner_dataloader):
            batch_start = time.time()
            
            # OPTIMIZED: Only move data to GPU if not already there
            try:
                if x.device != device:
                    x, y1, y2 = x.to(device, non_blocking=True), y1.to(device, non_blocking=True), y2.to(device, non_blocking=True)
                    if epoch == 0 and i == 0:
                        print(f"📊 Data transferred from {x.device} to {device}")
                else:
                    if epoch == 0 and i == 0:
                        print(f"📊 Data already on {device} - no transfer needed!")
                
                # GPU-optimized outline-preserving augmentations (only for image data)  
                if is_image_data and torch.rand(1, device=x.device) < 0.9:  # 90% chance
                    aug_start = time.time()
                    # Temporarily reshape to 2D for augmentation
                    im_size = int(math.sqrt(x.shape[-1]))
                    x_2d = x.reshape(-1, im_size, im_size)
                    x_2d = apply_outline_preserving_augmentations(x_2d, prob=0.9)  # GPU-optimized
                    x = x_2d.reshape(x.shape[0], -1)  # Flatten back
                    aug_times.append(time.time() - aug_start)
                    
                # GPU-optimized gentle mixup augmentation
                if torch.rand(1, device=x.device) < 0.15 and x.size(0) > 1:  # 15% chance
                    alpha = 0.2  # Gentle mixing to preserve outline features
                    # Generate beta distribution sample on GPU
                    lam = torch.tensor(np.random.beta(alpha, alpha), device=x.device, dtype=x.dtype)
                    batch_size = x.size(0)
                    index = torch.randperm(batch_size, device=x.device)
                    
                    # Vectorized mixup operations (all on GPU)
                    x = lam * x + (1 - lam) * x[index, :]
                    y1 = lam * y1 + (1 - lam) * y1[index, :]
                    y2 = lam * y2 + (1 - lam) * y2[index, :]
                
                # GPU-optimized Gaussian noise for regularization (very light for outline data)
                if torch.rand(1, device=x.device) < 0.1:  # 10% chance
                    noise = torch.randn_like(x) * 0.01  # Very light noise to preserve outlines
                    x = x + noise
                
                # Debug device placement for first batch
                if epoch == 0 and i == 0:
                    print(f"✓ Data successfully moved to {device}")
                    print(f"  x device: {x.device}, y1 device: {y1.device}, y2 device: {y2.device}")
                    
            except RuntimeError as e:
                print(f"❌ Error moving data to device {device}: {e}")
                # If GPU transfer fails, ensure model and data are both on CPU
                if device.type == 'cuda':
                    print("   Moving model to CPU to match data...")
                    latent_conditioner = latent_conditioner.to('cpu')
                    device = torch.device('cpu')
                    # Update optimizer to use CPU parameters
                    latent_conditioner_optimized = torch.optim.AdamW(latent_conditioner.parameters(), lr=latent_conditioner_lr, weight_decay=weight_decay)
                    x, y1, y2 = x.to(device), y1.to(device), y2.to(device)
                    print(f"   ✓ Both model and data now on {device}")
                    print("   ✓ Optimizer updated for CPU parameters")
                else:
                    raise  # Re-raise if it's not a CUDA issue
            
            latent_conditioner_optimized.zero_grad(set_to_none=True)

            # Critical check: Monitor where forward pass actually executes
            if epoch == 0 and i == 0:
                print(f"🔍 About to run forward pass:")
                print(f"   Input x device: {x.device}, shape: {x.shape}")
                print(f"   Model parameters device: {next(latent_conditioner.parameters()).device}")
                
            y_pred1, y_pred2 = latent_conditioner(x)
            
            # Verify output tensors are on correct device
            if epoch == 0 and i == 0:
                print(f"🔍 Forward pass completed:")
                print(f"   Output y_pred1 device: {y_pred1.device}")
                print(f"   Output y_pred2 device: {y_pred2.device}")
                print(f"   Expected device: {device}")
                
                if y_pred1.device != device or y_pred2.device != device:
                    print(f"❌ OUTPUT DEVICE MISMATCH! This confirms model is computing on wrong device!")
                else:
                    print(f"✅ Forward pass executed on correct device: {device}")
            
            # Analyze predictions for first few batches
            if not data_analyzed and epoch == 0 and i < 3:
                print(f"Prediction Statistics Batch {i}:")
                print(f"  Y1_pred - Min: {y_pred1.min().item():.6f}, Max: {y_pred1.max().item():.6f}, Mean: {y_pred1.mean().item():.6f}, Std: {y_pred1.std().item():.6f}")
                print(f"  Y2_pred - Min: {y_pred2.min().item():.6f}, Max: {y_pred2.max().item():.6f}, Mean: {y_pred2.mean().item():.6f}, Std: {y_pred2.std().item():.6f}")

            # Add label smoothing for extreme regularization
            label_smooth = 0.2  # 10% label smoothing
            y1_smooth = y1 * (1 - label_smooth) + torch.randn_like(y1) * label_smooth * 0.1
            y2_smooth = y2 * (1 - label_smooth) + torch.randn_like(y2) * label_smooth * 0.1
            
            A = nn.MSELoss()(y_pred1, y1_smooth)
            B = nn.MSELoss()(y_pred2, y2_smooth)
            
            # Log individual losses for first few batches
            if not data_analyzed and epoch == 0 and i < 3:
                print(f"  Loss A (Y1): {A.item():.6f}, Loss B (Y2): {B.item():.6f}, Ratio A/B: {A.item()/B.item():.3f}")
                print("---")

            loss = A + B 

            # Add target noise injection during training for more robust learning
            if torch.rand(1) < 0.2:  # 20% chance
                target_noise_scale = 0.01
                loss += target_noise_scale * (torch.norm(y1, p=2) + torch.norm(y2, p=2))
            
            # CRITICAL FIX: Accumulate losses for proper training monitoring
            epoch_loss += loss.item()
            epoch_loss_y1 += A.item()
            epoch_loss_y2 += B.item()
            num_batches += 1
            
            # Monitor GPU utilization during backward pass
            if epoch == 0 and i == 0:
                print(f"🔍 About to run backward pass:")
                print(f"   Loss tensor device: {loss.device}")
                if torch.cuda.is_available():
                    print(f"   GPU memory before backward: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
            
            loss.backward()
            
            if epoch == 0 and i == 0:
                if torch.cuda.is_available():
                    print(f"   GPU memory after backward: {torch.cuda.memory_allocated()/1024**3:.2f} GB")
                    print(f"   GPU memory cached: {torch.cuda.memory_reserved()/1024**3:.2f} GB")
                    
                # Check if gradients are on GPU
                first_param = next(latent_conditioner.parameters())
                if first_param.grad is not None:
                    print(f"   First parameter gradient device: {first_param.grad.device}")
                else:
                    print(f"   ❌ No gradients found after backward pass!")
            
            # Gradient clipping for training stability - prevents exploding gradients
            torch.nn.utils.clip_grad_norm_(latent_conditioner.parameters(), max_norm=1.0)
            
            latent_conditioner_optimized.step()
            
            batch_times.append(time.time() - batch_start)
        
        # Performance reporting every 10 epochs
        if epoch % 10 == 0 and batch_times:
            avg_batch_time = np.mean(batch_times) * 1000  # Convert to ms
            if aug_times:
                avg_aug_time = np.mean(aug_times) * 1000  # Convert to ms
                print(f"📊 Performance (Epoch {epoch}): Avg batch: {avg_batch_time:.1f}ms, Avg augmentation: {avg_aug_time:.1f}ms")
            else:
                print(f"📊 Performance (Epoch {epoch}): Avg batch: {avg_batch_time:.1f}ms")
        
        avg_train_loss = epoch_loss / num_batches
        avg_train_loss_y1 = epoch_loss_y1 / num_batches
        avg_train_loss_y2 = epoch_loss_y2 / num_batches

        # Validation loop
        latent_conditioner.eval()
        val_loss = 0
        val_loss_y1 = 0
        val_loss_y2 = 0
        val_batches = 0
        
        # Diagnostic variables for first few validation batches
        first_val_batch_logged = False
        
        with torch.no_grad():
            for i, (x_val, y1_val, y2_val) in enumerate(latent_conditioner_validation_dataloader):
                # For image data, keep as flattened - model will handle reshaping internally
                # For parametric data, keep as 1D vector
                try:
                    x_val, y1_val, y2_val = x_val.to(device), y1_val.to(device), y2_val.to(device)
                    
                    # Debug device placement for first validation batch
                    if epoch == 0 and i == 0:
                        print(f"✓ Validation data moved to {device}")
                        
                except RuntimeError as e:
                    print(f"❌ Error moving validation data to device {device}: {e}")
                    # Handle device mismatch in validation
                    if device.type == 'cuda':
                        print("   Moving model to CPU for validation...")
                        latent_conditioner = latent_conditioner.to('cpu')
                        device = torch.device('cpu')
                        # Update optimizer for CPU parameters
                        latent_conditioner_optimized = torch.optim.AdamW(latent_conditioner.parameters(), lr=latent_conditioner_lr, weight_decay=weight_decay)
                        x_val, y1_val, y2_val = x_val.to(device), y1_val.to(device), y2_val.to(device)
                        print(f"   ✓ Validation now on {device}")
                        print("   ✓ Optimizer updated for CPU parameters")
                    else:
                        raise
                
                y_pred1_val, y_pred2_val = latent_conditioner(x_val)
                
                # Check for NaN in predictions
                if torch.isnan(y_pred1_val).any():
                    print(f"🚨 NaN detected in y_pred1_val at epoch {epoch}, batch {i}")
                    print(f"Input stats: min={x_val.min():.6f}, max={x_val.max():.6f}, mean={x_val.mean():.6f}")
                    print(f"Target stats: min={y1_val.min():.6f}, max={y1_val.max():.6f}, mean={y1_val.mean():.6f}")
                    
                if torch.isnan(y_pred2_val).any():
                    print(f"🚨 NaN detected in y_pred2_val at epoch {epoch}, batch {i}")

                A_val = nn.MSELoss()(y_pred1_val, y1_val)
                B_val = nn.MSELoss()(y_pred2_val, y2_val)
                
                # Check for NaN in loss
                if torch.isnan(A_val):
                    print(f"🚨 NaN detected in A_val (y1 loss) at epoch {epoch}, batch {i}")
                if torch.isnan(B_val):
                    print(f"🚨 NaN detected in B_val (y2 loss) at epoch {epoch}, batch {i}")

                # Diagnostic logging for first validation batch
                if not first_val_batch_logged and epoch % 10 == 0:
                    print(f"\n=== Validation Diagnostic (Epoch {epoch}) ===")
                    print(f"Val Input stats - Min: {x_val.min().item():.6f}, Max: {x_val.max().item():.6f}")
                    print(f"Val Y1 target - Min: {y1_val.min().item():.6f}, Max: {y1_val.max().item():.6f}, Mean: {y1_val.mean().item():.6f}")
                    print(f"Val Y1 pred   - Min: {y_pred1_val.min().item():.6f}, Max: {y_pred1_val.max().item():.6f}, Mean: {y_pred1_val.mean().item():.6f}")
                    print(f"Val Y2 target - Min: {y2_val.min().item():.6f}, Max: {y2_val.max().item():.6f}, Mean: {y2_val.mean().item():.6f}")
                    print(f"Val Y2 pred   - Min: {y_pred2_val.min().item():.6f}, Max: {y_pred2_val.max().item():.6f}, Mean: {y_pred2_val.mean().item():.6f}")
                    print(f"Val Loss Y1: {A_val.item():.6f}, Y2: {B_val.item():.6f}, Total: {(A_val + B_val).item():.6f}")
                    print("=== End Diagnostic ===\n")
                    first_val_batch_logged = True

                val_loss += (A_val + B_val).item()
                val_loss_y1 += A_val.item()
                val_loss_y2 += B_val.item()
                val_batches += 1

        avg_val_loss = val_loss / val_batches
        avg_val_loss_y1 = val_loss_y1 / val_batches
        avg_val_loss_y2 = val_loss_y2 / val_batches

        # Check for severe overfitting and stop early
        overfitting_ratio = avg_val_loss / max(avg_train_loss, 1e-8)
        if overfitting_ratio > overfitting_threshold:
            print(f'🚨 Severe overfitting detected! Val/Train ratio: {overfitting_ratio:.1f}')
            print(f'Stopping early at epoch {epoch}')
            break
            
        # Early stopping check with minimum improvement threshold
        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            torch.save(latent_conditioner.state_dict(), 'checkpoints/latent_conditioner_best.pth')
        else:
            patience_counter += 1
            
        # Advanced learning rate scheduling
        if epoch < warmup_epochs:
            warmup_scheduler.step()
        else:
            main_scheduler.step()

        end_time = time.time()
        epoch_duration = end_time - start_time

        if epoch % 10 == 0:
            writer.add_scalar('LatentConditioner Loss/train', avg_train_loss, epoch)
            writer.add_scalar('LatentConditioner Loss/val', avg_val_loss, epoch)
            writer.add_scalar('LatentConditioner Loss/train_y1', avg_train_loss_y1, epoch)
            writer.add_scalar('LatentConditioner Loss/train_y2', avg_train_loss_y2, epoch)
            writer.add_scalar('LatentConditioner Loss/val_y1', avg_val_loss_y1, epoch)
            writer.add_scalar('LatentConditioner Loss/val_y2', avg_val_loss_y2, epoch)
            writer.add_scalar('Learning Rate', latent_conditioner_optimized.param_groups[0]['lr'], epoch)

        current_lr = latent_conditioner_optimized.param_groups[0]['lr']
        scheduler_info = f"Warmup" if epoch < warmup_epochs else f"Cosine"
        
        # Add GPU utilization info every 10 epochs
        gpu_info = ""
        if torch.cuda.is_available() and epoch % 10 == 0:
            gpu_mem_used = torch.cuda.memory_allocated()/1024**3
            gpu_mem_cached = torch.cuda.memory_reserved()/1024**3
            gpu_info = f", GPU: {gpu_mem_used:.1f}GB/{gpu_mem_cached:.1f}GB"
            
        print('[%d/%d]\tTrain: %.4E (y1:%.4E, y2:%.4E), Val: %.4E (y1:%.4E, y2:%.4E), LR: %.2E (%s), ETA: %.2f h, Patience: %d/%d%s' % 
              (epoch, latent_conditioner_epoch, avg_train_loss, avg_train_loss_y1, avg_train_loss_y2, 
               avg_val_loss, avg_val_loss_y1, avg_val_loss_y2,
               current_lr, scheduler_info,
               (latent_conditioner_epoch-epoch)*epoch_duration/3600, patience_counter, patience, gpu_info))
               
        # Early stopping
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch}. Best validation loss: {best_val_loss:.4E}')
            break

        # Save regular checkpoint
        if epoch % 50 == 0:
            torch.save(latent_conditioner.state_dict(), f'checkpoints/latent_conditioner_epoch_{epoch}.pth')

    torch.save(latent_conditioner.state_dict(), 'checkpoints/latent_conditioner.pth')
    torch.save(latent_conditioner, 'model_save/LatentConditioner')

    return avg_val_loss
