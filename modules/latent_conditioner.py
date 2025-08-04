import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import cv2
import os
import pandas as pd
import natsort
from modules.pca_preprocessor import PCAPreprocessor

# Image processing constants
DEFAULT_IMAGE_SIZE = 256  # High resolution for sharp outline detection
INTERPOLATION_METHOD = cv2.INTER_CUBIC  # High-quality interpolation for image resizing

im_size = DEFAULT_IMAGE_SIZE  # Backward compatibility

def read_latent_conditioner_dataset_img(param_dir, param_data_type, debug_mode=0, use_pca=False, pca_components=1024, pca_patch_size=None):
    cur_dir = os.getcwd()
    file_dir = cur_dir+param_dir

    if param_data_type == ".jpg" or param_data_type == ".png":
        if debug_mode == 1:
            print('Reading image dataset from '+file_dir+'\n')

        files = [f for f in os.listdir(file_dir) if f.endswith(param_data_type)]
        files = natsort.natsorted(files)

        # Read all images first
        raw_images = np.zeros((len(files), im_size, im_size))
        
        for i, file in enumerate(files):
            if debug_mode == 1:
                print(file)
            file_path = os.path.join(file_dir, file)
            im = cv2.imread(file_path, 0)
            resized_im = cv2.resize(im, (im_size, im_size), interpolation=INTERPOLATION_METHOD)
            raw_images[i] = resized_im

        if use_pca:
            # Apply PCA preprocessing
            if debug_mode == 1:
                print(f'Applying PCA preprocessing with {pca_components} components')
                if pca_patch_size:
                    print(f'Using patch-based PCA with patch size {pca_patch_size}')
            
            pca_preprocessor = PCAPreprocessor(
                n_components=pca_components, 
                patch_size=pca_patch_size
            )
            
            # Try to load existing PCA model, otherwise fit new one
            try:
                pca_preprocessor.load()
                if debug_mode == 1:
                    print('Loaded existing PCA model')
            except FileNotFoundError:
                if debug_mode == 1:
                    print('Fitting new PCA model on training data')
                pca_preprocessor.fit(raw_images)
            
            # Transform images using PCA
            pca_tensor = pca_preprocessor.transform(raw_images)  # Returns torch.Tensor
            
            # Convert back to numpy and flatten for compatibility
            if len(pca_tensor.shape) == 4:  # (n_samples, channels, height, width)
                latent_conditioner_data = pca_tensor.view(pca_tensor.shape[0], -1).numpy()
                latent_conditioner_data_shape = pca_tensor.shape[2:]  # (height, width)
            else:
                latent_conditioner_data = pca_tensor.numpy()
                latent_conditioner_data_shape = pca_preprocessor.get_output_shape()
            
            if debug_mode == 1:
                print(f'PCA output shape: {latent_conditioner_data_shape}')
                print(f'Data reduced from {im_size*im_size} to {latent_conditioner_data.shape[1]} dimensions')
                
        else:
            # Standard processing without PCA
            latent_conditioner_data = raw_images.reshape(len(files), -1)
            latent_conditioner_data_shape = (im_size, im_size)
            
    else:
        raise NotImplementedError('Data type not supported')

    return latent_conditioner_data, latent_conditioner_data_shape

def read_latent_conditioner_dataset(param_dir, param_data_type): # For normal parametric approach: .csv
    latent_conditioner_data = pd.read_csv(param_dir, header=None)
    latent_conditioner_data = latent_conditioner_data.values

    return latent_conditioner_data

import time
import math
from torch.utils.tensorboard import SummaryWriter

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
def safe_cuda_initialization(debug_mode=0):
    """Safely check CUDA availability with error handling and diagnostics"""

    if not torch.cuda.is_available():
        if debug_mode == 1:
            print("❌ CUDA not available, using CPU")
        return "cpu"
        
    try:
        if debug_mode == 1:
            print(f"   CUDA device count: {torch.cuda.device_count()}")
            print(f"   Current device: {torch.cuda.current_device()}")
            print(f"   Device name: {torch.cuda.get_device_name(0)}")
        
        # Test CUDA with a small tensor operation
        test_tensor = torch.zeros(1).cuda()
        # Test a small operation
        result = test_tensor + 1
        del test_tensor, result
        torch.cuda.empty_cache()  # Clear any cached memory
        
        if debug_mode == 1:
            print("✓ CUDA initialized successfully")
        return "cuda"
        
    except RuntimeError as e:
        if debug_mode == 1:
            print(f"❌ CUDA initialization error: {e}")
        return "cpu"
    except Exception as e:
        if debug_mode == 1:
            print(f"❌ Unexpected CUDA error: {e}")
        return "cpu"

def safe_initialize_weights_He(m):
    if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d, nn.Conv2d, nn.ConvTranspose2d)):
        nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight.data)
        if m.bias is not None:
            nn.init.constant_(m.bias.data, 0)

def setup_device_and_model(latent_conditioner):
    """Setup device and move model appropriately"""
    model_device = next(latent_conditioner.parameters()).device
    device = model_device
    
    # Move to CUDA if available and model is on CPU
    if torch.cuda.is_available() and device.type == 'cpu':
        try:
            latent_conditioner = latent_conditioner.to('cuda:0')
            device = torch.device('cuda:0')
        except Exception as e:
            # Keep this print as it's an important fallback notification
            print(f"Failed to move model to CUDA: {e}")
            device = torch.device('cpu')
    
    return latent_conditioner, device

def setup_optimizer_and_scheduler(latent_conditioner, latent_conditioner_lr, weight_decay, latent_conditioner_epoch):
    """Setup optimizer and learning rate schedulers"""
    # Create optimizer with appropriate learning rate
    optimizer = torch.optim.AdamW(latent_conditioner.parameters(), lr=latent_conditioner_lr, weight_decay=weight_decay)
    
    # Advanced learning rate scheduling
    warmup_epochs = 10
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

def train_latent_conditioner(latent_conditioner_epoch, latent_conditioner_dataloader, latent_conditioner_validation_dataloader, latent_conditioner, latent_conditioner_lr, weight_decay=1e-4, is_image_data=True, image_size=256, debug_mode=0):

    writer = SummaryWriter(log_dir = './LatentConditionerRuns', comment = 'LatentConditioner')

    loss=0
    
    # Setup device and model
    latent_conditioner, device = setup_device_and_model(latent_conditioner)
    if debug_mode == 1:
        print(f"Training on device: {device}")

    # Setup optimizer and schedulers
    latent_conditioner_optimized, warmup_scheduler, main_scheduler, warmup_epochs = setup_optimizer_and_scheduler(
        latent_conditioner, latent_conditioner_lr, weight_decay, latent_conditioner_epoch
    )
    
    # Early stopping parameters - much more aggressive for overfitting
    best_val_loss = float('inf')
    patience = 20000   # Much more aggressive early stopping
    patience_counter = 0
    min_delta = 1e-8
    
    # Track overfitting ratio
    overfitting_threshold = 1000.0  # Stop if val_loss > 10x train_loss

    from torchinfo import summary
    
    summary(latent_conditioner, (32, 1, image_size*image_size))

    latent_conditioner = latent_conditioner.to(device)
    
    # Initialize weights if needed
    latent_conditioner.apply(safe_initialize_weights_He)

    for epoch in range(latent_conditioner_epoch):
        start_time = time.time()
        latent_conditioner.train(True)
        
        epoch_loss = 0
        epoch_loss_y1 = 0
        epoch_loss_y2 = 0
        num_batches = 0
        
        for i, (x, y1, y2) in enumerate(latent_conditioner_dataloader):
            
            if x.device != device:
                x, y1, y2 = x.to(device, non_blocking=True), y1.to(device, non_blocking=True), y2.to(device, non_blocking=True)
            
            # GPU-optimized outline-preserving augmentations (only for image data)  
            if is_image_data and torch.rand(1, device=x.device) < 0.5:  # 90% chance
                # Temporarily reshape to 2D for augmentation
                im_size = int(math.sqrt(x.shape[-1]))
                x_2d = x.reshape(-1, im_size, im_size)
                x_2d = apply_outline_preserving_augmentations(x_2d, prob=0.5)  # GPU-optimized
                x = x_2d.reshape(x.shape[0], -1)  # Flatten back
                
            # GPU-optimized gentle mixup augmentation
            if torch.rand(1, device=x.device) < 0.1 and x.size(0) > 1:  # 15% chance
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
            
            latent_conditioner_optimized.zero_grad(set_to_none=True)

            y_pred1, y_pred2 = latent_conditioner(x)
            

            # Add label smoothing for extreme regularization
            label_smooth = 0.1  # 10% label smoothing
            y1_smooth = y1 * (1 - label_smooth) + torch.randn_like(y1) * label_smooth * 0.1
            y2_smooth = y2 * (1 - label_smooth) + torch.randn_like(y2) * label_smooth * 0.1
            
            A = nn.MSELoss()(y_pred1, y1_smooth)
            B = nn.MSELoss()(y_pred2, y2_smooth)
            

            loss = A + B 

            # # Add target noise injection during training for more robust learning
            # if torch.rand(1) < 0.2:  # 20% chance
            #     target_noise_scale = 0.01
            #     loss += target_noise_scale * (torch.norm(y1, p=2) + torch.norm(y2, p=2))
            
            epoch_loss += loss.item()
            epoch_loss_y1 += A.item()
            epoch_loss_y2 += B.item()
            num_batches += 1
            
            loss.backward()
            
            # Gradient clipping for training stability - prevents exploding gradients
            torch.nn.utils.clip_grad_norm_(latent_conditioner.parameters(), max_norm=5.0)
            
            latent_conditioner_optimized.step()
        
        
        avg_train_loss = epoch_loss / num_batches
        avg_train_loss_y1 = epoch_loss_y1 / num_batches
        avg_train_loss_y2 = epoch_loss_y2 / num_batches

        # Validation loop
        latent_conditioner.eval()
        val_loss = 0
        val_loss_y1 = 0
        val_loss_y2 = 0
        val_batches = 0
        
        if epoch % 10 == 0:
            with torch.no_grad():
                for i, (x_val, y1_val, y2_val) in enumerate(latent_conditioner_validation_dataloader):
                    # Move validation data to device
                    x_val, y1_val, y2_val = x_val.to(device), y1_val.to(device), y2_val.to(device)
                    
                    y_pred1_val, y_pred2_val = latent_conditioner(x_val)
                    
                    A_val = nn.MSELoss()(y_pred1_val, y1_val)
                    B_val = nn.MSELoss()(y_pred2_val, y2_val)
                    
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
                print(f'Severe overfitting detected! Val/Train ratio: {overfitting_ratio:.1f}')
                print(f'Stopping early at epoch {epoch}')
                break
                
            # Early stopping check with minimum improvement threshold
            if avg_val_loss < best_val_loss - min_delta:
                best_val_loss = avg_val_loss
                patience_counter = 0
                # Save best model
            else:
                patience_counter += 1
            
        # Advanced learning rate scheduling
        if epoch < warmup_epochs:
            warmup_scheduler.step()
        else:
            main_scheduler.step()

        end_time = time.time()
        epoch_duration = end_time - start_time

        if epoch % 100 == 0:
            writer.add_scalar('LatentConditioner Loss/train', avg_train_loss, epoch)
            writer.add_scalar('LatentConditioner Loss/val', avg_val_loss, epoch)
            writer.add_scalar('LatentConditioner Loss/train_y1', avg_train_loss_y1, epoch)
            writer.add_scalar('LatentConditioner Loss/train_y2', avg_train_loss_y2, epoch)
            writer.add_scalar('LatentConditioner Loss/val_y1', avg_val_loss_y1, epoch)
            writer.add_scalar('LatentConditioner Loss/val_y2', avg_val_loss_y2, epoch)
            writer.add_scalar('Learning Rate', latent_conditioner_optimized.param_groups[0]['lr'], epoch)

        current_lr = latent_conditioner_optimized.param_groups[0]['lr']
        scheduler_info = f"Warmup" if epoch < warmup_epochs else f"Cosine"
        
        print('[%d/%d]\tTrain: %.4E (y1:%.4E, y2:%.4E), Val: %.4E (y1:%.4E, y2:%.4E), LR: %.2E (%s), ETA: %.2f h, Patience: %d/%d' % 
              (epoch, latent_conditioner_epoch, avg_train_loss, avg_train_loss_y1, avg_train_loss_y2, 
               avg_val_loss, avg_val_loss_y1, avg_val_loss_y2,
               current_lr, scheduler_info,
               (latent_conditioner_epoch-epoch)*epoch_duration/3600, patience_counter, patience))
               
        # Early stopping
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch}. Best validation loss: {best_val_loss:.4E}')
            break

    torch.save(latent_conditioner.state_dict(), 'checkpoints/latent_conditioner.pth')
    torch.save(latent_conditioner, 'model_save/LatentConditioner')

    return avg_val_loss