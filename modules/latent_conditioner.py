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

# Outline-specific augmentation functions - preserves edge information
def apply_outline_preserving_augmentations(x, prob=0.5):
    """
    Apply outline-preserving augmentations for edge detection tasks
    x: tensor of shape (batch, height, width) - 2D grayscale outline images
    
    CRITICAL: Avoids intensity changes that could destroy outline visibility
    """
    if not torch.rand(1) < prob:
        return x  # Skip augmentation
    
    batch_size, height, width = x.shape
    augmented = x.clone()
    
    for i in range(batch_size):
        img = x[i].unsqueeze(0)  # Add channel dim: (1, H, W)
        
        # Small rotation ONLY (-5 to +5 degrees) - preserve outline topology
        if torch.rand(1) < 0.4:
            angle = random.uniform(-5, 5)
            img = TF.rotate(img, angle, fill=0)
        
        # Small translation (±1 pixel) - very conservative to preserve outlines
        if torch.rand(1) < 0.5:
            translate_x = random.randint(-1, 1)
            translate_y = random.randint(-1, 1)
            img = TF.affine(img, angle=0, translate=[translate_x, translate_y], 
                           scale=1.0, shear=0, fill=0)
        
        # Random horizontal flip - only if outline symmetry allows
        if torch.rand(1) < 0.3:
            img = TF.hflip(img)
        
        # Very subtle scale changes (±5%) - preserve outline proportions
        if torch.rand(1) < 0.3:
            scale = random.uniform(0.95, 1.05)
            img = TF.affine(img, angle=0, translate=[0, 0], 
                           scale=scale, shear=0, fill=0)
        
        # AVOID: brightness, contrast, gamma, noise - these destroy outline clarity!
        
        augmented[i] = img.squeeze(0)  # Remove channel dim
    
    return augmented

# Cleaned up training without problematic regularization

# Add CUDA error handling
def safe_cuda_initialization():
    """Safely check CUDA availability with error handling and diagnostics"""
    try:
        if torch.cuda.is_available():
            # Test CUDA with a small tensor operation
            test_tensor = torch.zeros(1).cuda()
            del test_tensor
            print("✓ CUDA initialized successfully")
            return "cuda"
        else:
            print("CUDA not available, using CPU")
            return "cpu"
    except RuntimeError as e:
        print(f"⚠️ CUDA initialization error: {e}")
        print("Falling back to CPU. To enable device side assertions, recompile PyTorch with torch_USA_CUDA_DSA=1")
        # Get CUDA diagnostic information
        try:
            if torch.cuda.is_available():
                print(f"CUDA device count: {torch.cuda.device_count()}")
                print(f"Current device: {torch.cuda.current_device()}")
                print(f"Device name: {torch.cuda.get_device_name(0)}")
        except:
            print("Could not retrieve CUDA diagnostic information")
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

def train_latent_conditioner(latent_conditioner_epoch, latent_conditioner_dataloader, latent_conditioner_validation_dataloader, latent_conditioner, latent_conditioner_lr, weight_decay=1e-4, is_image_data=True):

    writer = SummaryWriter(log_dir = './LatentConditionerRuns', comment = 'LatentConditioner')

    loss=0
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    latent_conditioner_optimized = torch.optim.AdamW(latent_conditioner.parameters(), lr=latent_conditioner_lr, weight_decay=weight_decay)
    
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
    overfitting_threshold = 10.0  # Stop if val_loss > 10x train_loss

    from torchinfo import summary
    import math

    summary(latent_conditioner, (64,1,im_size*im_size))

    latent_conditioner = latent_conditioner.to(device)
    
    latent_conditioner.apply(safe_initialize_weights_He)
    latent_conditioner.apply(add_sn)  # Re-enabled for regularization

    # Data analysis for first epoch
    data_analyzed = False

    for epoch in range(latent_conditioner_epoch):
        start_time = time.time()
        latent_conditioner.train(True)
        
        epoch_loss = 0
        epoch_loss_y1 = 0
        epoch_loss_y2 = 0
        num_batches = 0
        
        for i, (x, y1, y2) in enumerate(latent_conditioner_dataloader):
            
            # For image data, keep as flattened - model will handle reshaping internally
            # For parametric data, keep as 1D vector (no reshaping needed)
            
            if epoch==0 and i==0:
                print('dataset_shape', x.shape,y1.shape,y2.shape)
            
            # Apply outline-preserving augmentations for CNN (only for image data)
            if is_image_data and torch.rand(1) < 0.6:  # 60% chance - more conservative for outlines
                # Temporarily reshape to 2D for augmentation
                im_size = int(math.sqrt(x.shape[-1]))
                x_2d = x.reshape(-1, im_size, im_size)
                x_2d = apply_outline_preserving_augmentations(x_2d, prob=0.6)
                x = x_2d.reshape(x.shape[0], -1)  # Flatten back
                
            # EXTREME mixup augmentation for better generalization  
            if torch.rand(1) < 0.2 and x.size(0) > 1:  # 20% chance
                alpha = 0.8  # Much more aggressive mixing
                lam = np.random.beta(alpha, alpha)
                batch_size = x.size(0)
                index = torch.randperm(batch_size).to(x.device)
                
                x = lam * x + (1 - lam) * x[index, :]
                y1 = lam * y1 + (1 - lam) * y1[index, :]
                y2 = lam * y2 + (1 - lam) * y2[index, :]
            
            x, y1, y2 = x.to(device), y1.to(device), y2.to(device)
            
            # Add STRONG Gaussian noise for regularization (reduced since we have image augmentations)
            if torch.rand(1) < 0.1:  # Reduced from 20% to 10% chance
                noise = torch.randn_like(x) * 0.03  # Reduced noise - we have better augmentations now
                x = x + noise
            
            latent_conditioner_optimized.zero_grad(set_to_none=True)

            y_pred1, y_pred2 = latent_conditioner(x)
            
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
            
            loss.backward()
            
            # Gradient clipping for training stability - prevents exploding gradients
            torch.nn.utils.clip_grad_norm_(latent_conditioner.parameters(), max_norm=1.0)
            
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
        
        # Diagnostic variables for first few validation batches
        first_val_batch_logged = False
        
        with torch.no_grad():
            for i, (x_val, y1_val, y2_val) in enumerate(latent_conditioner_validation_dataloader):
                # For image data, keep as flattened - model will handle reshaping internally
                # For parametric data, keep as 1D vector
                x_val, y1_val, y2_val = x_val.to(device), y1_val.to(device), y2_val.to(device)
                
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
        
        print('[%d/%d]\tTrain: %.4E (y1:%.4E, y2:%.4E), Val: %.4E (y1:%.4E, y2:%.4E), LR: %.2E (%s), ETA: %.2f h, Patience: %d/%d' % 
              (epoch, latent_conditioner_epoch, avg_train_loss, avg_train_loss_y1, avg_train_loss_y2, 
               avg_val_loss, avg_val_loss_y1, avg_val_loss_y2,
               current_lr, scheduler_info,
               (latent_conditioner_epoch-epoch)*epoch_duration/3600, patience_counter, patience))
               
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
