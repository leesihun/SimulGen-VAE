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
from torch.utils.tensorboard import SummaryWriter
from torchinfo import summary
from modules.pca_preprocessor import PCAPreprocessor

DEFAULT_IMAGE_SIZE = 256
INTERPOLATION_METHOD = cv2.INTER_CUBIC
im_size = DEFAULT_IMAGE_SIZE

def read_latent_conditioner_dataset_img(param_dir, param_data_type, debug_mode=0):
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
            # Normalize to [0, 1] range for better training stability
            raw_images[i] = resized_im / 255.0

        # Standard processing for CNN/ViT (no PCA here - use read_latent_conditioner_dataset_img_pca for PCA_MLP mode)
        latent_conditioner_data = raw_images.reshape(len(files), -1)
        latent_conditioner_data_shape = (im_size, im_size)
            
    else:
        raise NotImplementedError('Data type not supported')

    return latent_conditioner_data, latent_conditioner_data_shape

def read_latent_conditioner_dataset_img_pca(param_dir, param_data_type, debug_mode=0, pca_components=256, pca_patch_size=0):
    """
    PCA_MLP mode: Read images and convert them to PCA coefficients for MLP-based latent conditioner.
    This function forces PCA preprocessing and returns flattened PCA coefficients suitable for MLP input.
    """
    cur_dir = os.getcwd()
    file_dir = cur_dir + param_dir

    if param_data_type == ".jpg" or param_data_type == ".png":
        if debug_mode == 1:
            print('PCA_MLP mode: Reading image dataset from ' + file_dir + '\n')

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
            # Normalize to [0, 1] range for better training stability
            raw_images[i] = resized_im / 255.0

        # Apply PCA preprocessing (forced for PCA_MLP mode)
        print(f'PCA_MLP mode: Applying PCA preprocessing with {pca_components} components')
        if pca_patch_size > 0:
            print(f'Using patch-based PCA with patch size {pca_patch_size}')
        
        pca_preprocessor = PCAPreprocessor(
            n_components=pca_components, 
            patch_size=pca_patch_size if pca_patch_size > 0 else None
        )
        
        if debug_mode == 1:
            print('Fitting new PCA model on training data')
        pca_preprocessor.fit(raw_images)
    
        # Transform images using PCA
        pca_tensor = pca_preprocessor.transform(raw_images)  # Returns torch.Tensor
        
        # Convert to numpy and flatten for MLP compatibility
        if len(pca_tensor.shape) == 4:  # (n_samples, channels, height, width)
            latent_conditioner_data = pca_tensor.view(pca_tensor.shape[0], -1).numpy()
        else:
            latent_conditioner_data = pca_tensor.numpy()
        
        # For MLP mode, we return flattened shape
        latent_conditioner_data_shape = (latent_conditioner_data.shape[1],)
        
        if debug_mode == 1:
            print(f'PCA_MLP output shape: {latent_conditioner_data_shape}')
            print(f'Data reduced from {im_size*im_size} to {latent_conditioner_data.shape[1]} dimensions')
            
    else:
        raise NotImplementedError('PCA_MLP mode only supports .jpg/.png files')

    return latent_conditioner_data, latent_conditioner_data_shape

def read_latent_conditioner_dataset(param_dir, param_data_type): # For normal parametric approach: .csv
    latent_conditioner_data = pd.read_csv(param_dir, header=None)
    latent_conditioner_data = latent_conditioner_data.values

    return latent_conditioner_data

# GPU-optimized outline-preserving augmentation functions
def apply_outline_preserving_augmentations(x, prob=0.5):
    if not torch.rand(1, device=x.device) < prob:
        return x  # Skip augmentation
    
    batch_size, height, width = x.shape
    
    # Horizontal flip
    if torch.rand(1, device=x.device) < 0.3:
        flip_mask = torch.rand(batch_size, device=x.device) < 0.5
        if flip_mask.any():
            x_flipped = torch.flip(x, dims=[2])
            x = torch.where(flip_mask.unsqueeze(1).unsqueeze(2), x_flipped, x)
    
    # Small translation
    if torch.rand(1, device=x.device) < 0.5:
        shift_x = torch.randint(-1, 2, (batch_size,), device=x.device)
        shift_y = torch.randint(-1, 2, (batch_size,), device=x.device)
        for i in range(batch_size):
            if shift_x[i] != 0:
                x[i] = torch.roll(x[i], shifts=int(shift_x[i]), dims=1)
            if shift_y[i] != 0:
                x[i] = torch.roll(x[i], shifts=int(shift_y[i]), dims=0)
    
    # Small rotation
    if torch.rand(1, device=x.device) < 0.3:
        angles = (torch.rand(batch_size, device=x.device) - 0.5) * 10
        angles_rad = angles * math.pi / 180
        
        # Simple rotation using affine transformation for small angles
        for i in range(batch_size):
            if abs(angles[i]) > 0.5:
                cos_a, sin_a = torch.cos(angles_rad[i]), torch.sin(angles_rad[i])
                theta = torch.tensor([[cos_a, -sin_a, 0], [sin_a, cos_a, 0]], 
                                   device=x.device, dtype=x.dtype).unsqueeze(0)
                grid = F.affine_grid(theta, (1, 1, height, width), align_corners=False)
                x[i:i+1] = F.grid_sample(x[i:i+1].unsqueeze(0), grid, 
                                       mode='bilinear', padding_mode='border', align_corners=False).squeeze(0)
    
    # Slight scaling
    if torch.rand(1, device=x.device) < 0.3:
        scales = 0.95 + (torch.rand(batch_size, device=x.device) * 0.1)
        
        for i in range(batch_size):
            if abs(scales[i] - 1.0) > 0.01:
                scale = scales[i]
                theta = torch.tensor([[scale, 0, 0], [0, scale, 0]], 
                                   device=x.device, dtype=x.dtype).unsqueeze(0)
                grid = F.affine_grid(theta, (1, 1, height, width), align_corners=False)
                x[i:i+1] = F.grid_sample(x[i:i+1].unsqueeze(0), grid, 
                                       mode='bilinear', padding_mode='border', align_corners=False).squeeze(0)
    
    
    return x


def safe_cuda_initialization(debug_mode=0):
    """Safely check CUDA availability with error handling and diagnostics"""

    if not torch.cuda.is_available():
        if debug_mode == 1:
            print("❌ CUDA not available, using CPU")
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
    optimizer = torch.optim.AdamW(latent_conditioner.parameters(), lr=latent_conditioner_lr, weight_decay=weight_decay)
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
    
    best_val_loss = float('inf')
    patience = 100000
    patience_counter = 0
    min_delta = 1e-8
    overfitting_threshold = 1000.0

    
    latent_conditioner = latent_conditioner.to(device)
    
    latent_conditioner.apply(safe_initialize_weights_He)
    
    model_summary_shown = False

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
            
            if not model_summary_shown:
                batch_size = x.shape[0]
                input_features = x.shape[-1]
                summary(latent_conditioner, (batch_size, 1, input_features))
                model_summary_shown = True
            
            if is_image_data and torch.rand(1, device=x.device) < 0.8:
                im_size = int(math.sqrt(x.shape[-1]))
                x_2d = x.reshape(-1, im_size, im_size)
                x_2d = apply_outline_preserving_augmentations(x_2d, prob=0.8)
                x = x_2d.reshape(x.shape[0], -1)
                
            if torch.rand(1, device=x.device) < 0.1 and x.size(0) > 1:
                alpha = 0.2
                lam = torch.tensor(np.random.beta(alpha, alpha), device=x.device, dtype=x.dtype)
                batch_size = x.size(0)
                index = torch.randperm(batch_size, device=x.device)
                
                x = lam * x + (1 - lam) * x[index, :]
                y1 = lam * y1 + (1 - lam) * y1[index, :]
                y2 = lam * y2 + (1 - lam) * y2[index, :]
            
            if torch.rand(1, device=x.device) < 0.1:
                noise = torch.randn_like(x) * 0.01
                x = x + noise
            
            latent_conditioner_optimized.zero_grad(set_to_none=True)

            y_pred1, y_pred2 = latent_conditioner(x)
            

            label_smooth = 0.1
            y1_smooth = y1 * (1 - label_smooth) + torch.randn_like(y1) * label_smooth * 0.1
            y2_smooth = y2 * (1 - label_smooth) + torch.randn_like(y2) * label_smooth * 0.1
            
            A = nn.MSELoss()(y_pred1, y1_smooth)
            B = nn.MSELoss()(y_pred2, y2_smooth)
            

            loss = A*9 + B 

            
            epoch_loss += loss.item()
            epoch_loss_y1 += A.item()
            epoch_loss_y2 += B.item()
            num_batches += 1
            
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(latent_conditioner.parameters(), max_norm=5.0)
            
            latent_conditioner_optimized.step()
        
        
        avg_train_loss = epoch_loss / num_batches
        avg_train_loss_y1 = epoch_loss_y1 / num_batches
        avg_train_loss_y2 = epoch_loss_y2 / num_batches

        latent_conditioner.eval()
        val_loss = 0
        val_loss_y1 = 0
        val_loss_y2 = 0
        val_batches = 0
        
        if epoch % 1 == 0:
            with torch.no_grad():
                for i, (x_val, y1_val, y2_val) in enumerate(latent_conditioner_validation_dataloader):
                        x_val, y1_val, y2_val = x_val.to(device), y1_val.to(device), y2_val.to(device)
                    
                        y_pred1_val, y_pred2_val = latent_conditioner(x_val)
                        
                        A_val = nn.MSELoss()(y_pred1_val, y1_val)
                        B_val = nn.MSELoss()(y_pred2_val, y2_val)
                        
                        val_loss += (A_val*9 + B_val).item()
                        val_loss_y1 += A_val.item()
                        val_loss_y2 += B_val.item()
                        val_batches += 1

                avg_val_loss = val_loss / val_batches
                avg_val_loss_y1 = val_loss_y1 / val_batches
                avg_val_loss_y2 = val_loss_y2 / val_batches

                overfitting_ratio = avg_val_loss / max(avg_train_loss, 1e-8)
                if overfitting_ratio > overfitting_threshold:
                    print(f'Severe overfitting detected! Val/Train ratio: {overfitting_ratio:.1f}')
                    print(f'Stopping early at epoch {epoch}')
                    break
                    
                if avg_val_loss < best_val_loss - min_delta:
                    best_val_loss = avg_val_loss
                    patience_counter = 0
                else:
                    patience_counter += 1
            
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
               
        if patience_counter >= patience:
            print(f'Early stopping at epoch {epoch}. Best validation loss: {best_val_loss:.4E}')
            break

    torch.save(latent_conditioner.state_dict(), 'checkpoints/latent_conditioner.pth')
    torch.save(latent_conditioner, 'model_save/LatentConditioner')

    return avg_val_loss