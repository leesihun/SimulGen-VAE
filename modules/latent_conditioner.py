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

def read_latent_conditioner_dataset_img(param_dir, param_data_type):
    cur_dir = os.getcwd()
    file_dir = cur_dir+param_dir

    im_size = 128

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


class LatentConditioner(nn.Module):
    def __init__(self, latent_conditioner_filter, latent_dim_end, input_shape, latent_dim, size2, dropout_rate=0.3):
        super(LatentConditioner, self).__init__()
        self.latent_dim = latent_dim
        self.size2 = size2
        self.latent_conditioner_filter = latent_conditioner_filter
        self.latent_dim_end = latent_dim_end
        self.input_shape = input_shape
        self.num_latent_conditioner_filter = len(self.latent_conditioner_filter)
        self.dropout_rate = dropout_rate

        # Backbone feature extractor
        modules = []
        modules.append(nn.Linear(self.input_shape, self.latent_conditioner_filter[0]))
        for i in range(1, self.num_latent_conditioner_filter-1):
            modules.append(nn.Linear(self.latent_conditioner_filter[i-1], self.latent_conditioner_filter[i]))
            modules.append(nn.LeakyReLU(0.2))
            modules.append(nn.Dropout(0.1))  # Reduced dropout
        self.latent_conditioner = nn.Sequential(*modules)

        # Simplified output heads - same as image version
        final_feature_size = self.latent_conditioner_filter[-2]
        
        # Balanced heads to match image version
        hidden_size = final_feature_size // 2
        self.latent_out = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(final_feature_size, hidden_size),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_size, self.latent_dim_end),
            nn.Tanh()
        )

        self.xs_out = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(final_feature_size, hidden_size),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_size, self.latent_dim * self.size2),
            nn.Tanh()
        )

    def forward(self, x):
        features = self.latent_conditioner(x)
        latent_out = self.latent_out(features)
        
        xs_out = self.xs_out(features)
        xs_out = xs_out.unflatten(1, (self.size2, self.latent_dim))

        return latent_out, xs_out



class ImprovedConvResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=True)
        self.gn1 = nn.GroupNorm(min(32, out_channel//4), out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, bias=True)
        self.gn2 = nn.GroupNorm(min(32, out_channel//4), out_channel)
        
        # Skip connection handling
        self.skip = nn.Sequential()
        if stride != 1 or in_channel != out_channel:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, bias=True),
                nn.GroupNorm(min(32, out_channel//4), out_channel)
            )

    def forward(self, x):
        residual = self.skip(x)
        
        out = F.gelu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        out += residual
        out = F.gelu(out)
        return out

class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel attention"""
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.GELU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        if x is None or x.size(0) == 0:
            return x
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()

        self.seq = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1),
            nn.GroupNorm(min(32, out_channel//4), out_channel),
            nn.GELU(),
            nn.AvgPool2d(2)
        )

    def forward(self, x):
        return self.seq(x)

class LatentConditionerImg(nn.Module):
    def __init__(self, latent_conditioner_filter, latent_dim_end, input_shape, latent_dim, size2, latent_conditioner_data_shape, dropout_rate=0.3):
        super(LatentConditionerImg, self).__init__()
        self.latent_dim = latent_dim
        self.size2 = size2
        self.latent_conditioner_filter = latent_conditioner_filter
        self.latent_dim_end = latent_dim_end
        self.input_shape = input_shape
        self.num_latent_conditioner_filter = len(self.latent_conditioner_filter)
        self.latent_conditioner_data_shape = latent_conditioner_data_shape
        self.dropout_rate = dropout_rate

        # Shared feature extractor backbone
        self.backbone = nn.ModuleList()
        
        # Initial conv
        self.backbone.append(nn.Sequential(
            nn.Conv2d(1, self.latent_conditioner_filter[0], kernel_size=7, stride=2, padding=3, bias=True),
            nn.GroupNorm(min(32, self.latent_conditioner_filter[0]//4), self.latent_conditioner_filter[0]),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        ))
        
        # Progressive feature extraction with attention
        for i in range(1, self.num_latent_conditioner_filter):
            stride = 2 if i < self.num_latent_conditioner_filter - 1 else 1
            block = nn.Sequential(
                ImprovedConvResBlock(self.latent_conditioner_filter[i-1], self.latent_conditioner_filter[i], stride),
                ImprovedConvResBlock(self.latent_conditioner_filter[i], self.latent_conditioner_filter[i], 1)
            )
            self.backbone.append(block)
        
        # Adaptive pooling and feature size calculation
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        final_feature_size = self.latent_conditioner_filter[-1] * 16  # 4*4
        
        # Simplified output heads - removing complexity that causes overfitting
        # Keep feature dimension to avoid bottleneck
        
        # Balanced latent head - single hidden layer to prevent underfitting
        hidden_size = final_feature_size // 2
        self.latent_out = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(final_feature_size, hidden_size),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_size, self.latent_dim_end),
            nn.Tanh()
        )
        
        # Balanced xs head - single hidden layer to prevent underfitting  
        self.xs_out = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(final_feature_size, hidden_size),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_size, self.latent_dim * self.size2),
            nn.Tanh()
        )

    def forward(self, x):
        im_size = 128
        x = x.reshape(-1, 1, im_size, im_size)
        
        # Shared feature extraction
        features = x
        for block in self.backbone:
            features = block(features)
        
        # Global feature pooling
        features = self.adaptive_pool(features)
        features = features.flatten(1)
        
        # Simplified forward pass through output heads
        latent_out = self.latent_out(features)
        
        xs_out = self.xs_out(features)
        xs_out = xs_out.unflatten(1, (self.size2, self.latent_dim))

        return latent_out, xs_out




import time
from modules.common import initialize_weights_He, add_sn
from torchvision.transforms import v2
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import pytorch_warmup as warmup

# SAM (Sharpness-Aware Minimization) integrated directly into training loop

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

def train_latent_conditioner(latent_conditioner_epoch, latent_conditioner_dataloader, latent_conditioner_validation_dataloader, latent_conditioner, latent_conditioner_lr, weight_decay=1e-4):
    im_size = 128

    writer = SummaryWriter(log_dir = './LatentConditionerRuns', comment = 'LatentConditioner')

    loss=0
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Reduce learning rate and add weight decay for regularization
    latent_conditioner_optimized = torch.optim.AdamW(latent_conditioner.parameters(), lr=latent_conditioner_lr, weight_decay=weight_decay)
    
    # SAM parameters - can disable for testing
    use_sam = True
    sam_rho = 0.05
    print(f"SAM enabled: {use_sam}, rho: {sam_rho}")
    
    # EMA for weight averaging
    ema_decay = 0.999
    ema_model = {name: param.clone() for name, param in latent_conditioner.named_parameters()}
    
    # Snapshot ensemble - save models at different stages
    snapshot_models = []
    snapshot_interval = latent_conditioner_epoch // 10  # Save 10 snapshots
    
    # Advanced learning rate scheduling
    warmup_epochs = 10
    # Linear warmup scheduler for first 10 epochs - increased initial LR
    warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
        latent_conditioner_optimized, 
        start_factor=0.1,  # Increased from 0.01 to help validation learning
        total_iters=warmup_epochs
    )
    
    # Main cosine scheduler with much lower eta_min
    main_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        latent_conditioner_optimized, 
        T_max=latent_conditioner_epoch - warmup_epochs, 
        eta_min=1e-8
    )
    
    # Plateau scheduler as backup with more conservative settings
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        latent_conditioner_optimized, 
        mode='min', 
        patience=100,  # Increased from 50 to 100
        factor=0.5,    # More aggressive reduction from 0.8 to 0.5
        min_lr=1e-7,   # Added minimum learning rate
        verbose=True
    )
    
    # Early stopping parameters - reduced patience for faster convergence and stricter improvement threshold
    best_val_loss = float('inf')
    patience = 2000   # Increased to allow more training for validation improvement
    patience_counter = 0
    min_delta = 1e-5  # Relaxed for easier improvement detection

    # Data augmentation transforms
    augmentation = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
    ])

    from torchinfo import summary
    import math

    summary(latent_conditioner, (64,1,im_size,im_size))

    
    latent_conditioner = latent_conditioner.to(device)
    def safe_initialize_weights_He(m):
        if isinstance(m, (nn.Conv1d, nn.ConvTranspose1d, nn.Conv2d, nn.ConvTranspose2d)):
            nn.init.kaiming_uniform_(m.weight.data, nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)
        elif isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight.data)
            if m.bias is not None:  # Add this check
                nn.init.constant_(m.bias.data, 0)
    
    latent_conditioner.apply(safe_initialize_weights_He)
    latent_conditioner.apply(add_sn)  # Apply spectral normalization for 1-Lipschitz constraint

    # Data analysis for first epoch
    data_analyzed = False

    for epoch in range(latent_conditioner_epoch):
        start_time = time.time()
        latent_conditioner.train(True)
        
        # Progressive dropout - start high, reduce over time
        current_dropout = max(0.1, 0.3 * (1 - epoch / latent_conditioner_epoch))
        for module in latent_conditioner.modules():
            if isinstance(module, nn.Dropout):
                module.p = current_dropout
                
        epoch_loss = 0
        epoch_loss_y1 = 0
        epoch_loss_y2 = 0
        num_batches = 0
        
        for i, (x, y1, y2) in enumerate(latent_conditioner_dataloader):

            x = x.reshape([x.shape[0], int(math.sqrt(x.shape[1])), int(math.sqrt(x.shape[1]))])
            
            if epoch==0 and i==0:
                print('dataset_shape', x.shape,y1.shape,y2.shape)
                
            # Comprehensive data analysis for first few batches
            if not data_analyzed and epoch == 0 and i < 3:
                print(f"\n=== Data Analysis Batch {i} ===")
                print(f"Input Statistics:")
                print(f"  X - Min: {x.min().item():.6f}, Max: {x.max().item():.6f}, Mean: {x.mean().item():.6f}, Std: {x.std().item():.6f}")
                print(f"Target Statistics:")
                print(f"  Y1 - Min: {y1.min().item():.6f}, Max: {y1.max().item():.6f}, Mean: {y1.mean().item():.6f}, Std: {y1.std().item():.6f}")
                print(f"  Y2 - Min: {y2.min().item():.6f}, Max: {y2.max().item():.6f}, Mean: {y2.mean().item():.6f}, Std: {y2.std().item():.6f}")
                
                # Check for outliers
                y1_outliers = torch.sum(torch.abs(y1) > 3 * y1.std()).item()
                y2_outliers = torch.sum(torch.abs(y2) > 3 * y2.std()).item()
                print(f"Outliers (>3σ): Y1={y1_outliers}, Y2={y2_outliers}")
                
                if i == 2:  # Mark analysis as complete after 3 batches
                    data_analyzed = True
                    print("=== Data Analysis Complete ===\n")
            # Cutout - randomly mask patches for severe overfitting
            if torch.rand(1) < 0.4:  # 40% chance
                cutout_size = 16  # 16x16 patches on 128x128 images
                for b in range(x.size(0)):
                    if torch.rand(1) < 0.5:  # 50% of samples get cutout
                        img_size = int(math.sqrt(x.shape[1]))
                        cx = np.random.randint(0, img_size)
                        cy = np.random.randint(0, img_size)
                        x1 = max(0, cx - cutout_size // 2)
                        y1_cut = max(0, cy - cutout_size // 2)
                        x2 = min(img_size, cx + cutout_size // 2)
                        y2_cut = min(img_size, cy + cutout_size // 2)
                        
                        # Convert to flattened indices
                        mask = torch.ones_like(x[b])
                        for i in range(x1, x2):
                            for j in range(y1_cut, y2_cut):
                                mask[i * img_size + j] = 0
                        x[b] = x[b] * mask
            
            # Mixup augmentation for better generalization  
            if torch.rand(1) < 0.2 and x.size(0) > 1:  # Reduced to 20% to combine with cutout
                alpha = 0.2
                lam = np.random.beta(alpha, alpha)
                batch_size = x.size(0)
                index = torch.randperm(batch_size).to(x.device)
                
                x = lam * x + (1 - lam) * x[index, :]
                y1 = lam * y1 + (1 - lam) * y1[index, :]
                y2 = lam * y2 + (1 - lam) * y2[index, :]
            
            x, y1, y2 = x.to(device), y1.to(device), y2.to(device)
            
            latent_conditioner_optimized.zero_grad(set_to_none=True)

            y_pred1, y_pred2 = latent_conditioner(x)
            
            # Analyze predictions for first few batches
            if not data_analyzed and epoch == 0 and i < 3:
                print(f"Prediction Statistics Batch {i}:")
                print(f"  Y1_pred - Min: {y_pred1.min().item():.6f}, Max: {y_pred1.max().item():.6f}, Mean: {y_pred1.mean().item():.6f}, Std: {y_pred1.std().item():.6f}")
                print(f"  Y2_pred - Min: {y_pred2.min().item():.6f}, Max: {y_pred2.max().item():.6f}, Mean: {y_pred2.mean().item():.6f}, Std: {y_pred2.std().item():.6f}")

            # Label smoothing for better generalization
            epsilon = 0.1
            A = nn.MSELoss()(y_pred1, y1) + epsilon * torch.mean(y_pred1**2)
            B = nn.MSELoss()(y_pred2, y2) + epsilon * torch.mean(y_pred2**2)
            
            # Log individual losses for first few batches
            if not data_analyzed and epoch == 0 and i < 3:
                print(f"  Loss A (Y1): {A.item():.6f}, Loss B (Y2): {B.item():.6f}, Ratio A/B: {A.item()/B.item():.3f}")
                print("---")

            # Gradient penalty for smoothness
            if epoch > 10:  # Apply after initial convergence
                x.requires_grad_(True)
                y_pred1_gp, y_pred2_gp = latent_conditioner(x)
                
                grad_outputs1 = torch.ones_like(y_pred1_gp)
                grad_outputs2 = torch.ones_like(y_pred2_gp)
                
                gradients1 = torch.autograd.grad(outputs=y_pred1_gp, inputs=x, 
                                               grad_outputs=grad_outputs1, 
                                               create_graph=True, retain_graph=True)[0]
                gradients2 = torch.autograd.grad(outputs=y_pred2_gp, inputs=x, 
                                               grad_outputs=grad_outputs2, 
                                               create_graph=True, retain_graph=True)[0]
                
                gradient_penalty = torch.mean(gradients1**2) + torch.mean(gradients2**2)
                x.requires_grad_(False)
                
                loss = A + B + 0.01 * gradient_penalty
            else:
                loss = A + B
                
            # Information bottleneck regularization (most aggressive)
            kl_loss = 0
            if epoch > 50:  # Apply after some convergence
                for name, param in latent_conditioner.named_parameters():
                    if 'weight' in name and len(param.shape) > 1:  # Only weight matrices
                        # Encourage low-rank structure (information bottleneck)
                        U, S, V = torch.svd(param)
                        # Penalize large singular values beyond top-k
                        k = min(param.shape) // 4  # Keep only 1/4 of singular values large
                        if len(S) > k:
                            kl_loss += 0.001 * torch.sum(S[k:] ** 2)
            
            loss = loss + kl_loss
            
            # Update EMA model
            with torch.no_grad():
                for name, param in latent_conditioner.named_parameters():
                    ema_model[name] = ema_decay * ema_model[name] + (1 - ema_decay) * param
            epoch_loss += loss.item()
            epoch_loss_y1 += A.item()
            epoch_loss_y2 += B.item()
            num_batches += 1

            # SAM implementation integrated into training loop
            if use_sam and epoch > 10:  # Apply SAM after initial convergence
                # First forward-backward pass
                loss.backward(create_graph=True)
                
                # Compute gradient norm for SAM
                grad_norm = 0.0
                for param in latent_conditioner.parameters():
                    if param.grad is not None:
                        grad_norm += param.grad.norm().item() ** 2
                grad_norm = grad_norm ** 0.5
                
                # SAM perturbation step - move weights in gradient direction
                scale = sam_rho / (grad_norm + 1e-12)
                old_params = {}
                for name, param in latent_conditioner.named_parameters():
                    if param.grad is not None:
                        old_params[name] = param.data.clone()
                        param.data.add_(param.grad, alpha=scale)
                
                # Zero gradients and recompute loss with perturbed weights
                latent_conditioner_optimized.zero_grad()
                y_pred1_sam, y_pred2_sam = latent_conditioner(x)
                A_sam = nn.MSELoss()(y_pred1_sam, y1) + epsilon * torch.mean(y_pred1_sam**2)
                B_sam = nn.MSELoss()(y_pred2_sam, y2) + epsilon * torch.mean(y_pred2_sam**2)
                
                # Add other losses if they exist
                sam_loss = A_sam + B_sam
                if 'kl_loss' in locals():
                    sam_loss += kl_loss
                    
                # Second backward pass
                sam_loss.backward()
                
                # Restore original weights before optimizer step
                for name, param in latent_conditioner.named_parameters():
                    if name in old_params:
                        param.data = old_params[name]
                
                # Gradient clipping and step
                torch.nn.utils.clip_grad_norm_(latent_conditioner.parameters(), max_norm=5.0)
                latent_conditioner_optimized.step()
                
            else:
                # Regular training without SAM
                loss.backward()
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
        
        # Diagnostic variables for first few validation batches
        first_val_batch_logged = False
        
        with torch.no_grad():
            for i, (x_val, y1_val, y2_val) in enumerate(latent_conditioner_validation_dataloader):
                x_val = x_val.reshape([x_val.shape[0], int(math.sqrt(x_val.shape[1])), int(math.sqrt(x_val.shape[1]))])
                x_val, y1_val, y2_val = x_val.to(device), y1_val.to(device), y2_val.to(device)
                
                # Test-Time Augmentation - average multiple predictions
                if epoch % 10 == 0:  # Only every 10 epochs to save compute
                    tta_predictions_1 = []
                    tta_predictions_2 = []
                    
                    for tta_iter in range(5):  # 5 different noise patterns
                        x_tta = x_val.clone()
                        if tta_iter > 0:  # Add different noise patterns
                            noise = torch.randn_like(x_tta) * 0.01  # 1% noise
                            x_tta = x_tta + noise
                        
                        with torch.no_grad():
                            pred1, pred2 = latent_conditioner(x_tta)
                            tta_predictions_1.append(pred1)
                            tta_predictions_2.append(pred2)
                    
                    # Average predictions
                    y_pred1_val = torch.mean(torch.stack(tta_predictions_1), dim=0)
                    y_pred2_val = torch.mean(torch.stack(tta_predictions_2), dim=0)
                else:
                    y_pred1_val, y_pred2_val = latent_conditioner(x_val)

                A_val = nn.MSELoss()(y_pred1_val, y1_val) + epsilon * torch.mean(y_pred1_val**2)
                B_val = nn.MSELoss()(y_pred2_val, y2_val) + epsilon * torch.mean(y_pred2_val**2)
                
                # Cosine similarity loss for better latent structure
                if y1_val.numel() > 1:  # Need multiple samples
                    cos_sim_target = torch.cosine_similarity(y1_val[:-1], y1_val[1:], dim=-1)
                    cos_sim_pred = torch.cosine_similarity(y_pred1_val[:-1], y_pred1_val[1:], dim=-1)
                    cosine_loss = nn.MSELoss()(cos_sim_pred, cos_sim_target)
                    A_val += 0.05 * cosine_loss

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
        
        # Plateau scheduler (monitors validation loss)
        plateau_scheduler.step(avg_val_loss)

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

        # Save snapshots for ensemble
        if epoch % snapshot_interval == 0 and epoch > 0:
            snapshot_state = latent_conditioner.state_dict().copy()
            snapshot_models.append(snapshot_state)
            print(f"Saved snapshot {len(snapshot_models)} at epoch {epoch}")
        
        # Save regular checkpoint
        if epoch % 50 == 0:
            torch.save(latent_conditioner.state_dict(), f'checkpoints/latent_conditioner_epoch_{epoch}.pth')

    torch.save(latent_conditioner.state_dict(), 'checkpoints/latent_conditioner.pth')
    torch.save(latent_conditioner, 'model_save/LatentConditioner')

    return avg_val_loss
