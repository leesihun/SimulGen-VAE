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
    def __init__(self, latent_conditioner_filter, latent_dim_end, input_shape, latent_dim, size2):
        super(LatentConditioner, self).__init__()
        self.latent_dim = latent_dim
        self.size2 = size2
        self.latent_conditioner_filter = latent_conditioner_filter
        self.latent_dim_end = latent_dim_end
        self.input_shape = input_shape
        self.num_latent_conditioner_filter = len(self.latent_conditioner_filter)

        modules = []
        modules.append(nn.Linear(self.input_shape, self.latent_conditioner_filter[0]))
        for i in range(1, self.num_latent_conditioner_filter-1):
            modules.append(nn.Linear(self.latent_conditioner_filter[i-1], self.latent_conditioner_filter[i]))
            modules.append(nn.LeakyReLU(0.2))
            modules.append(nn.Dropout(0.2))
        self.latent_conditioner = nn.Sequential(*modules)

        self.latent_out = nn.Sequential(nn.Linear(self.latent_conditioner_filter[-2], self.latent_conditioner_filter[-1]),
                                        nn.LeakyReLU(0.2),
                                        nn.GroupNorm(1, self.latent_conditioner_filter[-1]),
                                        nn.Dropout(0.2),
                                        nn.Linear(self.latent_conditioner_filter[-1], self.latent_dim_end),
                                        nn.Tanh())

        self.xs_out = nn.Sequential(nn.Linear(self.latent_conditioner_filter[-2], self.latent_conditioner_filter[-1]),
                                    nn.LeakyReLU(0.2),
                                    nn.GroupNorm(1, self.latent_conditioner_filter[-1]),
                                    nn.Dropout(0.2),
                                    nn.Linear(self.latent_conditioner_filter[-1], self.latent_dim*self.size2),
                                    nn.Unflatten(1, (self.size2, self.latent_dim)),
                                    nn.Tanh())

    def forward(self, x):
        x = self.latent_conditioner(x)
        latent_out = self.latent_out(x)
        xs_out = self.xs_out(x)

        return latent_out, xs_out



class ImprovedConvResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super().__init__()
        
        self.conv1 = nn.Conv2d(in_channel, out_channel, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.conv2 = nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        
        # Skip connection handling
        self.skip = nn.Sequential()
        if stride != 1 or in_channel != out_channel:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channel, out_channel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channel)
            )

    def forward(self, x):
        residual = self.skip(x)
        
        out = F.gelu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
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
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()

        self.seq = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channel),
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
            nn.Conv2d(1, self.latent_conditioner_filter[0], kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(self.latent_conditioner_filter[0]),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        ))
        
        # Progressive feature extraction with attention
        for i in range(1, self.num_latent_conditioner_filter):
            stride = 2 if i < self.num_latent_conditioner_filter - 1 else 1
            block = nn.Sequential(
                ImprovedConvResBlock(self.latent_conditioner_filter[i-1], self.latent_conditioner_filter[i], stride),
                ImprovedConvResBlock(self.latent_conditioner_filter[i], self.latent_conditioner_filter[i], 1),
                SEBlock(self.latent_conditioner_filter[i])
            )
            self.backbone.append(block)
        
        # Adaptive pooling and feature size calculation
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        final_feature_size = self.latent_conditioner_filter[-1] * 16  # 4*4
        
        # Separate heads for different outputs
        self.latent_head = nn.Sequential(
            nn.Dropout(self.dropout_rate),
            nn.Linear(final_feature_size, final_feature_size // 2),
            nn.BatchNorm1d(final_feature_size // 2),
            nn.GELU(),
            nn.Dropout(self.dropout_rate // 2),
            nn.Linear(final_feature_size // 2, self.latent_dim_end),
            nn.Tanh()
        )
        
        self.xs_head = nn.Sequential(
            nn.Dropout(self.dropout_rate),
            nn.Linear(final_feature_size, final_feature_size // 2),
            nn.BatchNorm1d(final_feature_size // 2),
            nn.GELU(),
            nn.Dropout(self.dropout_rate // 2),
            nn.Linear(final_feature_size // 2, self.latent_dim * self.size2),
            nn.Unflatten(1, (self.size2, self.latent_dim)),
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
        
        # Separate outputs
        latent_out = self.latent_head(features)
        xs_out = self.xs_head(features)

        return latent_out, xs_out




import time
from modules.common import initialize_weights_He
from torchvision.transforms import v2
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import pytorch_warmup as warmup

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
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(latent_conditioner_optimized, T_max = latent_conditioner_epoch, eta_min = latent_conditioner_lr * 0.01)
    
    # Early stopping parameters
    best_val_loss = float('inf')
    patience = 100
    patience_counter = 0

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

    latent_conditioner.apply(initialize_weights_He)

    for epoch in range(latent_conditioner_epoch):
        start_time = time.time()
        latent_conditioner.train(True)
        epoch_loss = 0
        epoch_loss_y1 = 0
        epoch_loss_y2 = 0
        num_batches = 0
        
        for i, (x, y1, y2) in enumerate(latent_conditioner_dataloader):

            x = x.reshape([x.shape[0], int(math.sqrt(x.shape[1])), int(math.sqrt(x.shape[1]))])
            if epoch==0 and i==0:
                print('dataset_shape', x.shape,y1.shape,y2.shape)
            # Apply data augmentation randomly to some samples
            if torch.rand(1) < 0.3:  # 30% chance of augmentation
                x_aug = []
                for img in x:
                    img_np = (img.squeeze().cpu().numpy() * 255).astype(np.uint8)
                    img_aug = augmentation(img_np)
                    x_aug.append(img_aug.unsqueeze(0))
                x = torch.cat(x_aug, dim=0)
            
            x, y1, y2 = x.to(device), y1.to(device), y2.to(device)
            
            for param in latent_conditioner.parameters():
                param.grad = None

            y_pred1, y_pred2 = latent_conditioner(x)

            A = nn.MSELoss()(y_pred1, y1)
            B = nn.MSELoss()(y_pred2, y2)

            loss = A + B
            epoch_loss += loss.item()
            epoch_loss_y1 += A.item()
            epoch_loss_y2 += B.item()
            num_batches += 1

            loss.backward()
            
            # Gradient clipping to prevent exploding gradients
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
        
        with torch.no_grad():
            for i, (x_val, y1_val, y2_val) in enumerate(latent_conditioner_validation_dataloader):
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

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            patience_counter = 0
            # Save best model
            torch.save(latent_conditioner.state_dict(), 'checkpoints/latent_conditioner_best.pth')
        else:
            patience_counter += 1
            
        # Learning rate scheduling
        scheduler.step()

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

        print('[%d/%d]\tTrain: %.4E (y1:%.4E, y2:%.4E), Val: %.4E (y1:%.4E, y2:%.4E), LR: %.2E, ETA: %.2f h, Patience: %d/%d' % 
              (epoch, latent_conditioner_epoch, avg_train_loss, avg_train_loss_y1, avg_train_loss_y2, 
               avg_val_loss, avg_val_loss_y1, avg_val_loss_y2,
               latent_conditioner_optimized.param_groups[0]['lr'], 
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
