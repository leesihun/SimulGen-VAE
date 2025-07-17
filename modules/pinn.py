import torch
import torch.nn as nn
import numpy as np
from torchsummaryX import summary
import torch.nn.functional as F
import cv2
import os
import pandas as pd
import matplotlib.pyplot as plt
import natsort

def read_pinn_dataset_img(param_dir, param_data_type):
    cur_dir = os.getcwd()
    file_dir = cur_dir+param_dir

    im_size = 128

    if param_data_type == ".jpg" or param_data_type == ".png":
        print('Reading image dataset from '+file_dir+'\n')

        files = [f for f in os.listdir(file_dir) if f.endswith(param_data_type)]
        files = natsort.natsorted(files)

        pinn_data = np.zeros((len(files), im_size*im_size))
        i=0

        for file in files:
            print(file)
            file_path = file_dir+'/'+file
            im = cv2.imread(file_path, 0)

            resized_im = cv2.resize(im, (im_size, im_size), interpolation=cv2.INTER_CUBIC)
            pinn_data[i, :] = resized_im.reshape(-1)[:]
            pinn_data_shape = resized_im.shape
            i=i+1
    else:
        raise NotImplementedError('Data type not supported')

    return pinn_data, pinn_data_shape

def read_pinn_dataset(param_dir, param_data_type): # For normal parametric approach: .csv
    pinn_data = pd.read_csv(param_dir, header=None)
    pinn_data = pinn_data.values

    return pinn_data


class PINN(nn.Module):
    def __init__(self, pinn_filter, latent_dim_end, input_shape, latent_dim, size2):
        super(PINN, self).__init__()
        self.latent_dim = latent_dim
        self.size2 = size2
        self.pinn_filter = pinn_filter
        self.latent_dim_end = latent_dim_end
        self.input_shape = input_shape
        self.num_pinn_filter = len(self.pinn_filter)

        modules = []
        modules.append(nn.Linear(self.input_shape, self.pinn_filter[0]))
        for i in range(1, self.num_pinn_filter-1):
            modules.append(nn.Linear(self.pinn_filter[i-1], self.pinn_filter[i]))
            modules.append(nn.LeakyReLU(0.2))
            modules.append(nn.Dropout(0.2))
        self.pinn = nn.Sequential(*modules)

        self.latent_out = nn.Sequence(nn.Linear(self.pinn_filter[-2], self.pinn_filter[-1]),
                                      nn.LeakyReLU(0.2),
                                      nn.GroupNorm(1, self.pinn_filter[-1]),
                                      nn.Dropout(0.2),
                                      nn.Linear(self.pinn_filter[-1], self.latent_dim_end),
                                      nn.Tanh())

        self.xs_out = nn.Sequential(nn.Linear(self.pinn_filter[-2], self.pinn_filter[-1]),
                                    nn.LeakyReLU(0.2),
                                    nn.GroupNorm(1, self.pinn_filter[-1]),
                                    nn.Dropout(0.2),
                                    nn.Linear(self.pinn_filter[-1], self.latent_dim*self.size2),
                                    nn.Unflatten(1, torch.size(self.size2, self.latent_dim)),
                                    nn.Tanh())

    def forward(self, x):
        x = self.pinn(x)
        latent_out = self.latent_out(x)
        xs_out = self.xs_out(x)

        return latent_out, xs_out



class ConvResBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()
        multiple = 5

        self.seq = nn.Sequential(
            nn.Conv2d(in_channel, in_channel*multiple, kernel_size=3, padding=1),
            nn.GroupNorm(min(8, max(1, (in_channel*multiple)//4)), in_channel*multiple),
            nn.GELU(),
            nn.Conv2d(in_channel*multiple, out_channel, kernel_size=5, padding=2),
            nn.GroupNorm(min(8, max(1, out_channel//4)), out_channel),
            nn.GELU(),
            nn.Conv2d(out_channel, out_channel, kernel_size=3, padding=1),
            nn.GroupNorm(min(8, max(1, out_channel//4)), out_channel),
            nn.GELU(),
        )

    def forward(self, x):
        # Reduce residual connection strength to make training more stable
        x = x + 0.05 * self.seq(x)
        return x

class ConvBlock(nn.Module):
    def __init__(self, in_channel, out_channel):
        super().__init__()

        self.seq = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, kernel_size=3, padding=1),
            nn.GroupNorm(min(8, max(1, out_channel//4)), out_channel),
            nn.GELU(),
            nn.AvgPool2d(2)
        )

    def forward(self, x):
        return self.seq(x)

class PINN_img(nn.Module):
    def __init__(self, pinn_filter, latent_dim_end, input_shape, latent_dim, size2, pinn_data_shape, dropout_rate=0.3):
        super(PINN_img, self).__init__()
        self.latent_dim = latent_dim
        self.size2 = size2
        self.pinn_filter = pinn_filter
        self.latent_dim_end = latent_dim_end
        self.input_shape = input_shape
        self.num_pinn_filter = len(self.pinn_filter)
        self.pinn_data_shape = pinn_data_shape
        self.dropout_rate = dropout_rate

        modules = []
        modules.append(nn.Conv2d(1, self.pinn_filter[0], kernel_size=3, padding=1))
        modules.append(nn.GroupNorm(min(8, max(1, self.pinn_filter[0]//4)), self.pinn_filter[0]))
        modules.append(nn.GELU())
        
        for i in range(1, self.num_pinn_filter):
            modules.append(ConvResBlock(self.pinn_filter[i-1], self.pinn_filter[i-1]))
            modules.append(ConvBlock(self.pinn_filter[i-1], self.pinn_filter[i]))

        modules.append(nn.Flatten())
        modules.append(nn.LazyLinear(self.latent_dim_end*8))
        modules.append(nn.GroupNorm(min(8, max(1, (self.latent_dim_end*8)//4)), self.latent_dim_end*8))
        modules.append(nn.GELU())
        modules.append(nn.Dropout(self.dropout_rate))  # Keep dropout only for FC layers
        modules.append(nn.Linear(self.latent_dim_end*8, self.latent_dim_end))
        modules.append(nn.Tanh())

        self.latent_out = nn.Sequential(*modules)

        modules = []
        modules.append(nn.Conv2d(1, self.pinn_filter[0], kernel_size=3, padding=1))
        modules.append(nn.GroupNorm(min(8, max(1, self.pinn_filter[0]//4)), self.pinn_filter[0]))
        modules.append(nn.GELU())
        
        for i in range(1, self.num_pinn_filter):
            modules.append(ConvResBlock(self.pinn_filter[i-1], self.pinn_filter[i-1]))
            modules.append(ConvBlock(self.pinn_filter[i-1], self.pinn_filter[i]))

        modules.append(nn.Flatten())
        modules.append(nn.LazyLinear(self.latent_dim*self.size2*8))
        modules.append(nn.GroupNorm(min(8, max(1, (self.latent_dim*self.size2*8)//4)), self.latent_dim*self.size2*8))
        modules.append(nn.GELU())
        modules.append(nn.Dropout(self.dropout_rate))  # Keep dropout only for FC layers
        modules.append(nn.Linear(self.latent_dim*self.size2*8, self.latent_dim*self.size2))
        modules.append(nn.Unflatten(1, (self.size2, self.latent_dim)))
        modules.append(nn.Tanh())

        self.xs_out = nn.Sequential(*modules)


    def forward(self, x):
        im_size = 128
        x = x.reshape(-1, 1, im_size, im_size)
        latent_out = self.latent_out(x)
        xs_out = self.xs_out(x)

        return latent_out, xs_out




import time
from modules.common import initialize_weights_He
from torchvision.transforms import v2
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
import pytorch_warmup as warmup

def train_pinn(pinn_epoch, pinn_dataloader, pinn_validation_dataloader, pinn, pinn_lr, weight_decay=1e-4):
    im_size = 128

    writer = SummaryWriter(log_dir = './PINNruns', comment = 'PINN')

    loss=0
    
    # Robust CUDA initialization with error handling
    try:
        # Reset GPU state and clear memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            
            # Test if CUDA is actually working with a small tensor
            print("Testing CUDA availability...")
            test_tensor = torch.zeros(1).cuda()
            del test_tensor  # Free memory
            
            device = "cuda:0"
            print(f"CUDA initialized successfully. Using device: {device}")
            print(f"GPU: {torch.cuda.get_device_name(0)}")
            print(f"Available memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            device = "cpu"
            print("CUDA not available. Using CPU.")
    except RuntimeError as e:
        print(f"CUDA initialization failed with error: {e}")
        print("Falling back to CPU.")
        device = "cpu"

    # Reduce learning rate and add weight decay for regularization
    pinn_optimized = torch.optim.AdamW(pinn.parameters(), lr=pinn_lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(pinn_optimized, T_max = pinn_epoch, eta_min = pinn_lr * 0.01)

    # Data augmentation transforms
    augmentation = transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomRotation(10),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.ColorJitter(brightness=0.1, contrast=0.1),
        transforms.ToTensor(),
    ])

    from torchinfo import summary

    # Safely run model summary
    try:
        summary(pinn, (64,1,im_size,im_size))
    except Exception as e:
        print(f"Warning: Could not generate model summary: {e}")
    
    # Safely move model to device
    try:
        pinn = pinn.to(device)
        print(f"Model successfully moved to {device}")
    except RuntimeError as e:
        print(f"Error moving model to {device}: {e}")
        if device != "cpu":
            print("Attempting to fall back to CPU...")
            device = "cpu"
            pinn = pinn.to(device)

    pinn.apply(initialize_weights_He)

    for epoch in range(pinn_epoch):
        start_time = time.time()
        pinn.train(True)
        epoch_loss = 0
        num_batches = 0
        
        for i, (x, y1, y2) in enumerate(pinn_dataloader):
            # Apply data augmentation randomly to some samples
            if torch.rand(1) < 0.3:  # 30% chance of augmentation
                x_aug = []
                for img in x:
                    img_np = (img.squeeze().cpu().numpy() * 255).astype(np.uint8)
                    img_aug = augmentation(img_np)
                    x_aug.append(img_aug.unsqueeze(0))
                x = torch.cat(x_aug, dim=0)
            
            # Safely move tensors to device
            try:
                x, y1, y2 = x.to(device), y1.to(device), y2.to(device)
            except RuntimeError as e:
                print(f"Error moving data to {device}: {e}")
                if device != "cpu":
                    print("Falling back to CPU...")
                    device = "cpu"
                    pinn = pinn.to(device)
                    x, y1, y2 = x.to(device), y1.to(device), y2.to(device)
            
            for param in pinn.parameters():
                param.grad = None

            # Forward pass with error handling
            try:
                y_pred1, y_pred2 = pinn(x)
            except RuntimeError as e:
                print(f"Forward pass error: {e}")
                print(f"Input shape: {x.shape}, Device: {x.device}")
                # Try to recover by reducing batch size
                if len(x) > 1:
                    print("Trying with reduced batch size...")
                    half_size = len(x) // 2
                    x_half, y1_half, y2_half = x[:half_size], y1[:half_size], y2[:half_size]
                    y_pred1, y_pred2 = pinn(x_half)
                    x, y1, y2 = x_half, y1_half, y2_half
                else:
                    print("Cannot reduce batch size further, skipping batch")
                    continue

            A = nn.MSELoss()(y_pred1, y1)
            B = nn.MSELoss()(y_pred2, y2)

            loss = A + B
            epoch_loss += loss.item()
            num_batches += 1

            # Backward pass with error handling
            try:
                loss.backward()
                # Gradient clipping to prevent exploding gradients
                torch.nn.utils.clip_grad_norm_(pinn.parameters(), max_norm=1.0)
                pinn_optimized.step()
            except RuntimeError as e:
                print(f"Backward pass error: {e}")
                # Skip this batch and continue
                for param in pinn.parameters():
                    param.grad = None
                continue
        
        avg_train_loss = epoch_loss / num_batches

        # Validation loop
        pinn.eval()
        val_loss = 0
        val_batches = 0
        
        with torch.no_grad():
            for i, (x_val, y1_val, y2_val) in enumerate(pinn_validation_dataloader):
                x_val, y1_val, y2_val = x_val.to(device), y1_val.to(device), y2_val.to(device)
                
                y_pred1_val, y_pred2_val = pinn(x_val)

                A_val = nn.MSELoss()(y_pred1_val, y1_val)
                B_val = nn.MSELoss()(y_pred2_val, y2_val)

                val_loss += (A_val + B_val).item()
                val_batches += 1

        avg_val_loss = val_loss / val_batches

        # Learning rate scheduling
        scheduler.step()

        end_time = time.time()
        epoch_duration = end_time - start_time

        if epoch % 10 == 0:
            writer.add_scalar('PINN Loss/train', avg_train_loss, epoch)
            writer.add_scalar('PINN Loss/val', avg_val_loss, epoch)
            writer.add_scalar('Learning Rate', pinn_optimized.param_groups[0]['lr'], epoch)

        print('[%d/%d]\tTrain_Loss: %.4E, Val_Loss: %.4E, LR: %.2E, ETA: %.2f h' % 
              (epoch, pinn_epoch, avg_train_loss, avg_val_loss, 
               pinn_optimized.param_groups[0]['lr'], 
               (pinn_epoch-epoch)*epoch_duration/3600))

        # Save regular checkpoint
        if epoch % 50 == 0:
            torch.save(pinn.state_dict(), f'checkpoints/pinn_epoch_{epoch}.pth')

    torch.save(pinn.state_dict(), 'checkpoints/pinn.pth')
    torch.save(pinn, 'model_save/PINN')

    return avg_val_loss

def get_overfitting_reduction_suggestions():
    """
    Returns a dictionary of hyperparameter suggestions to reduce overfitting in PINN_img training.
    
    Usage example:
    suggestions = get_overfitting_reduction_suggestions()
    print("Recommended hyperparameters:")
    for key, value in suggestions.items():
        print(f"{key}: {value}")
    """
    return {
        'learning_rate': 1e-4,  # Reduced from typical 1e-3
        'weight_decay': 1e-4,   # L2 regularization
        'dropout_rate': 0.3,    # For fully connected layers only
        'batch_size': 32,       # Smaller batch size can help generalization
        'max_epochs': 500,      # Adjust based on your training needs
        'gradient_clip_norm': 1.0,  # Gradient clipping
        'augmentation_probability': 0.3,  # Data augmentation chance
        'scheduler_min_lr_ratio': 0.01,  # Minimum LR as ratio of initial LR
        'validation_frequency': 1,  # Validate every epoch
        'checkpoint_frequency': 50,  # Save checkpoint every 50 epochs
        
        # Additional architectural suggestions
        'reduce_model_complexity': 'Consider reducing pinn_filter sizes by 20-30%',
        'batch_normalization': 'BatchNorm for conv layers, dropout for FC layers only',
        'residual_connection_weight': 0.05,  # Reduced from 0.1
        'alternative_optimizers': ['AdamW with weight_decay', 'SGD with momentum=0.9'],
        
        # Training strategy suggestions
        'training_tips': [
            'Monitor train/val loss gap - manually stop when gap > 10x',
            'Use learning rate scheduling (CosineAnnealingLR)',
            'Apply data augmentation sparingly (30% of samples)',
            'Use gradient clipping to prevent exploding gradients',
            'Monitor training progress and save checkpoints regularly',
            'BatchNorm for conv layers, dropout only for fully connected layers',
            'Consider reducing epochs if overfitting persists'
        ]
    }

def example_usage_with_overfitting_reduction():
    """
    Example of how to use the improved PINN_img with overfitting reduction techniques.
    
    This function demonstrates the recommended way to initialize and train the model
    with the new anti-overfitting features.
    """
    print("Example: Using PINN_img with overfitting reduction")
    print("=" * 50)
    
    # Get recommended hyperparameters
    suggestions = get_overfitting_reduction_suggestions()
    
    # Example model initialization
    pinn_filter = [32, 64, 128]  # Reduced from potentially larger values
    latent_dim_end = 64
    input_shape = 128*128
    latent_dim = 32
    size2 = 10
    pinn_data_shape = (128, 128)
    
    # Initialize model with higher dropout rate
    model = PINN_img(
        pinn_filter=pinn_filter,
        latent_dim_end=latent_dim_end,
        input_shape=input_shape,
        latent_dim=latent_dim,
        size2=size2,
        pinn_data_shape=pinn_data_shape,
        dropout_rate=suggestions['dropout_rate']
    )
    
    print(f"Model initialized with dropout_rate: {suggestions['dropout_rate']}")
    print(f"Recommended learning rate: {suggestions['learning_rate']}")
    print(f"Recommended weight decay: {suggestions['weight_decay']}")
    print(f"Recommended max epochs: {suggestions['max_epochs']}")
    
    # Example training call (uncomment when ready to train)
    # loss = train_pinn(
    #     pinn_epoch=suggestions['max_epochs'],
    #     pinn_dataloader=your_dataloader,
    #     pinn_validation_dataloader=your_validation_dataloader,
    #     pinn=model,
    #     pinn_lr=suggestions['learning_rate'],
    #     weight_decay=suggestions['weight_decay']
    # )
    
    print("\nTraining tips:")
    for tip in suggestions['training_tips']:
        print(f"• {tip}")
    
    return model, suggestions