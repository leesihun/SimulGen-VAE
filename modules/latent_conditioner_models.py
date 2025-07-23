import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import math
import time
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

# Data reading functions moved to latent_conditioner.py to avoid duplication

# ============================================================================
# CNN-BASED LATENT CONDITIONERS (moved from latent_conditioner.py)
# ============================================================================

class LatentConditioner(nn.Module):
    """MLP-based latent conditioner for parametric data"""
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
            modules.append(nn.Dropout(0.1))
        self.latent_conditioner = nn.Sequential(*modules)

        # Simplified output heads
        final_feature_size = self.latent_conditioner_filter[-2]
        
        # ULTRA-EXTREME bottleneck with single output heads
        hidden_size = max(8, final_feature_size // 32)
        
        # Single prediction head for latent output
        self.latent_out = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(final_feature_size, hidden_size),
            nn.GELU(),
            nn.Dropout(0.15), 
            nn.Linear(hidden_size, self.latent_dim_end),
            nn.Tanh()
        )
        
        # Single prediction head for xs output
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
        
        # Direct prediction from single heads
        latent_out = self.latent_out(features)
        xs_out = self.xs_out(features)
        xs_out = xs_out.unflatten(1, (self.size2, self.latent_dim))

        return latent_out, xs_out


class ImprovedConvResBlock(nn.Module):
    """Improved convolutional residual block"""
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
    """Simple convolutional block"""
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
    """CNN-based latent conditioner for image data"""
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
        
        # ULTRA-SIMPLE feature extraction - replace ResBlocks with basic conv to prevent overfitting
        for i in range(1, min(3, self.num_latent_conditioner_filter)):  # Limit to max 3 layers
            stride = 2 if i < self.num_latent_conditioner_filter - 1 else 1
            # Replace complex ResBlocks with simple ConvBlocks
            block = ConvBlock(self.latent_conditioner_filter[i-1], self.latent_conditioner_filter[i])
            self.backbone.append(block)
        
        # Adaptive pooling and feature size calculation
        self.adaptive_pool = nn.AdaptiveAvgPool2d((4, 4))
        final_feature_size = self.latent_conditioner_filter[-1] * 16  # 4*4
        
        # ULTRA-EXTREME bottleneck with single output heads
        hidden_size = max(8, final_feature_size // 32)
        
        # Single prediction head for latent output
        self.latent_out = nn.Sequential(
            nn.Dropout(0.2),
            nn.Linear(final_feature_size, hidden_size),
            nn.GELU(),
            nn.Dropout(0.15), 
            nn.Linear(hidden_size, self.latent_dim_end),
            nn.Tanh()
        )
        
        # Single prediction head for xs output
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
        
        # Direct prediction from single heads
        latent_out = self.latent_out(features)
        xs_out = self.xs_out(features)
        xs_out = xs_out.unflatten(1, (self.size2, self.latent_dim))

        return latent_out, xs_out

# ============================================================================
# VIT-BASED LATENT CONDITIONERS (new implementation)
# ============================================================================


class PatchEmbedding(nn.Module):
    """Convert image to patches and embed them"""
    def __init__(self, img_size=128, patch_size=16, in_channels=1, embed_dim=64, dropout=0.3):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.patch_dim = in_channels * patch_size * patch_size
        
        # Ultra-simple patch projection to prevent overfitting
        self.projection = nn.Sequential(
            nn.Linear(self.patch_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
        # Learnable position embeddings
        self.position_embeddings = nn.Parameter(torch.randn(1, self.num_patches, embed_dim) * 0.02)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, C, H, W = x.shape
        
        # Convert to patches: (B, C, H, W) -> (B, num_patches, patch_dim)
        x = x.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        x = x.contiguous().view(B, C, -1, self.patch_size, self.patch_size)
        x = x.permute(0, 2, 1, 3, 4).contiguous()
        x = x.view(B, -1, self.patch_dim)
        
        # Project patches to embedding dimension
        x = self.projection(x)
        
        # Add position embeddings
        x = x + self.position_embeddings
        x = self.dropout(x)
        
        return x


class MultiHeadSelfAttention(nn.Module):
    """Self-attention with extreme regularization"""
    def __init__(self, embed_dim=64, num_heads=4, dropout=0.4, attention_dropout=0.3):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert embed_dim % num_heads == 0
        
        # Single linear layer for Q, K, V to reduce parameters
        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=False)
        self.attention_dropout = nn.Dropout(attention_dropout)
        self.projection = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
        # Learnable temperature for attention sharpening
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, x):
        B, N, C = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Scaled dot-product attention with temperature
        attention_scores = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5) / self.temperature.clamp(min=0.1)
        attention_weights = F.softmax(attention_scores, dim=-1)
        attention_weights = self.attention_dropout(attention_weights)
        
        # Apply attention to values
        out = (attention_weights @ v).transpose(1, 2).reshape(B, N, C)
        
        # Final projection
        out = self.projection(out)
        out = self.dropout(out)
        
        return out, attention_weights


class TransformerBlock(nn.Module):
    """Minimal transformer block with heavy regularization"""
    def __init__(self, embed_dim=64, num_heads=4, mlp_ratio=2, dropout=0.4):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attention = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        
        # Tiny MLP to prevent overfitting
        mlp_hidden = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden, embed_dim),
            nn.Dropout(dropout)
        )
        
        # Stochastic depth for regularization
        self.drop_path_prob = dropout * 0.5

    def forward(self, x):
        # Self-attention with residual connection
        normed_x = self.norm1(x)
        attn_out, attention_weights = self.attention(normed_x)
        
        # Stochastic depth
        if self.training and torch.rand(1) < self.drop_path_prob:
            x = x  # Skip attention
        else:
            x = x + attn_out
        
        # MLP with residual connection
        normed_x = self.norm2(x)
        mlp_out = self.mlp(normed_x)
        
        # Stochastic depth for MLP
        if self.training and torch.rand(1) < self.drop_path_prob:
            pass  # Skip MLP
        else:
            x = x + mlp_out
            
        return x, attention_weights


class TinyViTLatentConditioner(nn.Module):
    """Ultra-minimal ViT for latent conditioning with extreme anti-overfitting"""
    def __init__(self, latent_dim_end, latent_dim, size2, 
                 img_size=128, patch_size=16, embed_dim=64, num_layers=2, num_heads=4, 
                 mlp_ratio=2, dropout=0.5):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.latent_dim_end = latent_dim_end
        self.latent_dim = latent_dim
        self.size2 = size2
        
        # Patch embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, 1, embed_dim, dropout)
        
        # Minimal transformer layers
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Global average pooling instead of CLS token to reduce parameters
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # EXTREME bottleneck - even smaller than CNN version
        hidden_size = max(4, embed_dim // 8)  # Minimum 4 features
        
        # Single prediction head for latent output
        self.latent_out = nn.Sequential(
            nn.Dropout(0.6),
            nn.Linear(embed_dim, hidden_size),
            nn.GELU(), 
            nn.Dropout(0.5),
            nn.Linear(hidden_size, latent_dim_end),
            nn.Tanh()
        )
        
        # Single prediction head for xs output
        self.xs_out = nn.Sequential(
            nn.Dropout(0.6),
            nn.Linear(embed_dim, hidden_size),
            nn.GELU(),
            nn.Dropout(0.5), 
            nn.Linear(hidden_size, latent_dim * size2),
            nn.Tanh()
        )

    def forward(self, x):
        B = x.shape[0]
        
        # Reshape flattened input to image
        x = x.reshape(B, 1, self.img_size, self.img_size)
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)
        
        # Apply transformer blocks
        attention_maps = []
        for block in self.transformer_blocks:
            x, attn_weights = block(x)
            attention_maps.append(attn_weights)
        
        x = self.norm(x)
        
        # Global average pooling over patches
        x = x.transpose(1, 2)  # (B, embed_dim, num_patches)
        x = self.global_pool(x).squeeze(-1)  # (B, embed_dim)
        
        # Direct predictions from single heads
        latent_out = self.latent_out(x)
        xs_out = self.xs_out(x)
        xs_out = xs_out.unflatten(1, (self.size2, self.latent_dim))
        
        return latent_out, xs_out


# ViT is ONLY for image data - parametric data should use the original CNN latent conditioner
# or a simple MLP-based approach. This was a conceptual error.


def safe_cuda_initialization():
    """Safely check CUDA availability with error handling"""
    try:
        if torch.cuda.is_available():
            test_tensor = torch.zeros(1).cuda()
            del test_tensor
            print("✓ CUDA initialized successfully")
            return "cuda"
        else:
            print("CUDA not available, using CPU")
            return "cpu"
    except RuntimeError as e:
        print(f"⚠️ CUDA initialization error: {e}")
        print("Falling back to CPU")
        return "cpu"


def train_vit_latent_conditioner(latent_conditioner_epoch, latent_conditioner_dataloader, 
                                 latent_conditioner_validation_dataloader, latent_conditioner, 
                                 latent_conditioner_lr, weight_decay=5e-3, is_image_data=True):
    """Training function for ViT latent conditioner with extreme anti-overfitting"""
    
    writer = SummaryWriter(log_dir='./ViTLatentConditionerRuns', comment='ViTLatentConditioner')
    device = safe_cuda_initialization()
    
    # EXTREME weight decay and learning rate
    optimizer = torch.optim.AdamW(latent_conditioner.parameters(), 
                                  lr=latent_conditioner_lr * 0.5,  # Lower LR for ViT
                                  weight_decay=weight_decay,
                                  betas=(0.9, 0.95))  # ViT-optimized betas
    
    # Simple cosine annealing - no warmup complexity
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=latent_conditioner_epoch, eta_min=1e-7
    )
    
    # AGGRESSIVE early stopping
    best_val_loss = float('inf')
    patience = 15  # Very aggressive
    patience_counter = 0
    min_delta = 1e-5
    
    # Overfitting detection
    overfitting_threshold = 5.0  # Lower threshold than CNN
    
    # EXTREME data augmentation for images
    if is_image_data:
        augmentation = transforms.Compose([
            transforms.ToPILImage(),
            transforms.RandomRotation(15),  # More rotation
            transforms.RandomHorizontalFlip(p=0.6),
            transforms.RandomVerticalFlip(p=0.3),  # Add vertical flip
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            transforms.ToTensor(),
        ])
    
    latent_conditioner = latent_conditioner.to(device)
    
    # Simplified weight initialization
    def init_vit_weights(m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    latent_conditioner.apply(init_vit_weights)
    
    # EMA with faster decay for ViT
    ema_decay = 0.99
    ema_model = {name: param.clone() for name, param in latent_conditioner.named_parameters()}
    
    print(f"Starting ViT training with {sum(p.numel() for p in latent_conditioner.parameters()):,} parameters")
    
    for epoch in range(latent_conditioner_epoch):
        start_time = time.time()
        latent_conditioner.train()
        
        # EXTREME progressive dropout 
        current_dropout = max(0.1, 0.9 * (1 - epoch / latent_conditioner_epoch))  # 90% -> 10%
        for module in latent_conditioner.modules():
            if isinstance(module, nn.Dropout):
                module.p = current_dropout
        
        epoch_loss = 0
        epoch_loss_y1 = 0
        epoch_loss_y2 = 0
        num_batches = 0
        
        for i, (x, y1, y2) in enumerate(latent_conditioner_dataloader):
            if is_image_data and epoch == 0 and i == 0:
                x = x.reshape([x.shape[0], int(math.sqrt(x.shape[1])), int(math.sqrt(x.shape[1]))])
                print('ViT dataset_shape', x.shape, y1.shape, y2.shape)
            
            if is_image_data:
                x = x.reshape([x.shape[0], int(math.sqrt(x.shape[1])), int(math.sqrt(x.shape[1]))])
                
                # EXTREME augmentation for ViT
                # Patch dropout - randomly mask entire patches
                if torch.rand(1) < 0.8:  # 80% chance
                    patch_size = 16
                    img_size = x.shape[1]
                    num_patches_per_dim = img_size // patch_size
                    num_patches_to_mask = int(0.3 * num_patches_per_dim ** 2)  # Mask 30% of patches
                    
                    for b in range(x.shape[0]):
                        if torch.rand(1) < 0.7:  # 70% of samples
                            x_img = x[b].reshape(img_size, img_size)
                            for _ in range(num_patches_to_mask):
                                patch_i = torch.randint(0, num_patches_per_dim, (1,))
                                patch_j = torch.randint(0, num_patches_per_dim, (1,))
                                start_i, start_j = patch_i * patch_size, patch_j * patch_size
                                end_i, end_j = start_i + patch_size, start_j + patch_size
                                x_img[start_i:end_i, start_j:end_j] = 0
                            x[b] = x_img.flatten()
                
                # Token mixup - mix patches from different images
                if torch.rand(1) < 0.6 and x.size(0) > 1:  # 60% chance
                    alpha = 1.0  # More aggressive for ViT
                    lam = np.random.beta(alpha, alpha)
                    batch_size = x.size(0)
                    index = torch.randperm(batch_size)
                    
                    x = lam * x + (1 - lam) * x[index, :]
                    y1 = lam * y1 + (1 - lam) * y1[index, :]
                    y2 = lam * y2 + (1 - lam) * y2[index, :]
                
                # Strong noise injection
                if torch.rand(1) < 0.7:  # 70% chance
                    noise = torch.randn_like(x) * 0.08  # 8% noise
                    x = x + noise
            
            x, y1, y2 = x.to(device), y1.to(device), y2.to(device)
            
            optimizer.zero_grad(set_to_none=True)
            y_pred1, y_pred2 = latent_conditioner(x)
            
            # Simpler loss - no label smoothing for ViT initially
            A = F.mse_loss(y_pred1, y1)
            B = F.mse_loss(y_pred2, y2)
            
            # Ensemble diversity penalty removed - using single heads now
            loss = A + B
            
            # EMA update
            with torch.no_grad():
                for name, param in latent_conditioner.named_parameters():
                    ema_model[name] = ema_decay * ema_model[name] + (1 - ema_decay) * param
            
            epoch_loss += loss.item()
            epoch_loss_y1 += A.item()
            epoch_loss_y2 += B.item()
            num_batches += 1
            
            loss.backward()
            
            # Gradient clipping for ViT stability
            torch.nn.utils.clip_grad_norm_(latent_conditioner.parameters(), max_norm=1.0)
            optimizer.step()
        
        avg_train_loss = epoch_loss / num_batches
        avg_train_loss_y1 = epoch_loss_y1 / num_batches
        avg_train_loss_y2 = epoch_loss_y2 / num_batches
        
        # Validation
        latent_conditioner.eval()
        val_loss = 0
        val_loss_y1 = 0
        val_loss_y2 = 0
        val_batches = 0
        
        with torch.no_grad():
            for i, (x_val, y1_val, y2_val) in enumerate(latent_conditioner_validation_dataloader):
                if is_image_data:
                    x_val = x_val.reshape([x_val.shape[0], int(math.sqrt(x_val.shape[1])), int(math.sqrt(x_val.shape[1]))])
                
                x_val, y1_val, y2_val = x_val.to(device), y1_val.to(device), y2_val.to(device)
                y_pred1_val, y_pred2_val = latent_conditioner(x_val)
                
                A_val = F.mse_loss(y_pred1_val, y1_val)
                B_val = F.mse_loss(y_pred2_val, y2_val)
                
                val_loss += (A_val + B_val).item()
                val_loss_y1 += A_val.item()
                val_loss_y2 += B_val.item()
                val_batches += 1
        
        avg_val_loss = val_loss / val_batches
        avg_val_loss_y1 = val_loss_y1 / val_batches
        avg_val_loss_y2 = val_loss_y2 / val_batches
        
        # Overfitting check
        overfitting_ratio = avg_val_loss / max(avg_train_loss, 1e-8)
        if overfitting_ratio > overfitting_threshold:
            print(f'🚨 ViT overfitting detected! Val/Train ratio: {overfitting_ratio:.1f}')
            print(f'Stopping early at epoch {epoch}')
            break
        
        # Early stopping
        if avg_val_loss < best_val_loss - min_delta:
            best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save(latent_conditioner.state_dict(), 'checkpoints/vit_latent_conditioner_best.pth')
        else:
            patience_counter += 1
        
        scheduler.step()
        
        end_time = time.time()
        epoch_duration = end_time - start_time
        
        if epoch % 5 == 0:  # Less frequent logging
            writer.add_scalar('ViTLatentConditioner Loss/train', avg_train_loss, epoch)
            writer.add_scalar('ViTLatentConditioner Loss/val', avg_val_loss, epoch)
            writer.add_scalar('ViTLatentConditioner Loss/train_y1', avg_train_loss_y1, epoch)
            writer.add_scalar('ViTLatentConditioner Loss/train_y2', avg_train_loss_y2, epoch)
            writer.add_scalar('ViTLatentConditioner Loss/val_y1', avg_val_loss_y1, epoch)
            writer.add_scalar('ViTLatentConditioner Loss/val_y2', avg_val_loss_y2, epoch)
            writer.add_scalar('ViT Learning Rate', optimizer.param_groups[0]['lr'], epoch)
        
        current_lr = optimizer.param_groups[0]['lr']
        
        print('[%d/%d] ViT Train: %.4E (y1:%.4E, y2:%.4E), Val: %.4E (y1:%.4E, y2:%.4E), LR: %.2E, Dropout: %.1f%%, ETA: %.2f h, Patience: %d/%d' % 
              (epoch, latent_conditioner_epoch, avg_train_loss, avg_train_loss_y1, avg_train_loss_y2,
               avg_val_loss, avg_val_loss_y1, avg_val_loss_y2, current_lr, current_dropout*100,
               (latent_conditioner_epoch-epoch)*epoch_duration/3600, patience_counter, patience))
        
        if patience_counter >= patience:
            print(f'ViT early stopping at epoch {epoch}. Best validation loss: {best_val_loss:.4E}')
            break
        
        # Save checkpoint
        if epoch % 25 == 0:
            torch.save(latent_conditioner.state_dict(), f'checkpoints/vit_latent_conditioner_epoch_{epoch}.pth')
    
    torch.save(latent_conditioner.state_dict(), 'checkpoints/vit_latent_conditioner.pth')
    torch.save(latent_conditioner, 'model_save/ViTLatentConditioner')
    
    return avg_val_loss