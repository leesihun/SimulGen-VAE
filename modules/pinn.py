import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import os
import numpy as np
from PIL import Image
import glob
import pandas as pd

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

# Original PINN class for CSV data
class PINN(nn.Module):
    def __init__(self, num_filters, latent_dim_end, input_shape, latent_dim, size2, dropout_rate=0.3):
        super(PINN, self).__init__()
        
        # Use safe CUDA initialization
        self.device = safe_cuda_initialization()
        
        self.latent_dim = latent_dim
        self.latent_dim_end = latent_dim_end
        self.size2 = size2
        
        # Main layers
        layers = []
        in_channels = input_shape
        
        for f in num_filters:
            layers.append(nn.Linear(in_channels, f))
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.Dropout(p=dropout_rate))
            in_channels = f
            
        self.main = nn.Sequential(*layers)
        
        # Output layers for latent vectors
        self.z_layer = nn.Linear(num_filters[-1], latent_dim_end)
        
        # Output layers for hierarchical latent vectors
        self.xs_layers = nn.ModuleList([nn.Linear(num_filters[-1], latent_dim) for _ in range(size2)])

    def forward(self, x):
        x = x.to(self.device)  # Use self.device instead of hardcoded device
        
        try:
            x = self.main(x)
            z = self.z_layer(x)
            
            # Generate hierarchical latent vectors
            xs = torch.zeros(x.size(0), self.size2, self.latent_dim).to(self.device)  # Use self.device
            
            for i in range(self.size2):
                xs[:, i, :] = self.xs_layers[i](x)
                
            return z, xs
        except RuntimeError as e:
            if "CUDA" in str(e):
                print(f"CUDA error during forward pass: {e}")
                print("Temporarily moving computation to CPU")
                # Move model to CPU and retry
                self.to("cpu")
                self.device = "cpu"
                x = x.cpu()
                
                # Retry on CPU
                x = self.main(x)
                z = self.z_layer(x)
                xs = torch.zeros(x.size(0), self.size2, self.latent_dim)
                
                for i in range(self.size2):
                    xs[:, i, :] = self.xs_layers[i](x)
                    
                return z, xs
            else:
                # Re-raise other errors
                raise e

# Original PINN class for image data
class PINN_img(nn.Module):
    def __init__(self, num_filters, latent_dim_end, input_shape, latent_dim, size2, img_shape, dropout_rate=0.3):
        super(PINN_img, self).__init__()
        
        # Use safe CUDA initialization
        self.device = safe_cuda_initialization()
        
        self.latent_dim = latent_dim
        self.latent_dim_end = latent_dim_end
        self.size2 = size2
        
        # Image processing layers
        self.img_shape = img_shape  # (C, H, W)
        self.img_features = 64  # Number of features to extract from image
        
        # For image input
        self.conv_layers = nn.Sequential(
            nn.Conv2d(img_shape[0], 16, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.LeakyReLU(0.2)
        )
        
        # Calculate flattened size after convolutions
        # This is an approximation, adjust based on your image dimensions
        conv_output_size = self._get_conv_output_size(img_shape)
        
        # Main layers for combined features
        layers = []
        in_channels = conv_output_size + input_shape  # Combined image features + numerical inputs
        
        for f in num_filters:
            layers.append(nn.Linear(in_channels, f))
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.Dropout(p=dropout_rate))
            in_channels = f
            
        self.main = nn.Sequential(*layers)
        
        # Output layers for latent vectors
        self.z_layer = nn.Linear(num_filters[-1], latent_dim_end)
        
        # Output layers for hierarchical latent vectors
        self.xs_layers = nn.ModuleList([nn.Linear(num_filters[-1], latent_dim) for _ in range(size2)])
        
    def _get_conv_output_size(self, shape):
        # Helper function to calculate the output size of the convolutions
        batch_size = 1
        input_tensor = torch.zeros(batch_size, *shape)
        
        try:
            input_tensor = input_tensor.to(self.device)
            output_tensor = self.conv_layers(input_tensor)
            return int(np.prod(output_tensor.size()[1:]))
        except RuntimeError as e:
            print(f"Error calculating conv output size: {e}")
            print("Performing calculation on CPU")
            input_tensor = input_tensor.cpu()
            self.conv_layers = self.conv_layers.cpu()
            output_tensor = self.conv_layers(input_tensor)
            return int(np.prod(output_tensor.size()[1:]))

    def forward(self, x):
        try:
            # Separate numerical inputs and image
            num_inputs = x[:, :, 0].to(self.device)
            img_inputs = x[:, :, 1:].to(self.device)
            
            # Process image data - reshape to proper dimensions
            batch_size = img_inputs.size(0)
            img_inputs = img_inputs.view(batch_size, self.img_shape[0], self.img_shape[1], self.img_shape[2])
            
            # Run through conv layers
            img_features = self.conv_layers(img_inputs)
            img_features = img_features.view(batch_size, -1)  # Flatten
            
            # Combine features
            combined = torch.cat((num_inputs, img_features), dim=1)
            
            # Process through main network
            x = self.main(combined)
            z = self.z_layer(x)
            
            # Generate hierarchical latent vectors
            xs = torch.zeros(batch_size, self.size2, self.latent_dim).to(self.device)
            
            for i in range(self.size2):
                xs[:, i, :] = self.xs_layers[i](x)
                
            return z, xs
            
        except RuntimeError as e:
            if "CUDA" in str(e):
                print(f"CUDA error during forward pass: {e}")
                print("Temporarily moving computation to CPU")
                # Move model and data to CPU
                self.to("cpu")
                self.device = "cpu"
                
                # Separate numerical inputs and image
                num_inputs = x[:, :, 0].cpu()
                img_inputs = x[:, :, 1:].cpu()
                
                # Process image data - reshape to proper dimensions
                batch_size = img_inputs.size(0)
                img_inputs = img_inputs.view(batch_size, self.img_shape[0], self.img_shape[1], self.img_shape[2])
                
                # Run through conv layers
                img_features = self.conv_layers(img_inputs)
                img_features = img_features.view(batch_size, -1)  # Flatten
                
                # Combine features
                combined = torch.cat((num_inputs, img_features), dim=1)
                
                # Process through main network
                x = self.main(combined)
                z = self.z_layer(x)
                
                # Generate hierarchical latent vectors
                xs = torch.zeros(batch_size, self.size2, self.latent_dim)
                
                for i in range(self.size2):
                    xs[:, i, :] = self.xs_layers[i](x)
                    
                return z, xs
            else:
                # Re-raise other errors
                raise e

# Enhanced PINN training function with error handling
def train_pinn(n_epochs, dataloader, val_dataloader, model, learning_rate, weight_decay=1e-4):
    # Safe device selection
    device = safe_cuda_initialization()
    model = model.to(device)
    
    writer = SummaryWriter(log_dir='PINNruns')
    criterion1 = nn.MSELoss()
    criterion2 = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    loss_save = []
    
    print(f"Starting PINN training on {device}")
    print(f"Training for {n_epochs} epochs with batch size {dataloader.batch_size}")
    
    try:
        for epoch in range(n_epochs):
            model.train()
            running_loss = 0.0
            
            for i, (inputs, targets1, targets2) in enumerate(dataloader):
                try:
                    # Move data to the model's device
                    inputs = inputs.to(device)
                    targets1 = targets1.to(device)
                    targets2 = targets2.to(device)
                    
                    # Zero the parameter gradients
                    optimizer.zero_grad()
                    
                    # Forward pass
                    outputs1, outputs2 = model(inputs)
                    
                    # Calculate loss
                    loss1 = criterion1(outputs1, targets1)
                    loss2 = criterion2(outputs2, targets2)
                    loss = loss1 + loss2
                    
                    # Backward pass and optimize
                    loss.backward()
                    optimizer.step()
                    
                    running_loss += loss.item()
                    
                except RuntimeError as e:
                    if "CUDA" in str(e):
                        print(f"CUDA error in training batch {i}: {e}")
                        print("Moving batch to CPU and continuing")
                        # Try to recover by moving to CPU for this batch
                        device = "cpu"
                        model.to(device)
                        model.device = device  # Update model's device attribute
                        
                        # Re-run the batch on CPU
                        inputs = inputs.to(device)
                        targets1 = targets1.to(device)
                        targets2 = targets2.to(device)
                        
                        optimizer.zero_grad()
                        outputs1, outputs2 = model(inputs)
                        loss1 = criterion1(outputs1, targets1)
                        loss2 = criterion2(outputs2, targets2)
                        loss = loss1 + loss2
                        loss.backward()
                        optimizer.step()
                        
                        running_loss += loss.item()
                    else:
                        raise e
            
            # Validation
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for inputs, targets1, targets2 in val_dataloader:
                    try:
                        inputs = inputs.to(device)
                        targets1 = targets1.to(device)
                        targets2 = targets2.to(device)
                        
                        outputs1, outputs2 = model(inputs)
                        
                        loss1 = criterion1(outputs1, targets1)
                        loss2 = criterion2(outputs2, targets2)
                        val_loss += (loss1 + loss2).item()
                    except RuntimeError as e:
                        if "CUDA" in str(e):
                            print(f"CUDA error during validation: {e}")
                            print("Moving validation to CPU")
                            # Move to CPU
                            device = "cpu"
                            model.to(device)
                            model.device = device  # Update model's device attribute
                            
                            # Re-run on CPU
                            inputs = inputs.to(device)
                            targets1 = targets1.to(device)
                            targets2 = targets2.to(device)
                            
                            outputs1, outputs2 = model(inputs)
                            loss1 = criterion1(outputs1, targets1)
                            loss2 = criterion2(outputs2, targets2)
                            val_loss += (loss1 + loss2).item()
                        else:
                            raise e
            
            # Logging
            train_loss = running_loss / len(dataloader)
            val_loss = val_loss / len(val_dataloader)
            writer.add_scalar('Loss/train', train_loss, epoch)
            writer.add_scalar('Loss/validation', val_loss, epoch)
            
            print(f'Epoch {epoch+1}/{n_epochs}, Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}')
            loss_save.append(train_loss)
            
            # Save checkpoint every 20 epochs
            if (epoch + 1) % 20 == 0:
                if not os.path.exists('./model_save'):
                    os.makedirs('./model_save')
                torch.save(model, f'./model_save/pinn_{epoch+1}')
        
        # Save final model
        torch.save(model, './model_save/pinn')
        print('PINN training finished')
        
        return loss_save
        
    except Exception as e:
        print(f"Error during PINN training: {e}")
        # Save model even if training fails
        try:
            torch.save(model, './model_save/pinn_emergency_save')
            print("Emergency model save completed")
        except:
            print("Could not save emergency model")
        return loss_save

def read_pinn_dataset_img(dir_path, param_data_type='jpg'):
    """Read image data for PINN training with error handling."""
    try:
        # Get all image files in the directory
        print('reading data from: ', dir_path)
        print(os.path.join(dir_path, '*.png'))
        # Why is there no png files?
        # there are bridge_DOE_99.png, etc...

        # Get all files in the directory
        all_files = glob.glob(os.path.join(dir_path, '*'))
        print(all_files)
        # Why are there no png files?
        # there are bridge_DOE_99.png, etc...
        # but why are there no png files?
        # there are bridge_DOE_99.png, etc...

        print(glob.glob(os.path.join(dir_path, '*.png')))

        if param_data_type == '.jpg':
            image_files = sorted(glob.glob(os.path.join(dir_path, '*.jpg')))
        elif param_data_type == '.png':
            image_files = sorted(glob.glob(os.path.join(dir_path, '*.png')))
        else:
            raise ValueError(f"Unsupported image format: {param_data_type}")
        
        if not image_files:
            raise FileNotFoundError(f"No {param_data_type} files found in {dir_path}")
        
        # Open as gray scale image.
        sample_img = Image.open(image_files[0]).convert('L')
        img_array = np.array(sample_img)
        height, width = img_array.shape
        channels = 1
        
        # Create dataset
        num_samples = len(image_files)
        
        # For storing image data (normalized)
        data = np.zeros((num_samples, 1, 1 + height*width*channels))
        
        print(f"Loading {num_samples} images with shape ({channels}, {height}, {width})")
        
        # Process all images
        for i, file_path in enumerate(image_files):
            print(f"Processing image {i+1}/{num_samples}: {file_path}")
            try:
                # Extract parameter from filename (assuming filename format includes parameter)
                param = float(os.path.basename(file_path).split('_')[0])
                
                # Load and process image
                img = Image.open(file_path)
                if channels == 1:
                    img = img.convert('L')  # Convert to grayscale if needed
                else:
                    img = img.convert('RGB')  # Convert to RGB
                
                # Normalize to [0,1]
                img_array = np.array(img) / 255.0
                
                # Flatten image
                if channels == 1:
                    img_flat = img_array.flatten()
                else:
                    img_flat = img_array.reshape(-1)
                
                # Store parameter and image
                data[i, 0, 0] = param  # Parameter value
                data[i, 0, 1:] = img_flat  # Flattened image
                
            except Exception as e:
                print(f"Error processing image {file_path}: {e}")
                # Use zeros as fallback
                data[i, 0, 0] = 0.0
                data[i, 0, 1:] = np.zeros(height*width*channels)
        
        # Return the dataset and image shape for reconstruction
        return data, (channels, height, width)
    
    except Exception as e:
        print(f" dataset: {e}")
        # Return minimal dummy dataset
        return np.zeros((1, 1, 2)), (1, 1, 1)

def read_pinn_dataset(dir_path, param_data_type='csv'):
    """Read CSV data for PINN training with error handling."""
    try:
        # Get all CSV files in the directory
        if param_data_type == 'csv':
            csv_files = sorted(glob.glob(os.path.join(dir_path, '*.csv')))
        else:
            raise ValueError(f"Unsupported file format: {param_data_type}")
        
        if not csv_files:
            raise FileNotFoundError(f"No {param_data_type} files found in {dir_path}")
        
        num_samples = len(csv_files)
        all_data = []
        
        for i, file_path in enumerate(csv_files):
            try:
                # Read CSV file
                df = pd.read_csv(file_path, header=None)
                
                # Extract parameter data (assuming first row is the parameter)
                param_data = df.values.flatten()
                
                all_data.append(param_data)
            except Exception as e:
                print(f"Error reading file {file_path}: {e}")
                # Use zeros as fallback
                if all_data:
                    # Use same length as previous data
                    all_data.append(np.zeros_like(all_data[0]))
                else:
                    # Just use a single zero
                    all_data.append(np.array([0.0]))
        
        # Convert to numpy array
        data = np.array(all_data)
        
        return data
    
    except Exception as e:
        print(f"Error reading PINN dataset: {e}")
        # Return minimal dummy dataset
        return np.zeros((1, 1))

def pinn_scaler(data, save_path=None):
    """Scale PINN data with improved error handling."""
    from sklearn.preprocessing import StandardScaler
    import pickle
    
    try:
        # Check for NaNs
        if np.isnan(data).any():
            print("Warning: NaN values detected in data, replacing with zeros")
            data = np.nan_to_num(data, nan=0.0)
        
        # Reshape data for scaler
        original_shape = data.shape
        data_reshaped = data.reshape(original_shape[0], -1)
        
        # Apply scaling
        scaler = StandardScaler()
        data_scaled = scaler.fit_transform(data_reshaped)
        
        # Reshape back to original dimensions
        data_scaled = data_scaled.reshape(original_shape)
        
        # Save scaler for later use
        if save_path:
            with open(save_path, 'wb') as f:
                pickle.dump(scaler, f)
        
        return data_scaled, scaler
        
    except Exception as e:
        print(f"Error during PINN scaling: {e}")
        # Return original data and a dummy scaler that doesn't transform
        class DummyScaler:
            def transform(self, X):
                return X
            def inverse_transform(self, X):
                return X
                
        return data, DummyScaler()

def pinn_scaler_input(data, load_path):
    """Apply existing scaler to PINN input data."""
    import pickle
    
    try:
        # Load scaler
        with open(load_path, 'rb') as f:
            scaler = pickle.load(f)
        
        # Check for NaNs
        if np.isnan(data).any():
            print("Warning: NaN values detected in input data, replacing with zeros")
            data = np.nan_to_num(data, nan=0.0)
        
        # Reshape data for scaler
        original_shape = data.shape
        data_reshaped = data.reshape(data.shape[0], -1)
        
        # Apply scaling
        data_scaled = scaler.transform(data_reshaped)
        
        # Reshape back to original dimensions
        data_scaled = data_scaled.reshape(original_shape)
        
        return data_scaled
        
    except Exception as e:
        print(f"Error applying PINN scaler: {e}")
        # Return original data if scaling fails
        return data