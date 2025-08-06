import numpy as np
import warnings
import time
import psutil
import gc
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from pickle import dump
import torch

def reduce_dataset(data_save, num_time_to, num_node_red, num_param, num_time, num_node_red_start, num_node_red_end):

    start = time.time()

    num_node = data_save.shape[-1]
    if num_time_to == num_time and num_node_red == num_node:
        FOM_data = data_save
    else:
        FOM_data_temp = data_save
        num_time = num_time_to
        FOM_data = np.zeros((num_param, num_time, num_node_red))
        FOM_data[:, 0:num_time,:] = FOM_data_temp[:, 0:num_time, num_node_red_start:num_node_red_end]
        del FOM_data_temp

        num_node = num_node_red
        FOM_data_temp = np.zeros((num_param, num_time, num_node_red))
        FOM_data_temp[:,:,0:num_node] = FOM_data
        FOM_data_temp[:,:,num_node:-1] = 0
        del FOM_data

        FOM_data = np.zeros((num_param, num_time, num_node))
        FOM_data = FOM_data_temp

    end = time.time()
    print(f"Time taken: {end - start} seconds")
    print('FOM shape:   ', FOM_data.shape)

    return num_time, FOM_data, num_node

def data_augmentation(stretch, FOM_data, num_param, num_node):
    #### T.B.D. w. audiomentation, librosa
    # Not currently used at the moment

    if stretch  == 1:
        new_x_train = FOM_data

        augment = Compose([
            AddGaussianNoise(min_amplitude = 0.001, max_amplitude = 0.05, p=1),
            Resample(min_sample_rate = 1000, max_sample_rate = 15000, p=1),
            Shift(p=1),
        ])

        for i in range(num_param):
            X = FOM_data[i, :,:]
            X = augment(samples = X, sample_rate = 10000)
            new_x_train[i,:,0:num_node] = X[:,0:num_node]

        FOM_data_aug = np.append(FOM_data, new_x_train, axis=0)

    else:
        FOM_data_aug = FOM_data

    return FOM_data_aug

def get_memory_usage():
    """Get current memory usage in GB"""
    process = psutil.Process()
    return process.memory_info().rss / 1024**3

def data_scaler(FOM_data_aug, FOM_data, num_time, num_node, directory, chunk_size=None):
    """
    Optimized data scaler with chunked processing and memory optimization.
    
    Args:
        chunk_size: Number of samples to process at once (auto-calculated if None)
    """
    start = time.time()
    initial_memory = get_memory_usage()
    print(f"Initial memory usage: {initial_memory:.2f} GB")
    
    # Force float32 to halve memory usage - use GPU conversion if available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_gpu = False#torch.cuda.is_available()
    
    # Auto-calculate chunk size based on available memory if not specified
    if chunk_size is None:
        if use_gpu:
            # Use GPU memory for calculation when GPU is available
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            available_memory_gb = gpu_memory_gb * 0.6  # Use 60% of GPU memory conservatively
        else:
            available_memory_gb = psutil.virtual_memory().available / 1024**3
        
        # Use ~10% of available memory for chunk processing
        chunk_memory_gb = available_memory_gb * 0.1
        # Estimate memory per sample (float32 * num_node * safety_factor)
        memory_per_sample = num_node * 4 * 2  # bytes, with safety factor
        chunk_size = max(1000, int(chunk_memory_gb * 1024**3 / memory_per_sample))
        print(f"Auto-calculated chunk_size: {chunk_size} (based on {available_memory_gb:.1f}GB {'GPU' if use_gpu else 'CPU'} available)")
    
    if FOM_data_aug.dtype != np.float32:
        print(f"Converting to float32{'... (using GPU)' if use_gpu else '... (using CPU)'}...")
        
        if use_gpu:
            # GPU conversion - much faster for large arrays
            conversion_chunk_size = min(5000, FOM_data_aug.shape[0])  # Larger chunks for GPU
            for i in range(0, FOM_data_aug.shape[0], conversion_chunk_size):
                end_idx = min(i + conversion_chunk_size, FOM_data_aug.shape[0])
                # Convert chunk to GPU tensor, change dtype, then back to numpy
                chunk_tensor = torch.from_numpy(FOM_data_aug[i:end_idx]).to(device)
                chunk_tensor = chunk_tensor.float()  # Convert to float32 on GPU
                FOM_data_aug[i:end_idx] = chunk_tensor.cpu().numpy()
                del chunk_tensor  # Free GPU memory
                if i % (conversion_chunk_size * 4) == 0:  # Progress update every 4 chunks
                    print(f"  Converted {end_idx}/{FOM_data_aug.shape[0]} samples...")
            torch.cuda.empty_cache()  # Clear GPU cache
        else:
            # CPU fallback - chunked conversion to avoid memory overflow
            conversion_chunk_size = min(1000, FOM_data_aug.shape[0])
            for i in range(0, FOM_data_aug.shape[0], conversion_chunk_size):
                end_idx = min(i + conversion_chunk_size, FOM_data_aug.shape[0])
                FOM_data_aug[i:end_idx] = FOM_data_aug[i:end_idx].astype(np.float32)
                if i % (conversion_chunk_size * 10) == 0:
                    print(f"  Converted {end_idx}/{FOM_data_aug.shape[0]} samples...")
        gc.collect()  # Force garbage collection
        
    if FOM_data.dtype != np.float32:
        print(f"Converting FOM_data to float32{'... (using GPU)' if use_gpu else '... (using CPU)'}...")
        
        if use_gpu:
            conversion_chunk_size = min(5000, FOM_data.shape[0])
            for i in range(0, FOM_data.shape[0], conversion_chunk_size):
                end_idx = min(i + conversion_chunk_size, FOM_data.shape[0])
                chunk_tensor = torch.from_numpy(FOM_data[i:end_idx]).to(device)
                chunk_tensor = chunk_tensor.float()
                FOM_data[i:end_idx] = chunk_tensor.cpu().numpy()
                del chunk_tensor
                if i % (conversion_chunk_size * 4) == 0:
                    print(f"  Converted {end_idx}/{FOM_data.shape[0]} samples...")
            torch.cuda.empty_cache()
        else:
            conversion_chunk_size = min(1000, FOM_data.shape[0])
            for i in range(0, FOM_data.shape[0], conversion_chunk_size):
                end_idx = min(i + conversion_chunk_size, FOM_data.shape[0])
                FOM_data[i:end_idx] = FOM_data[i:end_idx].astype(np.float32)
                if i % (conversion_chunk_size * 10) == 0:
                    print(f"  Converted {end_idx}/{FOM_data.shape[0]} samples...")
        gc.collect()
    
    after_conversion_memory = get_memory_usage()
    print(f"Memory after float32 conversion: {after_conversion_memory:.2f} GB (saved: {initial_memory - after_conversion_memory:.2f} GB)")
    
    # Wider range for better VAE training - VAEs work better with [-1, 1] or [-0.9, 0.9]
    scaler = MinMaxScaler(feature_range=(-0.9, 0.9))
    
    # For MinMaxScaler, we need to fit on a representative sample since it doesn't support partial_fit
    print("Fitting scaler on representative sample...")
    total_samples = FOM_data_aug.shape[0] * FOM_data_aug.shape[1]
    
    # Use every nth sample to create a representative subset for fitting
    sample_stride = max(1, total_samples // (chunk_size * 2))  # Ensure we don't exceed memory
    sample_indices = np.arange(0, total_samples, sample_stride)
    
    # Convert indices to param/time coordinates
    param_indices = sample_indices // num_time
    time_indices = sample_indices % num_time
    
    # Extract representative samples efficiently
    representative_data = []
    for i in range(0, len(param_indices), chunk_size):
        end_i = min(i + chunk_size, len(param_indices))
        batch_params = param_indices[i:end_i]
        batch_times = time_indices[i:end_i]
        
        # Extract samples
        samples = FOM_data_aug[batch_params, batch_times, :]
        representative_data.append(samples)
    
    # Concatenate and fit
    representative_samples = np.vstack(representative_data)
    scaler.fit(representative_samples)
    del representative_data, representative_samples
    gc.collect()
    
    print(f"Scaler fitted on {len(sample_indices)} representative samples")
    
    # Transform data in-place to save memory
    print("Transforming training data in chunks...")
    FOM_data_aug_flat = FOM_data_aug.reshape(-1, num_node)
    
    # Process in chunks to avoid memory issues
    for start_idx in range(0, FOM_data_aug_flat.shape[0], chunk_size):
        end_idx = min(start_idx + chunk_size, FOM_data_aug_flat.shape[0])
        FOM_data_aug_flat[start_idx:end_idx] = scaler.transform(FOM_data_aug_flat[start_idx:end_idx])
        
        # Progress indicator for large datasets
        if start_idx % (chunk_size * 10) == 0:
            progress = (start_idx / FOM_data_aug_flat.shape[0]) * 100
            current_memory = get_memory_usage()
            print(f"Progress: {progress:.1f}% | Memory: {current_memory:.2f} GB")
    
    # Reshape back in-place
    new_x_train = FOM_data_aug_flat.reshape(FOM_data_aug.shape)
    DATA_shape = new_x_train.shape[1:]
    
    # Clean up intermediate variables
    del FOM_data_aug_flat
    gc.collect()
    
    final_memory = get_memory_usage()
    print(f"Peak memory usage: {final_memory:.2f} GB")
    
    dump(scaler, open('./model_save/scaler.pkl', 'wb'))
    end = time.time()
    print(f"Optimized scaling time: {end - start:.2f} seconds")
    print(f"Final data shape: {new_x_train.shape}, dtype: {new_x_train.dtype}")
    print(f"Memory efficiency: {((initial_memory - final_memory) / initial_memory * 100):.1f}% reduction")
    
    return new_x_train, DATA_shape, scaler

def latent_conditioner_scaler(data, name):
    # Also update LatentConditioner scaler to match
    scaler = MinMaxScaler(feature_range=(-0.9, 0.9))

    # Handle 3D arrays by reshaping to 2D for scaling
    original_shape = data.shape
    
    # Check for empty data
    if original_shape[0] == 0:
        raise ValueError(f"Empty data array detected with shape {original_shape}. "
                        "Please check your data loading configuration. "
                        "If using 'input_type image', ensure PNG files exist in the specified directory.")
    
    if len(original_shape) == 3:
        # Reshape to 2D for scaler
        data_reshaped = data.reshape(original_shape[0], -1)
        print(f"Reshaping 3D data from {original_shape} to {data_reshaped.shape} for scaling")
    else:
        data_reshaped = data

    scaler.fit(data_reshaped)
    scaled_data = scaler.transform(data_reshaped)

    # Reshape back to original dimensions
    if len(original_shape) == 3:
        scaled_data = scaled_data.reshape(original_shape)

    dump(scaler, open(name, 'wb'))

    return scaled_data, scaler

def latent_conditioner_scaler_input(data, name):

    scaler = StandardScaler()

    # Handle 3D arrays by reshaping to 2D for scaling
    original_shape = data.shape
    if len(original_shape) == 3:
        # Reshape to 2D for scaler
        data_reshaped = data.reshape(original_shape[0], -1)
        print(f"Reshaping 3D data from {original_shape} to {data_reshaped.shape} for scaling")
    else:
        data_reshaped = data

    scaler.fit(data_reshaped)
    scaled_data = scaler.transform(data_reshaped)

    # Reshape back to original dimensions
    if len(original_shape) == 3:
        scaled_data = scaled_data.reshape(original_shape)

    dump(scaler, open(name, 'wb'))

    return scaled_data, scaler