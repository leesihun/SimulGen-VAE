#!/usr/bin/env python3
"""SimulGenVAE v1.4.3 - Main Training Script

A high-performance Variational Autoencoder for fast generation and inference 
of transient/static simulation data with Physics-Aware Neural Network (PANN) integration.

This script provides the main entry point for training SimulGenVAE models with support
for both single-GPU and multi-GPU distributed training. It supports three different
latent conditioning architectures (MLP, CNN, Vision Transformer) and comprehensive
anti-overfitting measures.

Usage:
    Single GPU: python SimulGen-VAE.py --preset=1 --plot=2 --lc_only=0 --size=small --load_all=1
    Multi-GPU:  torchrun --nproc_per_node=4 SimulGen-VAE.py --use_ddp --preset=1 --plot=2 --lc_only=0 --size=small --load_all=1

Author: SiHun Lee, Ph.D.
Email: kevin1007kr@gmail.com
LinkedIn: https://www.linkedin.com/in/시훈-이-13009a172/?originalSubdomain=kr

New in v1.4.3:
- Argument optimization: --train_latent_conditioner_only → --lc_only (73% shorter)
- Modern DDP with automatic local_rank detection via torchrun
- Enhanced error handling with graceful DDP fallback
- All documentation updated with streamlined command format
- Better user experience with shorter, memorable arguments

Previous v1.4.2 features:
- Advanced learning rate scheduling: warmup + deep cosine annealing + plateau backup
- Residual connections in LatentConditioner output heads for better gradient flow
- Comprehensive data analysis and diagnostics during training initialization
- Enhanced loss monitoring with Y1/Y2 ratio analysis and outlier detection
- Target: 50% improvement in loss convergence (train <1e-2, val <5e-2)

Previous v1.4.0 features:
- Completely redesigned LatentConditioner architecture with ResNet-style blocks
- SE (Squeeze-and-Excitation) attention blocks for better feature selection  
- Shared backbone architecture eliminating duplicate networks
- Early stopping mechanism with separate y1/y2 loss tracking
- Fixed batch size issues with LayerNorm instead of BatchNorm1d
- Optimized hyperparameters for better convergence (LR: 0.0001, dropout: 0.3)

Supported Tasks:
- Parametric estimations: multi-parametric estimations
- Non-parametric estimations: image, CAD input
- Probabilistic estimations: scattering analysis, up/down-sampling

Input Files:
1. SimulGen-VAE dataset: dataset#X.pickle (3D array: [num_param, num_time, num_node])
2. LatentConditioner dataset: .jpg/.png files in directory (images) or .csv (parametric)
3. preset.txt: dataset preset configuration
4. input_data/condition.txt: training hyperparameters

Usage Examples:
==============

*** Single GPU Training (Small Model)
python SimulGen-VAE.py --preset=1 --plot=2 --lc_only=0 --size=small --load_all=1

*** Single GPU Training (Large Model)  
python SimulGen-VAE.py --preset=1 --plot=2 --lc_only=0 --size=large --load_all=1

*** Multi-GPU Training (DDP) - 2 GPUs
torchrun --nproc_per_node=2 SimulGen-VAE.py --use_ddp --preset=1 --plot=2 --lc_only=0 --size=small --load_all=1

*** Multi-GPU Training (DDP) - 4 GPUs
torchrun --nproc_per_node=4 SimulGen-VAE.py --use_ddp --preset=1 --plot=2 --lc_only=0 --size=large --load_all=1

*** Multi-GPU Training (DDP) - 8 GPUs
torchrun --nproc_per_node=8 SimulGen-VAE.py --use_ddp --preset=1 --plot=2 --lc_only=0 --size=large --load_all=1

*** Multi-Node DDP (2 nodes, 4 GPUs each)
# Node 0 (master):
torchrun --nnodes=2 --nproc_per_node=4 --node_rank=0 --master_addr=192.168.1.100 --master_port=12345 SimulGen-VAE.py --use_ddp --preset=1 --plot=2 --lc_only=0 --size=large --load_all=1
# Node 1:
torchrun --nnodes=2 --nproc_per_node=4 --node_rank=1 --master_addr=192.168.1.100 --master_port=12345 SimulGen-VAE.py --use_ddp --preset=1 --plot=2 --lc_only=0 --size=large --load_all=1

*** LatentConditioner Only Training (Single GPU)
python SimulGen-VAE.py --preset=1 --plot=2 --lc_only=1 --size=small --load_all=1

*** LatentConditioner Only Training (Multi-GPU)
torchrun --nproc_per_node=4 SimulGen-VAE.py --use_ddp --preset=1 --plot=2 --lc_only=1 --size=small --load_all=1

*** ETX Supercomputer (H100)
phd run -ng 1 -p shr_gpu -GR H100 -l %J.log python SimulGen-VAE.py --preset=1 --plot=2 --lc_only=0 --size=small --load_all=1

*** DDP Troubleshooting Commands
nvidia-smi                                          # Check GPU status
ps aux | grep SimulGen-VAE                         # Monitor processes
export NCCL_DEBUG=INFO                             # Enable DDP debugging

*** Monitoring with TensorBoard
tensorboard --logdir=runs --port=6001              # VAE training logs
tensorboard --logdir=LatentConditionerRuns --port=6002  # LatentConditioner logs
"""

def main():
    """Main training function for SimulGenVAE.
    
    Orchestrates the complete training pipeline including:
    - Configuration parsing and argument handling
    - Distributed training setup (DDP) if requested
    - Dataset loading and preprocessing
    - Model initialization (VAE or LatentConditioner only)
    - Training loop execution with monitoring
    - Model checkpointing and result visualization
    
    Supports both single-GPU and multi-GPU distributed training with
    automatic fallback if distributed setup fails.
    
    Command Line Arguments:
        --preset: Dataset preset selection (1-5)
        --plot: Visualization mode (0=interactive, 1=save, 2=off)
        --lc_only: Train only LatentConditioner (1) or full VAE (0)
        --size: Model size preset (small/large/big)
        --load_all: Preload dataset to GPU memory (0/1)
        --use_ddp: Enable distributed data parallel training
    
    Examples:
        Single GPU: python SimulGen-VAE.py --preset=1 --plot=2 --lc_only=0 --size=small
        Multi-GPU: torchrun --nproc_per_node=4 SimulGen-VAE.py --use_ddp --preset=1
    """
    import torch
    import torch.nn as nn
    import pandas as pd
    import numpy as np
    import argparse
    import matplotlib
    import matplotlib.pyplot as plt
    import torch.multiprocessing  # For optimal DataLoader workers
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel as DDP
    from torch.utils.data.distributed import DistributedSampler
    import os

    from modules.common import add_sn
    from modules.input_variables import input_user_variables, input_dataset
    from modules.data_preprocess import reduce_dataset, data_augmentation, data_scaler, latent_conditioner_scaler, latent_conditioner_scaler_input
    # Import models from separate model files
    from modules.latent_conditioner_model_cnn import LatentConditionerImg
    from modules.latent_conditioner_model_parametric import LatentConditioner
    from modules.latent_conditioner_model_vit import TinyViTLatentConditioner
    # Import utilities and training functions from the main file  
    from modules.latent_conditioner import train_latent_conditioner, read_latent_conditioner_dataset_img, read_latent_conditioner_dataset, read_latent_conditioner_dataset_img_pca, safe_cuda_initialization
    from modules.plotter import temporal_plotter
    from modules.VAE_network import VAE
    from modules.train import train
    from modules.utils import MyBaseDataset, get_latest_file, LatentConditionerDataset, get_optimal_workers, setup_distributed_training, print_gpu_mem_checkpoint, parse_condition_file, parse_training_parameters, evaluate_vae_reconstruction, evaluate_vae_simple
    from modules.augmentation import AugmentedDataset, create_augmented_dataloaders

    from torchinfo import summary
    from torch.utils.data import DataLoader, Dataset, random_split

    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", dest = "preset", action = "store")
    parser.add_argument("--plot", dest = "plot", action = "store")
    parser.add_argument("--lc_only", dest = "train_latent_conditioner", action = "store", help="1 = train only LatentConditioner, 0 = train full VAE")
    parser.add_argument("--size", dest = "size", action="store")
    parser.add_argument("--load_all", dest = "load_all", action = "store")
    parser.add_argument("--debug", dest = "debug", action = "store", default="0", help="Enable debug output (0 = off, 1 = on)")
    # Add DDP arguments
    parser.add_argument("--use_ddp", action="store_true", help="Enable distributed data parallel training")
    args = parser.parse_args()

    # Setup distributed training if requested
    is_distributed = setup_distributed_training(args)

    # Parse configuration files
    params = parse_condition_file('input_data/condition.txt')
    config = parse_training_parameters(params)
    
    # Extract commonly used variables for backward compatibility
    num_param = config['num_param']
    num_time = config['num_time']
    num_time_to = config['num_time_to']
    num_node = config['num_node']
    num_node_to = config['num_node_to']
    num_var = config['num_var']
    n_epochs = config['n_epochs']
    batch_size = config['batch_size']
    LR = config['LR']
    latent_dim = config['latent_dim']
    latent_dim_end = config['latent_dim_end']
    loss_type = config['loss_type']
    stretch = config['stretch']
    alpha = config['alpha']
    num_samples_f = config['num_samples_f']
    num_samples_a = config['num_samples_a']
    print_graph_recon = config['print_graph_recon']
    recon_iter = config['recon_iter']
    num_physical_param = config['num_physical_param']
    n_sample = config['n_sample']
    param_dir = config['param_dir']
    latent_conditioner_epoch = config['latent_conditioner_epoch']
    latent_conditioner_lr = config['latent_conditioner_lr']
    latent_conditioner_batch_size = config['latent_conditioner_batch_size']
    latent_conditioner_data_type = config['latent_conditioner_data_type']
    param_data_type = config['param_data_type']
    latent_conditioner_weight_decay = config['latent_conditioner_weight_decay']
    latent_conditioner_dropout_rate = config['latent_conditioner_dropout_rate']
    use_spatial_attention = config['use_spatial_attention']
    use_pca = config['use_pca']
    pca_components = config['pca_components']
    pca_patch_size = config['pca_patch_size']

    if use_pca ==1:
        use_pca = True
        num_pca = pca_components
    else:
        use_pca = False
        num_pca = pca_components

    # Parse debug mode from both config file and command line (command line takes precedence)
    config_debug_mode = int(params.get('debug_mode', 0))  # Default to 0 (disabled)
    debug_mode = int(args.debug) if args.debug != "0" else config_debug_mode

    print('latent_conditioner_data_type: ', latent_conditioner_data_type)
    print('param_data_type: ', param_data_type)

    # Adjust batch size for DDP
    if is_distributed:
        world_size = dist.get_world_size()
        # Adjust batch size to maintain global batch size
        original_batch_size = batch_size
        batch_size = batch_size // world_size
        if dist.get_rank() == 0:
            print(f"Adjusted batch size from {original_batch_size} to {batch_size} per GPU (x{world_size} GPUs)")

    input_shape = num_physical_param

    preset = args.preset
    size = args.size
    load_all_num = args.load_all
    if load_all_num=='1':
        load_all = True
    else:
        load_all = False

    print('load_all to GPU?', load_all)
    
    if size=='small':
        small=True
        print('Small network? :', small)
    elif size=='large':
        small=False
        print('Small network? :', small)
    else:
        NotImplementedError('Unrecoginized size arg')

    if preset =='1':
        with open('preset.txt') as f:
            lines = [line.rstrip('\n') for line in f]

        data_No = int(lines[1])
        init_beta_diviser = int(lines[2])
        num_filter_enc = list(map(int, lines[3].split()))
        latent_conditioner_filter = list(map(int, lines[4].split()))
    else:
        data_No, init_beta_divisor, num_filter_enc, latent_conditioner_filter = input_user_variables()

    if loss_type == 1:
        loss = 'MSE'
    if loss_type == 2:
        loss = 'MAE'
    if loss_type == 3:
        loss = 'smoothL1'
    if loss_type == 4:
        loss = 'Huber'

    init_beta = 10**(-1*init_beta_diviser)

    num_filter_dec = num_filter_enc[::-1]
    strides = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
    num_layer_enc = len(num_filter_enc)
    num_layer_dec = len(num_filter_dec)

    num_node_red_start = 0
    num_node_red_end = num_node
    num_node_red = num_node_red_end - num_node_red_start
    initial_learning_rate = LR
    data_division = 1
    warm_up_rate = 1

    # Display lots of data
    if not is_distributed or dist.get_rank() == 0:
        print('SimulGen-VAE params')
        print('num_param: ', num_param)
        print('num_time: ', num_time)
        print('num_node: ', num_node)
        print('Encoder layers: ', num_filter_enc)
        print('num layer encoder: ', num_layer_enc)
        print('Decoder layers: ', num_filter_dec)
        print('num layer decoder: ', num_layer_dec)
        print('simulgen-vae epochs: ', n_epochs)
        print('batch size: ', batch_size)
        print('learning rate: ', LR)
        print('latent dim: ', latent_dim)
        print('latent dim end: ', latent_dim_end)
        print('loss type: ', loss)
        print('init beta: ', init_beta)

    print_graph = args.plot

    train_latent_conditioner_only = int(args.train_latent_conditioner)

    data_save = input_dataset(num_param, num_time, num_node, data_No)
    num_time, FOM_data, num_node = reduce_dataset(data_save, num_time_to, num_node_red, num_param, num_time, num_node_red_start, num_node_red_end)
    del data_save

    FOM_data_aug = data_augmentation(stretch, FOM_data, num_param, num_node)
    #temporal_plotter(FOM_data_aug, 0, 7,0,print_graph,7)
    new_x_train, _, _ = data_scaler(FOM_data_aug, FOM_data, num_time, num_node, data_No)

    del FOM_data, FOM_data_aug

    #pytorch: [batch_size, num_channels, seqe_length]
    new_x_train = new_x_train.transpose((0,2,1))
    new_x_train = np.float32(new_x_train)

    print('Dataset range: ', np.min(new_x_train), np.max(new_x_train))
    
    # Create augmented dataloaders with default configuration
    # Default config: noise=0.2, scaling=0.1, no shift/cutout, mixup=0.2
    print("Creating augmented dataset with on-the-fly data augmentation...")
    
    # Create dataloaders with error handling (uses default augmentation config)
    dataloader, val_dataloader = create_augmented_dataloaders(
        new_x_train, 
        batch_size=batch_size, 
        load_all=load_all,
        augmentation_config=None,  # Use defaults from create_augmented_dataloaders
        val_split=0.2,  # 80% train, 20% validation
        num_workers=None  # Auto-determine optimal workers
    )
    
    # Create a reference to the dataset for later use
    # This fixes the "dataset is not defined" error
    dataset = dataloader.dataset
    
    if debug_mode == 1:
        print("✓ Augmented dataloaders created successfully")
        print("  - Augmentations enabled: Noise, Scaling, Shift, Mixup, Cutout")
        print(f"  - Training samples: {int(len(new_x_train) * 0.8)}")
        print(f"  - Validation samples: {int(len(new_x_train) * 0.2)}")
        

    print('Dataloader initiated...')
    print_gpu_mem_checkpoint('After dataloader creation', debug_mode)
    # Estimate and print input data GPU memory usage
    def get_tensor_mem_mb(tensor):
        if isinstance(tensor, torch.Tensor):
            return tensor.element_size() * tensor.nelement() / 1024**2
        elif isinstance(tensor, np.ndarray):
            return tensor.nbytes / 1024**2
        else:
            return 0

    # new_x_train is a numpy array
    if debug_mode == 1:
        print(f"[INFO] new_x_train shape: {new_x_train.shape}, dtype: {new_x_train.dtype}, estimated GPU memory: {get_tensor_mem_mb(new_x_train):.2f} MB")
    print_gpu_mem_checkpoint('Before training starts (after data and dataloader setup)', debug_mode)



    from modules.decoder import reparameterize
    from modules.reconstruction_evaluator import ReconstructionEvaluator
    loss_total=0

    # VAE training
    if train_latent_conditioner_only ==0:

        VAE_loss, reconstruction_error, KL_divergence, loss_val_print = train(n_epochs, batch_size, dataloader, val_dataloader, LR, num_filter_enc, num_filter_dec, num_node, latent_dim_end, latent_dim, num_time, alpha, loss, small, load_all, debug_mode)

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        VAE_trained = torch.load('./model_save/SimulGen-VAE', map_location= device, weights_only=False)
        VAE = VAE_trained.eval()

        # Evaluate reconstruction loss on training data
        _ = evaluate_vae_reconstruction(
            VAE, dataloader, device, len(dataloader.dataset), num_filter_enc, latent_dim, 
            latent_dim_end, recon_iter, "Training Reconstruction"
        )

        
        # Evaluate validation loss
        _ = evaluate_vae_reconstruction(
            VAE, val_dataloader, device, len(val_dataloader.dataset), num_filter_enc, latent_dim, 
            latent_dim_end, recon_iter, "Validation"
        )

        # Evaluate on whole dataset for final latent vectors (used for LatentConditioner training)
        dataloader_whole = torch.utils.data.DataLoader(
            MyBaseDataset(new_x_train, False),  # Create dataset from raw data
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )
        latent_vectors, hierarchical_latent_vectors, reconstruction_loss, _, _ = evaluate_vae_reconstruction(
            VAE, dataloader_whole, device, num_param, num_filter_enc, latent_dim, 
            latent_dim_end, recon_iter, "Whole Dataset"
        )

        np.save('model_save/latent_vectors', latent_vectors)
        np.save('model_save/xs', hierarchical_latent_vectors)

        temp5 = './SimulGen-VAE_L2_loss.txt'
        np.savetxt(temp5, reconstruction_loss, fmt = '%e')


    elif train_latent_conditioner_only == 1:
        print('Training LatentConditioner only...')
        latent_vectors = np.load('model_save/latent_vectors.npy')
        hierarchical_latent_vectors = np.load('model_save/xs.npy')
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        VAE_trained = torch.load('model_save/SimulGen-VAE', map_location= device, weights_only=False)
        VAE = VAE_trained.eval()

        # Compare reconstructed validation dataset vs. true validation dataset
        loss_total = evaluate_vae_simple(VAE, val_dataloader, device, "Validation (LatentConditioner Mode)")

        print('--------------------------------')
        print('')
        print('')
        

       


    # LatentConditioner training (runs for both train_latent_conditioner_only == 0 and train_latent_conditioner_only == 1)
    out_latent_vectors = latent_vectors.reshape([num_param, latent_dim_end])
    xs_vectors = hierarchical_latent_vectors.reshape([num_param, -1])


    # Check for PCA_MLP mode
    if latent_conditioner_data_type == 'image' and use_pca:
        print('Loading image data for PCA_MLP mode...')
        image = False  # Use MLP architecture for PCA coefficients
        latent_conditioner_data, latent_conditioner_data_shape = read_latent_conditioner_dataset_img_pca(param_dir, param_data_type, debug_mode, num_pca, pca_patch_size)
    elif latent_conditioner_data_type=='image':
        print('Loading image data for CNN...')
        image=True
        latent_conditioner_data, latent_conditioner_data_shape = read_latent_conditioner_dataset_img(param_dir, param_data_type, debug_mode)
    elif latent_conditioner_data_type=='image_vit':
        print('Loading image data for ViT...')
        image=True
        latent_conditioner_data, latent_conditioner_data_shape = read_latent_conditioner_dataset_img(param_dir, param_data_type, debug_mode)
    elif latent_conditioner_data_type=='csv':
        print('Loading csv data for MLP...')
        image=False
        latent_conditioner_data = read_latent_conditioner_dataset(param_dir, param_data_type)
    else:
        raise NotImplementedError(f'Unrecognized latent_conditioner_data_type: {latent_conditioner_data_type}. Supported options: "image" (CNN), "image_vit" (ViT), "csv" (MLP)')

    physical_param_input = latent_conditioner_data

    # Debug: Print shapes before scaling
    if debug_mode == 1:
        print(f"LatentConditioner scaling - physical_param_input shape: {physical_param_input.shape}")
        print(f"LatentConditioner scaling - out_latent_vectors shape: {out_latent_vectors.shape}")
        print(f"LatentConditioner scaling - xs_vectors shape: {xs_vectors.shape}")

    physical_param_input, param_input_scaler = latent_conditioner_scaler(physical_param_input, './model_save/latent_conditioner_input_scaler.pkl')
    out_latent_vectors, latent_vectors_scaler = latent_conditioner_scaler(out_latent_vectors, './model_save/latent_vectors_scaler.pkl')
    out_hierarchical_latent_vectors, xs_scaler = latent_conditioner_scaler(xs_vectors, './model_save/xs_scaler.pkl')

    if debug_mode == 1:
        print('LatentConditioner data loaded...')
        print('LatentConditioner scale: ')
        print(f'Input: {np.max(physical_param_input)} {np.min(physical_param_input)}')
        print(f'Main latent: {np.max(out_latent_vectors)} {np.min(out_latent_vectors)}')
        print(f'Hierarchical latent: {np.max(out_hierarchical_latent_vectors)} {np.min(out_hierarchical_latent_vectors)}')
    out_hierarchical_latent_vectors = out_hierarchical_latent_vectors.reshape([num_param, len(num_filter_enc)-1, latent_dim])

    # Debug: Print shapes of LatentConditioner dataset inputs
    if debug_mode == 1:
        print(f"LatentConditioner dataset inputs:")
        print(f"  - physical_param_input shape: {physical_param_input.shape}")
        print(f"  - out_latent_vectors shape: {out_latent_vectors.shape}")
        print(f"  - out_hierarchical_latent_vectors shape: {out_hierarchical_latent_vectors.shape}")
    
    # Validate that all inputs have the same first dimension
    input_lengths = [len(physical_param_input), len(out_latent_vectors), len(out_hierarchical_latent_vectors)]
    if len(set(input_lengths)) > 1:
        print(f"Error: Input arrays have different lengths: {input_lengths}")
        print("This will cause the 'Sum of input lengths' error.")
        # Use the minimum length to avoid the error
        min_length = min(input_lengths)
        print(f"Truncating all arrays to length {min_length}")
        physical_param_input = physical_param_input[:min_length]
        out_latent_vectors = out_latent_vectors[:min_length]
        out_hierarchical_latent_vectors = out_hierarchical_latent_vectors[:min_length]
        if debug_mode == 1:
            print(f"Adjusted shapes:")
            print(f"  - physical_param_input shape: {physical_param_input.shape}")
            print(f"  - out_latent_vectors shape: {out_latent_vectors.shape}")
            print(f"  - out_hierarchical_latent_vectors shape: {out_hierarchical_latent_vectors.shape}")
    
    # Create GPU-optimized dataset with preloading
    try:
        latent_conditioner_dataset = LatentConditionerDataset(
            np.float32(physical_param_input), 
            np.float32(out_latent_vectors), 
            np.float32(out_hierarchical_latent_vectors),
            preload_gpu=True  # Enable GPU preloading for maximum speed
        )
    except Exception as e:
        print(f"Failed to create GPU-preloaded dataset: {e}")
        print("Falling back to CPU dataset with pinned memory...")
        latent_conditioner_dataset = LatentConditionerDataset(
            np.float32(physical_param_input), 
            np.float32(out_latent_vectors), 
            np.float32(out_hierarchical_latent_vectors),
            preload_gpu=False
        )

    # Debug: Print shapes of LatentConditioner dataset inputs
    if debug_mode == 1:
        print(f"LatentConditioner dataset inputs:")
        print(f"  - physical_param_input shape: {physical_param_input.shape}")
        print(f"  - out_latent_vectors shape: {out_latent_vectors.shape}")
        print(f"  - out_hierarchical_latent_vectors shape: {out_hierarchical_latent_vectors.shape}")
    
    # Get actual dataset size and calculate split sizes
    latent_conditioner_dataset_size = len(latent_conditioner_dataset)
    train_size = int(0.8 * latent_conditioner_dataset_size)
    val_size = latent_conditioner_dataset_size - train_size
    
    if debug_mode == 1:
        print(f"LatentConditioner dataset size: {latent_conditioner_dataset_size}")
        print(f"Training split: {train_size}, Validation split: {val_size}")
    
    # Verify that the split sizes add up to the dataset size
    if train_size + val_size != latent_conditioner_dataset_size:
        print(f"Warning: Split sizes don't add up! {train_size} + {val_size} != {latent_conditioner_dataset_size}")
        # Adjust val_size to make it work
        val_size = latent_conditioner_dataset_size - train_size
        print(f"Adjusted validation size to: {val_size}")
    
    latent_conditioner_train_dataset, latent_conditioner_validation_dataset = random_split(latent_conditioner_dataset, [train_size, val_size])
    # Optimize LatentConditioner DataLoaders with intelligent worker detection
    latent_conditioner_optimal_workers = get_optimal_workers(latent_conditioner_dataset_size, False, latent_conditioner_batch_size)  # LatentConditioner data is not load_all
    
    # Optimize DataLoader settings based on whether data is GPU-preloaded
    is_gpu_preloaded = hasattr(latent_conditioner_dataset, 'on_gpu') and latent_conditioner_dataset.on_gpu
    pin_memory_setting = False if is_gpu_preloaded else True  # No need for pinning if already on GPU
    
    if latent_conditioner_optimal_workers == 0:
        # Single-threaded LatentConditioner DataLoaders
        latent_conditioner_dataloader = torch.utils.data.DataLoader(
            latent_conditioner_train_dataset, 
            batch_size=latent_conditioner_batch_size, 
            shuffle=True, 
            num_workers=0,
            pin_memory=pin_memory_setting,
            drop_last=True,  # Ensures consistent batch sizes
            persistent_workers=False
        )
        latent_conditioner_validation_dataloader = torch.utils.data.DataLoader(
            latent_conditioner_validation_dataset, 
            batch_size=latent_conditioner_batch_size, 
            shuffle=False, 
            num_workers=0,
            pin_memory=pin_memory_setting,
            drop_last=False,
            persistent_workers=False
        )
        if debug_mode == 1:
            gpu_status = "GPU-preloaded" if is_gpu_preloaded else "CPU with pinned memory"
            print(f"   ✓ LatentConditioner DataLoader: Single-threaded, {gpu_status} ({latent_conditioner_dataset_size} samples)")
    else:
        # Multi-threaded LatentConditioner DataLoaders
        latent_conditioner_dataloader = torch.utils.data.DataLoader(
            latent_conditioner_train_dataset, 
            batch_size=latent_conditioner_batch_size, 
            shuffle=True, 
            num_workers=latent_conditioner_optimal_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2
        )
        latent_conditioner_validation_dataloader = torch.utils.data.DataLoader(
            latent_conditioner_validation_dataset, 
            batch_size=latent_conditioner_batch_size, 
            shuffle=False, 
            num_workers=latent_conditioner_optimal_workers,
            pin_memory=True,
            persistent_workers=True,
            prefetch_factor=2
        )
        if debug_mode == 1:
            print(f"   ✓ LatentConditioner DataLoader: {latent_conditioner_optimal_workers} workers ({latent_conditioner_dataset_size} samples)")

    size2 = len(num_filter_enc)-1


    # Get optimal device with comprehensive debugging
    if debug_mode == 1:
        print("\n🔍 === LATENT CONDITIONER INITIALIZATION DEBUGGING ===")
    device = safe_cuda_initialization(debug_mode)  # Get safe device
    if debug_mode == 1:
        print(f"Selected device: {device}")
        print("===================================================\n")
    
    if latent_conditioner_data_type=='image' and use_pca==False:
        try:
            if debug_mode == 1:
                print("Initializing LatentConditioner CNN image model...")
            latent_conditioner = LatentConditionerImg(latent_conditioner_filter, latent_dim_end, input_shape, latent_dim, size2, latent_conditioner_data_shape, dropout_rate=latent_conditioner_dropout_rate, use_attention=bool(use_spatial_attention)).to(device)
            if debug_mode == 1:
                print(f"✓ CNN model successfully initialized on {device}")
        except RuntimeError as e:
            print(f"❌ Error initializing LatentConditioner CNN image model: {e}")
            print("   Falling back to CPU-only model")
            device = torch.device('cpu')
            latent_conditioner = LatentConditionerImg(latent_conditioner_filter, latent_dim_end, input_shape, latent_dim, size2, latent_conditioner_data_shape, dropout_rate=latent_conditioner_dropout_rate, use_attention=bool(use_spatial_attention)).to(device)
            if debug_mode == 1:
                print(f"✓ CNN model fallback initialized on {device}")

    elif use_pca:
        print('Using PCA mode...')

        input_shape = num_pca #number of PCA coefficients
        try:
            if debug_mode == 1:
                print("Initializing LatentConditioner MLP model for PCA...")
            latent_conditioner = LatentConditioner(latent_conditioner_filter, latent_dim_end, input_shape, latent_dim, size2, dropout_rate=latent_conditioner_dropout_rate).to(device)
            if debug_mode == 1:
                print(f"✓ CNN model successfully initialized on {device}")
        except RuntimeError as e:
            print(f"❌ Error initializing LatentConditioner MLP model for PCA: {e}")
            print("   Falling back to CPU-only model")
            device = torch.device('cpu')
            latent_conditioner = LatentConditioner(latent_conditioner_filter, latent_dim_end, input_shape, latent_dim, size2, dropout_rate=latent_conditioner_dropout_rate).to(device)
            if debug_mode == 1:
                print(f"✓ MLP model fallback initialized on {device}")

    elif latent_conditioner_data_type=='image_vit':
        try:
            if debug_mode == 1:
                print("Initializing LatentConditioner ViT image model...")
            # ViT-specific parameters (can be made configurable later)
            img_size = int(latent_conditioner_data_shape[0])  # Should be 128
            patch_size = 16
            embed_dim = 64
            num_layers = 2
            num_heads = 4
            mlp_ratio = 2
            latent_conditioner = TinyViTLatentConditioner(
                latent_dim_end=latent_dim_end, 
                latent_dim=latent_dim, 
                size2=size2,
                img_size=img_size,
                patch_size=patch_size,
                embed_dim=embed_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=latent_conditioner_dropout_rate
            ).to(device)
            if debug_mode == 1:
                print(f"✓ ViT model successfully initialized on {device}")
        except RuntimeError as e:
            print(f"❌ Error initializing LatentConditioner ViT image model: {e}")
            print("   Falling back to CPU-only model")
            device = torch.device('cpu')
            latent_conditioner = TinyViTLatentConditioner(
                latent_dim_end=latent_dim_end, 
                latent_dim=latent_dim, 
                size2=size2,
                img_size=img_size,
                patch_size=patch_size,
                embed_dim=embed_dim,
                num_layers=num_layers,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=latent_conditioner_dropout_rate
            ).to(device)
            if debug_mode == 1:
                print(f"✓ ViT model fallback initialized on {device}")
    elif latent_conditioner_data_type=='csv':
        try:
            if debug_mode == 1:
                print("Initializing LatentConditioner MLP CSV model...")
            latent_conditioner = LatentConditioner(latent_conditioner_filter, latent_dim_end, input_shape, latent_dim, size2, dropout_rate=latent_conditioner_dropout_rate).to(device)
            if debug_mode == 1:
                print(f"✓ MLP model successfully initialized on {device}")
        except RuntimeError as e:
            print(f"❌ Error initializing LatentConditioner MLP CSV model: {e}")
            print("   Falling back to CPU-only model")
            device = torch.device('cpu')
            latent_conditioner = LatentConditioner(latent_conditioner_filter, latent_dim_end, input_shape, latent_dim, size2, dropout_rate=latent_conditioner_dropout_rate).to(device)
            if debug_mode == 1:
                print(f"✓ MLP model fallback initialized on {device}")
    else:
        raise NotImplementedError(f'Unrecognized latent_conditioner_data_type: {latent_conditioner_data_type}. Supported options: "image" (CNN), "image_vit" (ViT), "csv" (MLP)')

    # Print CUDA diagnostic information before LatentConditioner training
    if debug_mode == 1 and torch.cuda.is_available():
        try:
            print("\n=== CUDA Diagnostics ===")
            print(f"CUDA available: {torch.cuda.is_available()}")
            print(f"CUDA device count: {torch.cuda.device_count()}")
            print(f"Current device: {torch.cuda.current_device()}")
            print(f"Device name: {torch.cuda.get_device_name(0)}")
            print(f"CUDA version: {torch.version.cuda}")
            print(f"Current memory allocated: {torch.cuda.memory_allocated() / 1024**2:.2f} MB")
            print(f"Max memory allocated: {torch.cuda.max_memory_allocated() / 1024**2:.2f} MB")
            print(f"Memory reserved: {torch.cuda.memory_reserved() / 1024**2:.2f} MB")
            print("========================\n")
        except Exception as e:
            print(f"Error getting CUDA diagnostics: {e}")

    try:
        print("Starting LatentConditioner training...")
        LatentConditioner_loss = train_latent_conditioner(latent_conditioner_epoch, latent_conditioner_dataloader, latent_conditioner_validation_dataloader, latent_conditioner, latent_conditioner_lr, weight_decay=latent_conditioner_weight_decay, is_image_data=image, image_size=256, debug_mode=debug_mode)
        print("LatentConditioner training completed successfully")
    except Exception as e:
        print(f"Error during LatentConditioner training: {e}")
        print("If you're seeing CUDA errors about device side assertions, try recompiling PyTorch with torch_USA_CUDA_DSA=1")
        print("Attempting to save emergency checkpoint...")
        try:
            torch.save(latent_conditioner, './model_save/latent_conditioner_emergency_save')
            print("Emergency model saved")
        except:
            print("Could not save emergency model")

    latent_x = np.linspace(0, latent_dim_end-1, latent_dim_end)
    latent_hierarchical_x = np.linspace(0, latent_dim-1, latent_dim)

    # Ensure latent conditioner is on the correct device for evaluation
    if debug_mode == 1:
        print(f"\n🔍 Pre-evaluation device check:")
        print(f"  Model device before: {next(latent_conditioner.parameters()).device}")
    
    eval_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if debug_mode == 1:
        print(f"  Target evaluation device: {eval_device}")
    
    try:
        latent_conditioner = latent_conditioner.to(eval_device)
        if debug_mode == 1:
            print(f"  ✓ Model moved to: {next(latent_conditioner.parameters()).device}")
    except Exception as e:
        if debug_mode == 1:
            print(f"  ❌ Failed to move model: {e}")
        eval_device = next(latent_conditioner.parameters()).device
        if debug_mode == 1:
            print(f"  Using current device: {eval_device}")
    
    device = eval_device  # Update device variable for consistency

    if VAE in globals():
        del VAE

    VAE_trained = torch.load('model_save/SimulGen-VAE', map_location= device, weights_only=False)
    VAE = VAE_trained.eval()

    # Use the new reconstruction evaluator for cleaner, more accurate evaluation
    print("Starting reconstruction evaluation with proper data alignment...")
    evaluator = ReconstructionEvaluator(VAE, device, num_time, debug_mode)
    evaluator.evaluate_reconstruction_comparison(
        latent_conditioner, latent_conditioner_dataset, 
        new_x_train, latent_vectors_scaler, xs_scaler
    )

if __name__ == "__main__":
    main()
