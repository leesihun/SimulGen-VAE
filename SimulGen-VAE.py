#!/usr/bin/env python3
"""SimulGenVAE v2.0.0 - High-Performance Physics-Aware Variational Autoencoder

A clean, well-documented Variational Autoencoder system designed for fast generation 
and inference of transient/static simulation data with Physics-Aware Neural Network 
(PANN) integration.

Features:
- Single-GPU and multi-GPU distributed training (DDP)
- Three latent conditioning architectures: MLP, CNN, Vision Transformer
- Hierarchical latent space (main: 32D, hierarchical: 8D)
- Advanced anti-overfitting measures with early stopping
- Mixed precision training with gradient checkpointing
- Comprehensive data augmentation and validation

Supported Input Types:
- Simulation data: 3D arrays [parameters, timesteps, nodes]
- Parametric data: CSV files for MLP conditioning
- Image data: PNG/JPG files for CNN/ViT conditioning
- PCA-processed images for efficient MLP conditioning

Quick Start:
    Single GPU: python SimulGen-VAE.py --preset=1 --plot=2 --lc_only=0 --size=small
    Multi-GPU:  torchrun --nproc_per_node=4 SimulGen-VAE.py --use_ddp --preset=1

Author: SiHun Lee, Ph.D.
Contact: kevin1007kr@gmail.com
Version: 2.0.0 (Refactored & Documented)
"""

def main():
    """Main training function for SimulGenVAE.
    
    Orchestrates the complete training pipeline including configuration parsing,
    distributed training setup, dataset loading, model initialization, training
    execution, and result evaluation.
    
    Command Line Arguments:
        --preset (str): Dataset preset selection (1-5, default from preset.txt)
        --plot (str): Visualization mode (0=interactive, 1=save, 2=off)
        --lc_only (str): Train mode - '1' for LatentConditioner only, '0' for full VAE
        --size (str): Model architecture - 'small' or 'large'
        --load_all (str): Memory mode - '1' to preload all data, '0' for lazy loading
        --use_ddp (flag): Enable distributed data parallel training
    
    Training Modes:
        Full VAE (lc_only=0): Train VAE encoder/decoder + LatentConditioner
        LC Only (lc_only=1): Train only LatentConditioner using pre-trained VAE
        End-to-End: Set use_e2e_training=1 in condition.txt for direct data reconstruction optimization
    
    Examples:
        python SimulGen-VAE.py --preset=1 --plot=2 --lc_only=0 --size=small
        torchrun --nproc_per_node=4 SimulGen-VAE.py --use_ddp --preset=1
    
    Raises:
        NotImplementedError: If unsupported size or conditioning type specified
        FileNotFoundError: If required configuration files are missing
    """
    # Core PyTorch imports
    import torch
    import torch.nn as nn
    import torch.distributed as dist
    from torch.utils.data import random_split
    
    # Scientific computing and visualization
    import numpy as np
    
    # System utilities
    import argparse

    # SimulGenVAE core modules
    from modules.input_variables import input_user_variables, input_dataset
    from modules.data_preprocess import (
        reduce_dataset, data_augmentation, data_scaler, 
        latent_conditioner_scaler
    )
    
    # Model architectures
    from modules.VAE_network import VAE
    from modules.latent_conditioner_model_cnn import LatentConditionerImg
    from modules.latent_conditioner_model_parametric import LatentConditioner
    from modules.latent_conditioner_model_vit import TinyViTLatentConditioner
    
    # Training and evaluation
    from modules.train import train
    from modules.latent_conditioner import (
        train_latent_conditioner, read_latent_conditioner_dataset_img,
        read_latent_conditioner_dataset, safe_cuda_initialization
    )
    from modules.latent_conditioner_e2e import train_latent_conditioner_e2e
    from modules.enhanced_latent_conditioner_training import train_latent_conditioner_with_enhancements
    from modules.reconstruction_evaluator import ReconstructionEvaluator
    
    # Utilities and data handling
    from modules.utils import (
        MyBaseDataset, LatentConditionerDataset, get_optimal_workers,
        setup_distributed_training, parse_condition_file, parse_training_parameters,
        evaluate_vae_reconstruction
    )
    from modules.augmentation import create_augmented_dataloaders
    from modules.plotter import temporal_plotter, dual_view_plotter

    from modules.utils import initialize_folder

    # Parse command line arguments
    parser = argparse.ArgumentParser(
        description="SimulGenVAE: High-Performance Physics-Aware Variational Autoencoder",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Examples:
  Single GPU:  python SimulGen-VAE.py --preset=1 --plot=2 --lc_only=0 --size=small
  Multi-GPU:   torchrun --nproc_per_node=4 SimulGen-VAE.py --use_ddp --preset=1
  LC Training: python SimulGen-VAE.py --preset=1 --lc_only=1 --size=small"""
    )
    
    parser.add_argument("--preset", dest="preset", default="1", 
                       help="Dataset preset selection (1-5, default: 1)")
    parser.add_argument("--plot", dest="plot", default="2",
                       help="Visualization mode: 0=interactive, 1=save, 2=off (default: 2)")
    parser.add_argument("--lc_only", dest="train_latent_conditioner", default="0",
                       help="Training mode: 0=full VAE, 1=LatentConditioner only (default: 0)")
    parser.add_argument("--size", dest="size", default="small", choices=["small", "large"],
                       help="Model architecture size (default: small)")
    parser.add_argument("--load_all", dest="load_all", default="0",
                       help="Memory mode: 0=lazy loading, 1=preload all data (default: 0)")
    parser.add_argument("--use_ddp", action="store_true",
                       help="Enable distributed data parallel training")
    
    args = parser.parse_args()

    # Initialize distributed training if requested
    is_distributed = setup_distributed_training(args)
    
    # Load and parse configuration files
    print("Loading configuration...")
    try:
        params = parse_condition_file('input_data/condition.txt')
        config = parse_training_parameters(params)
    except FileNotFoundError as e:
        print(f"Configuration file not found: {e}")
        print("Please ensure input_data/condition.txt exists")
        return
    except Exception as e:
        print(f"Error parsing configuration: {e}")
        return
    
    # Extract configuration parameters for easy access
    # Data dimensions
    num_param = config['num_param']
    num_time = config['num_time']
    num_time_to = config['num_time_to']
    num_node = config['num_node']
    num_node_to = config['num_node_to']
    num_var = config['num_var']
    
    # Training parameters
    n_epochs = config['n_epochs']
    batch_size = config['batch_size']
    LR = config['LR']
    loss_type = config['loss_type']
    alpha = config['alpha']
    
    # Model architecture
    latent_dim = config['latent_dim']  # Hierarchical latent dimension
    latent_dim_end = config['latent_dim_end']  # Main latent dimension
    
    # Data processing
    stretch = config['stretch']
    num_samples_f = config['num_samples_f']
    num_samples_a = config['num_samples_a']
    
    # Evaluation parameters
    print_graph_recon = config['print_graph_recon']
    recon_iter = config['recon_iter']
    n_sample = config['n_sample']
    
    # LatentConditioner parameters
    num_physical_param = config['num_physical_param']
    param_dir = config['param_dir']
    latent_conditioner_epoch = config['latent_conditioner_epoch']
    latent_conditioner_lr = config['latent_conditioner_lr']
    latent_conditioner_batch_size = config['latent_conditioner_batch_size']
    latent_conditioner_data_type = config['latent_conditioner_data_type']
    param_data_type = config['param_data_type']
    latent_conditioner_weight_decay = config['latent_conditioner_weight_decay']
    latent_conditioner_dropout_rate = config['latent_conditioner_dropout_rate']
    use_spatial_attention = config['use_spatial_attention']

    print(f"Latent conditioner data type: {latent_conditioner_data_type}")
    print(f"Parameter data type: {param_data_type}")

    # Adjust batch size for DDP
    if is_distributed:
        world_size = dist.get_world_size()
        # Adjust batch size to maintain global batch size
        original_batch_size = batch_size
        batch_size = batch_size // world_size
        if dist.get_rank() == 0:
            print(f"Adjusted batch size from {original_batch_size} to {batch_size} per GPU (×{world_size} GPUs)")

    input_shape = num_physical_param

    preset = args.preset
    size = args.size
    load_all_num = args.load_all
    if load_all_num=='1':
        load_all = True
    else:
        load_all = False

    print(f"Load all data to GPU: {load_all}")
    
    if size == 'small':
        small = True
        print(f"Using small network architecture: {small}")
    elif size == 'large':
        small = False
        print(f"Using small network architecture: {small}")
    else:
        raise NotImplementedError(f"Unrecognized size argument: {size}")

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

    if not is_distributed or dist.get_rank() == 0:
        print("\n=== SimulGen-VAE Configuration ===")
        print(f"Parameters: {num_param}")
        print(f"Time steps: {num_time}")
        print(f"Nodes: {num_node}")
        print(f"Encoder layers: {num_filter_enc} (total: {num_layer_enc})")
        print(f"Decoder layers: {num_filter_dec} (total: {num_layer_dec})")
        print(f"Training epochs: {n_epochs}")
        print(f"Batch size: {batch_size}")
        print(f"Learning rate: {LR}")
        print(f"Latent dimensions: {latent_dim} (hierarchical), {latent_dim_end} (main)")
        print(f"Loss function: {loss}")
        print(f"Initial beta: {init_beta}")
        print("=" * 35)

    print_graph = args.plot

    train_latent_conditioner_only = int(args.train_latent_conditioner)

    if train_latent_conditioner_only == 0:
        # Initialize folder contents
        initialize_folder('model_save')
        initialize_folder('checkpoints')
        initialize_folder('LatentConditionerRuns')
        initialize_folder('model_save')
        initialize_folder('output')

    data_save = input_dataset(num_param, num_time, num_node, data_No)
    num_time, FOM_data, num_node = reduce_dataset(data_save, num_time_to, num_node_red, num_param, num_time, num_node_red_start, num_node_red_end)
    del data_save

    FOM_data_aug = data_augmentation(stretch, FOM_data, num_param, num_node)
    
    # Enhanced plotting with dual view (temporal + nodal)
    if print_graph != "2":  # If plotting is enabled (not off mode)
        print("Generating dual-view plots for data visualization...")
        dual_view_plotter(FOM_data_aug, param_idx=7, print_graph=print_graph)
        temporal_plotter(FOM_data_aug, 0, 7, 0, print_graph, 7)
    
    new_x_train, _, _ = data_scaler(FOM_data_aug, FOM_data, num_time, num_node, data_No)

    del FOM_data, FOM_data_aug

    #pytorch: [batch_size, num_channels, seqe_length]
    new_x_train = new_x_train.transpose((0,2,1))
    new_x_train = np.float32(new_x_train)

    print(f"Dataset value range: [{np.min(new_x_train):.4f}, {np.max(new_x_train):.4f}]")
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
    
    print(f"Augmented dataloaders created - Training: {int(len(new_x_train) * 0.8)}, Validation: {int(len(new_x_train) * 0.2)} samples")
    print("Dataloader initialization complete")

    # VAE training
    if train_latent_conditioner_only ==0:

        _ = train(n_epochs, batch_size, dataloader, val_dataloader, LR, num_filter_enc, num_filter_dec, num_node, latent_dim_end, latent_dim, num_time, alpha, loss, small, load_all)
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
        print("Training LatentConditioner only...")
        latent_vectors = np.load('model_save/latent_vectors.npy')
        hierarchical_latent_vectors = np.load('model_save/xs.npy')
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        VAE_trained = torch.load('model_save/SimulGen-VAE', map_location=device, weights_only=False)
        VAE = VAE_trained.eval()

    # LatentConditioner training (runs for both train_latent_conditioner_only == 0 and train_latent_conditioner_only == 1)
    out_latent_vectors = latent_vectors.reshape([num_param, latent_dim_end])
    xs_vectors = hierarchical_latent_vectors.reshape([num_param, -1])

    # Check for PCA_MLP mode
    if latent_conditioner_data_type=='image':
        print('Loading image data for CNN...')
        image=True
        latent_conditioner_data, latent_conditioner_data_shape = read_latent_conditioner_dataset_img(param_dir, param_data_type)
    elif latent_conditioner_data_type=='image_vit':
        print('Loading image data for ViT...')
        image=True
        latent_conditioner_data, latent_conditioner_data_shape = read_latent_conditioner_dataset_img(param_dir, param_data_type)
    elif latent_conditioner_data_type=='csv':
        print('Loading csv data for MLP...')
        image=False
        latent_conditioner_data = read_latent_conditioner_dataset(param_dir, param_data_type)
    else:
        raise NotImplementedError(f'Unrecognized latent_conditioner_data_type: {latent_conditioner_data_type}. Supported options: "image" (CNN), "image_vit" (ViT), "csv" (MLP)')

    physical_param_input = latent_conditioner_data

    #physical_param_input, param_input_scaler = latent_conditioner_scaler(physical_param_input, './model_save/latent_conditioner_input_scaler.pkl')
    if latent_conditioner_data_type == 'image':
        physical_param_input = physical_param_input / 255.0
    else:
        physical_param_input, param_input_scaler = latent_conditioner_scaler(physical_param_input, './model_save/latent_conditioner_input_scaler.pkl')
    out_latent_vectors, latent_vectors_scaler = latent_conditioner_scaler(out_latent_vectors, './model_save/latent_vectors_scaler.pkl')
    out_hierarchical_latent_vectors, xs_scaler = latent_conditioner_scaler(xs_vectors, './model_save/xs_scaler.pkl')

    print("LatentConditioner data loaded and scaled")
    print(f"Input range: [{np.min(physical_param_input):.4f}, {np.max(physical_param_input):.4f}]")
    print(f"Main latent range: [{np.min(out_latent_vectors):.4f}, {np.max(out_latent_vectors):.4f}]")
    print(f"Hierarchical latent range: [{np.min(out_hierarchical_latent_vectors):.4f}, {np.max(out_hierarchical_latent_vectors):.4f}]")
    out_hierarchical_latent_vectors = out_hierarchical_latent_vectors.reshape([num_param, len(num_filter_enc)-1, latent_dim])

    # Validate that all inputs have the same first dimension
    input_lengths = [len(physical_param_input), len(out_latent_vectors), len(out_hierarchical_latent_vectors)]
    if len(set(input_lengths)) > 1:
        min_length = min(input_lengths)
        print(f"Warning: Input arrays have different lengths {input_lengths}. Truncating to {min_length}")
        physical_param_input = physical_param_input[:min_length]
        out_latent_vectors = out_latent_vectors[:min_length]
        out_hierarchical_latent_vectors = out_hierarchical_latent_vectors[:min_length]
    
    latent_conditioner_dataset = LatentConditionerDataset(
        np.float32(physical_param_input), 
        np.float32(out_latent_vectors), 
        np.float32(out_hierarchical_latent_vectors),
        preload_gpu=True
    )

    # Get actual dataset size and calculate split sizes
    latent_conditioner_dataset_size = len(latent_conditioner_dataset)
    train_size = int(0.8 * latent_conditioner_dataset_size)
    val_size = latent_conditioner_dataset_size - train_size
    
    print(f"LatentConditioner dataset: {train_size} training, {val_size} validation samples")
    
    # Verify that the split sizes add up to the dataset size
    if train_size + val_size != latent_conditioner_dataset_size:
        val_size = latent_conditioner_dataset_size - train_size
        print(f"Adjusted validation size to: {val_size}")
    
    latent_conditioner_train_dataset, latent_conditioner_validation_dataset = random_split(latent_conditioner_dataset, [train_size, val_size])
    latent_conditioner_optimal_workers = get_optimal_workers(latent_conditioner_dataset_size, False, latent_conditioner_batch_size)
    
    latent_conditioner_dataloader = torch.utils.data.DataLoader(
        latent_conditioner_train_dataset, 
        batch_size=latent_conditioner_batch_size, 
        shuffle=True, 
        num_workers=latent_conditioner_optimal_workers,
        pin_memory=False,
        persistent_workers=latent_conditioner_optimal_workers > 0,
        prefetch_factor=2 if latent_conditioner_optimal_workers > 0 else None,
        drop_last=True
    )
    latent_conditioner_validation_dataloader = torch.utils.data.DataLoader(
        latent_conditioner_validation_dataset, 
        batch_size=latent_conditioner_batch_size, 
        shuffle=False, 
        num_workers=latent_conditioner_optimal_workers,
        pin_memory=False,
        persistent_workers=latent_conditioner_optimal_workers > 0,
        prefetch_factor=2 if latent_conditioner_optimal_workers > 0 else None,
        drop_last=False
    )

    size2 = len(num_filter_enc)-1


    device = safe_cuda_initialization()
    
    if latent_conditioner_data_type == 'image':
        print("Initializing LatentConditioner CNN image model...")
        latent_conditioner = LatentConditionerImg(latent_conditioner_filter, latent_dim_end, input_shape, latent_dim, size2, latent_conditioner_data_shape, dropout_rate=latent_conditioner_dropout_rate, use_attention=bool(use_spatial_attention)).to(device)

    elif latent_conditioner_data_type == 'image_vit':
        print("Initializing LatentConditioner ViT image model...")
        img_size = int(latent_conditioner_data_shape[0])
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
    elif latent_conditioner_data_type == 'csv':
        print("Initializing LatentConditioner MLP CSV model...")
        latent_conditioner = LatentConditioner(latent_conditioner_filter, latent_dim_end, input_shape, latent_dim, size2, dropout_rate=latent_conditioner_dropout_rate).to(device)
    else:
        raise NotImplementedError(f'Unrecognized latent_conditioner_data_type: {latent_conditioner_data_type}. Supported options: "image" (CNN), "image_vit" (ViT), "csv" (MLP)')


    print("Starting LatentConditioner training...")
    
    # Check for end-to-end training mode
    if config.get('use_e2e_training', 0) == 1:
        print("Using end-to-end latent conditioner training")
        print("Architecture: Input Conditions → Latent Conditioner → VAE Decoder → Reconstructed Data")
        
        # Create target dataloader for end-to-end training
        # Use the same VAE training data as target for reconstruction
        target_dataset = MyBaseDataset(FOM_data, load_all, transform=None)
        target_dataloader = torch.utils.data.DataLoader(
            target_dataset,
            batch_size=latent_conditioner_batch_size,
            shuffle=True,
            num_workers=latent_conditioner_optimal_workers,
            pin_memory=False,
            persistent_workers=latent_conditioner_optimal_workers > 0,
            prefetch_factor=2 if latent_conditioner_optimal_workers > 0 else None,
            drop_last=True
        )
        
        LatentConditioner_loss = train_latent_conditioner_e2e(
            latent_conditioner_epoch=latent_conditioner_epoch,
            latent_conditioner_dataloader=latent_conditioner_dataloader,
            latent_conditioner_validation_dataloader=latent_conditioner_validation_dataloader,
            latent_conditioner=latent_conditioner,
            target_dataloader=target_dataloader,
            latent_conditioner_lr=latent_conditioner_lr,
            weight_decay=latent_conditioner_weight_decay,
            is_image_data=image,
            image_size=256,
            config=config
        )
    
    # Use enhanced training if enabled for CNN models, otherwise use original training
    elif latent_conditioner_data_type == "image" and config.get('use_enhanced_loss', 0):
        print(f"Using enhanced CNN latent conditioner training")
        LatentConditioner_loss = train_latent_conditioner_with_enhancements(
            latent_conditioner_epoch=latent_conditioner_epoch,
            latent_conditioner_dataloader=latent_conditioner_dataloader,
            latent_conditioner_validation_dataloader=latent_conditioner_validation_dataloader,
            latent_conditioner=latent_conditioner,
            latent_conditioner_lr=latent_conditioner_lr,
            weight_decay=latent_conditioner_weight_decay,
            is_image_data=image,
            image_size=256,
            config=config,
            use_enhanced_loss=True
        )

        
    else:
        # Use original training (for non-CNN models or when enhanced loss is disabled)
        if latent_conditioner_data_type == "image" and not config.get('use_enhanced_loss', 0):
            print("Enhanced loss disabled - using original CNN training")
        else:
            print(f"Using original training for {latent_conditioner_data_type} model")
        
        LatentConditioner_loss = train_latent_conditioner(
            latent_conditioner_epoch, latent_conditioner_dataloader, 
            latent_conditioner_validation_dataloader, latent_conditioner, 
            latent_conditioner_lr, weight_decay=latent_conditioner_weight_decay, 
            is_image_data=image, image_size=256
        )
    
    print("LatentConditioner training completed successfully")

    eval_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    latent_conditioner = latent_conditioner.to(eval_device)
    device = eval_device

    if VAE in globals():
        del VAE

    VAE_trained = torch.load('model_save/SimulGen-VAE', map_location= device, weights_only=False)
    VAE = VAE_trained.eval()

    print("Starting reconstruction evaluation...")
    evaluator = ReconstructionEvaluator(VAE, device, num_time)
    evaluator.evaluate_reconstruction_comparison(
        latent_conditioner, latent_conditioner_dataset, 
        new_x_train, latent_vectors_scaler, xs_scaler
    )

if __name__ == "__main__":
    main()
