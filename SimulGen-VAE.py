# Single GPU: python SimulGen-VAE.py --preset=1 --plot=2 --lc_only=0 --size=small --load_all=1
# Multi-GPU:  torchrun --nproc_per_node=4 SimulGen-VAE.py --use_ddp --preset=1 --plot=2 --lc_only=0 --size=small --load_all=1

"""
SimulGen-VAE v1.4.3
====================

A high-performance VAE for fast generation and inference of transient/static simulation data 
with Physics-Aware Neural Network (PANN) integration.

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
    # Import models from the models file
    from modules.latent_conditioner_models import LatentConditionerImg, LatentConditioner, TinyViTLatentConditioner
    # Import utilities and training functions from the main file  
    from modules.latent_conditioner import train_latent_conditioner, read_latent_conditioner_dataset_img, read_latent_conditioner_dataset, safe_cuda_initialization
    from modules.plotter import temporal_plotter
    from modules.VAE_network import VAE
    from modules.train import train, print_gpu_mem_checkpoint
    from modules.utils import MyBaseDataset, get_latest_file, LatentConditionerDataset, get_optimal_workers
    from modules.augmentation import AugmentedDataset, create_augmented_dataloaders

    from torchinfo import summary
    from torch.utils.data import DataLoader, Dataset, random_split

    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", dest = "preset", action = "store")
    parser.add_argument("--plot", dest = "plot", action = "store")
    parser.add_argument("--lc_only", dest = "train_latent_conditioner", action = "store", help="1 = train only LatentConditioner, 0 = train full VAE")
    parser.add_argument("--size", dest = "size", action="store")
    parser.add_argument("--load_all", dest = "load_all", action = "store")
    # Add DDP arguments
    parser.add_argument("--use_ddp", action="store_true", help="Enable distributed data parallel training")
    args = parser.parse_args()

    # Setup distributed training if requested
    if args.use_ddp:
        try:
            # torchrun automatically sets environment variables
            local_rank = int(os.environ.get("LOCAL_RANK", -1))
            if local_rank == -1:
                print("For DDP training, please use: torchrun --nproc_per_node=NUM_GPUS SimulGen-VAE.py --use_ddp [other args]")
                is_distributed = False
            else:
                # Initialize the process group (torchrun handles most setup)
                torch.cuda.set_device(local_rank)
                dist.init_process_group(backend="nccl")
                is_distributed = True
                print(f"Initialized DDP process group. Rank {dist.get_rank()} of {dist.get_world_size()}")
        except Exception as e:
            print(f"Failed to initialize DDP: {e}")
            print("Falling back to single GPU training")
            is_distributed = False
    else:
        is_distributed = False
    
    def print_gpu_mem_checkpoint(msg):
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1024**2
            max_allocated = torch.cuda.max_memory_allocated() / 1024**2
            print(f"[GPU MEM] {msg}: Allocated={allocated:.2f}MB, Max Allocated={max_allocated:.2f}MB")
            torch.cuda.reset_peak_memory_stats()

    def parse_condition_file(filepath):
        params = {}
        with open(filepath, encoding='utf-8') as f:
            for line in f:
                # Remove comments and whitespace
                line = line.split('#')[0].strip()
                if not line or line.startswith('%') or line.startswith("'"):
                    continue  # skip empty lines and section markers
                parts = line.split()
                if len(parts) >= 2:
                    key = parts[0]
                    value = parts[1]
                    params[key] = value
        return params

    # Usage
    params = parse_condition_file('input_data/condition.txt')

    # Now you can access values by key, e.g.:
    num_param = int(params['Dim1'])
    num_time = int(params['Dim2'])
    num_time_to = int(params['Dim2_red'])
    num_node = int(params['Dim3'])
    num_node_to = int(params['Dim3_red'])
    num_var = int(params['num_var'])
    n_epochs = int(params['Training_epochs'])
    batch_size = int(params['Batch_size'])
    LR = float(params['LearningR'])
    latent_dim = int(params['Latent_dim'])
    latent_dim_end = int(params['Latent_dim_end'])
    loss_type = int(params['Loss_type'])
    stretch = int(params['Stretch'])
    alpha = int(params['alpha'])
    num_samples_f = int(params.get('num_aug_f', 0))
    num_samples_a = int(params.get('num_aug_a', 0))
    print_graph_recon = int(params.get('rec_graph', 0))
    recon_iter = int(params.get('Recon_iter', 1))
    num_physical_param = int(params['num_param'])
    n_sample = int(params['n_sample'])
    param_dir = params['param_dir']
    latent_conditioner_epoch = int(params['n_epoch'])
    latent_conditioner_lr = float(params['latent_conditioner_lr'])
    latent_conditioner_batch_size = int(params['latent_conditioner_batch'])
    latent_conditioner_data_type = params['input_type']
    param_data_type = params['param_data_type']
    latent_conditioner_weight_decay = float(params.get('latent_conditioner_weight_decay', 1e-4))  # Default to 1e-4 if not specified
    latent_conditioner_dropout_rate = float(params.get('latent_conditioner_dropout_rate', 0.3))  # Default to 0.3 if not specified
    use_spatial_attention = int(params.get('use_spatial_attention', 1))  # Default to 1 (enabled)

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

    print('Dataset reange: ', np.min(new_x_train), np.max(new_x_train))
    
    # Configure augmentation parameters
    augmentation_config = {
        'noise_prob': 0.2,        # Probability of adding noise
        'noise_level': 0.03,      # Noise intensity (0.03 = 3%)
        'scaling_prob': 0.1,      # Probability of scaling
        'scaling_range': (0.9, 1.1), # Scaling factor range
        'shift_prob': 0.0,        # Probability of time shifting
        'shift_max': 0.1,         # Maximum shift fraction
        'mixup_prob': 0.2,        # Probability of applying mixup
        'mixup_alpha': 0.2,       # Mixup interpolation strength
        'cutout_prob': 0.0,       # Probability of applying cutout
        'cutout_max': 0.1,        # Maximum cutout fraction
        'enabled': True           # Master switch for augmentation
    }
    
    # Use the augmented dataset and dataloaders
    print("Creating augmented dataset with on-the-fly data augmentation...")
    
    # Add CUDA error handling for dataset creation
    try:
        # Test CUDA if available
        if torch.cuda.is_available():
            try:
                test_tensor = torch.zeros(1).cuda()
                del test_tensor
                print("CUDA is working properly")
            except RuntimeError as e:
                print(f"CUDA error during initialization: {e}")
                print("Falling back to CPU for dataset handling")
                # Force CPU mode if CUDA is not working
                device = torch.device('cpu')
        
        # Create dataloaders with error handling
        dataloader, val_dataloader = create_augmented_dataloaders(
            new_x_train, 
            batch_size=batch_size, 
            load_all=load_all,
            augmentation_config=augmentation_config,
            val_split=0.2,  # 80% train, 20% validation
            num_workers=None  # Auto-determine optimal workers
        )
        
        # Create a reference to the dataset for later use
        # This fixes the "dataset is not defined" error
        dataset = dataloader.dataset
        
        print("✓ Augmented dataloaders created successfully")
        print("  - Augmentations enabled: Noise, Scaling, Shift, Mixup, Cutout")
        print(f"  - Training samples: {int(len(new_x_train) * 0.8)}")
        print(f"  - Validation samples: {int(len(new_x_train) * 0.2)}")
        
    except Exception as e:
        print(f"Error creating augmented dataloaders: {e}")
        print("Falling back to basic dataset creation...")
        
        # Fallback to basic dataset without augmentation
        dataset = MyBaseDataset(new_x_train, load_all)
        train_dataset, validation_dataset = random_split(dataset, [int(0.8*num_param), num_param - int(0.8*num_param)])
        
        dataloader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=0, 
            pin_memory=not load_all
        )
        val_dataloader = DataLoader(
            validation_dataset, 
            batch_size=batch_size, 
            shuffle=True, 
            num_workers=0, 
            pin_memory=not load_all
        )
        print("✓ Basic dataloaders created as fallback")

    print('Dataloader initiated...')
    print_gpu_mem_checkpoint('After dataloader creation')
    # Estimate and print input data GPU memory usage
    def get_tensor_mem_mb(tensor):
        if isinstance(tensor, torch.Tensor):
            return tensor.element_size() * tensor.nelement() / 1024**2
        elif isinstance(tensor, np.ndarray):
            return tensor.nbytes / 1024**2
        else:
            return 0

    # new_x_train is a numpy array
    print(f"[INFO] new_x_train shape: {new_x_train.shape}, dtype: {new_x_train.dtype}, estimated GPU memory: {get_tensor_mem_mb(new_x_train):.2f} MB")
    print_gpu_mem_checkpoint('Before training starts (after data and dataloader setup)')





    # VAE training
    if train_latent_conditioner_only ==0:

        VAE_loss, reconstruction_error, KL_divergence, loss_val_print = train(n_epochs, batch_size, dataloader, val_dataloader, LR, num_filter_enc, num_filter_dec, num_node, latent_dim_end, latent_dim, num_time, alpha, loss, small, load_all)
        del dataloader, val_dataloader

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        VAE_trained = torch.load('./model_save/SimulGen-VAE', map_location= device, weights_only=False)
        VAE = VAE_trained.eval()

        from modules.decoder import reparameterize

        latent_vectors = np.zeros([num_param, latent_dim_end])
        gen_x_node = np.zeros([1, num_node, num_time])
        loss_total = 0
        loss_save = np.zeros([num_var])
        reconstruction_loss = np.zeros([num_param])
        hierarchical_latent_vectors = np.zeros([num_param, len(num_filter_enc)-1, latent_dim])
        reconstructed = np.empty([num_param, num_node, num_time])
        # Optimize reconstruction DataLoader with intelligent worker detection  
        recon_optimal_workers = 0 if len(dataset) < 1000 else min(2, torch.multiprocessing.cpu_count())
        
        # For the whole dataset
        dataloader2 = DataLoader(
            new_x_train, 
            batch_size=1, 
            shuffle=False, 
            num_workers=recon_optimal_workers,
            pin_memory=True if not load_all and recon_optimal_workers > 0 else False, 
            drop_last=False,
            persistent_workers=True if recon_optimal_workers > 0 else False,
            prefetch_factor=2 if recon_optimal_workers > 0 else None
        )

        for j, image in enumerate(dataloader2):
            loss_save[:]=100
            x = image.to(device)
            del image

            mu, log_var, xs = VAE.encoder(x)
            for i in range(recon_iter):
                std = torch.exp(0.5*log_var)
                latent_vector = reparameterize(mu, std)

                gen_x, _ = VAE.decoder(latent_vector, xs, mode='random')
                gen_x_np = gen_x.cpu().detach().numpy()

                loss = nn.MSELoss()(gen_x, x)

                if loss<loss_save[0]:
                    loss_save[0] = loss
                    latent_vector_save = latent_vector
                    latent_vectors[j,:] = latent_vector_save[0,:].cpu().detach().numpy()

                    for k in range(len(xs)):
                        hierarchical_latent_vectors[j,k,:] = xs[k].cpu().detach().numpy()[0]

                    reconstruction_loss[j] = loss

                    reconstructed[j,:,:] = gen_x_np[0,:,:]

                    del latent_vector, x, mu, log_var, xs, std, gen_x, gen_x_np, latent_vector_save

            print('parameter {} is finished''-''MSE: {:.4E}'.format(j+1, loss))
            print('')
            loss_total = loss_total+loss.cpu().detach().numpy()

            del loss

        print('')

        print('Total MSE loss: {:.3e}'.format(loss_total/num_param))

        np.save('model_save/latent_vectors', latent_vectors)
        np.save('model_save/xs', hierarchical_latent_vectors)

        temp5 = './SimulGen-VAE_L2_loss.txt'
        np.savetxt(temp5, reconstruction_loss, fmt = '%e')

        if print_graph_recon == 1:
            print('Printing graph...')
            plt.semilogy(VAE_loss, label = 'VAE')
            plt.semilogy(loss_val_print, label = 'Validation')
            plt.legend()
            plt.show()

            plt.figure()
            plt.plot(reconstruction_loss, label = 'Reconstruction')
            plt.legend()
            plt.show()

            plt.figure()
            param_No = 0
            plt.title('Nodal data')
            plt.plot(dataset[param_No,:,0].cpu().detach().numpy(), marker = 'o', label = 'Original')
            plt.plot(reconstructed[param_No,:,0], marker = 'o', label = 'Reconstructed')
            plt.legend()
            plt.show()

            plt.figure()
            param_No = 10
            plt.title('Nodal data')
            plt.plot(dataset[param_No,:,0].cpu().detach().numpy(), marker = 'o', label = 'Original')
            plt.plot(reconstructed[param_No,:,0], marker = 'o', label = 'Reconstructed')
            plt.legend()
            plt.show()

            plt.figure()
            param_No = 0
            node_No = 900
            plt.title('Temporal data')
            plt.plot(dataset[param_No,node_No,:].cpu().detach().numpy(), marker = 'o', label = 'Original')
            plt.plot(reconstructed[param_No,node_No,:], marker = 'o', label = 'Reconstructed')
            plt.legend()
            plt.show()

            plt.figure()
            param_No = 0
            node_No = 800
            plt.title('Temporal data')
            plt.plot(dataset[param_No,node_No,:].cpu().detach().numpy(), marker = 'o', label = 'Original')
            plt.plot(reconstructed[param_No,node_No,:], marker = 'o', label = 'Reconstructed')
            plt.legend()
            plt.show()

            plt.figure()
            param_No = 0
            node_No = 700
            plt.title('Parametric data')
            plt.plot(dataset[param_No, node_No, :].cpu().detach().numpy(), marker = 'o', label = 'Original')
            plt.plot(reconstructed[param_No, node_No, :], marker = 'o', label = 'Reconstructed')
            plt.plot(reconstructed[param_No+1, node_No, :], marker = 'o', label = 'Reconstructed')
            plt.plot(reconstructed[param_No+2, node_No, :], marker = 'o', label = 'Reconstructed')
            plt.plot(reconstructed[param_No+3, node_No, :], marker = 'o', label = 'Reconstructed')
            plt.legend()
            plt.show()
    elif train_latent_conditioner_only == 1:
        print('Training LatentConditioner only...')
        latent_vectors = np.load('model_save/latent_vectors.npy')
        hierarchical_latent_vectors = np.load('model_save/xs.npy')
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        VAE_trained = torch.load('model_save/SimulGen-VAE', map_location= device, weights_only=False)
        VAE = VAE_trained.eval()

    # LatentConditioner training (runs for both train_latent_conditioner_only == 0 and train_latent_conditioner_only == 1)
    out_latent_vectors = latent_vectors.reshape([num_param, latent_dim_end])
    xs_vectors = hierarchical_latent_vectors.reshape([num_param, -1])


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

    # Debug: Print shapes before scaling
    print(f"LatentConditioner scaling - physical_param_input shape: {physical_param_input.shape}")
    print(f"LatentConditioner scaling - out_latent_vectors shape: {out_latent_vectors.shape}")
    print(f"LatentConditioner scaling - xs_vectors shape: {xs_vectors.shape}")

    physical_param_input, param_input_scaler = latent_conditioner_scaler(physical_param_input, './model_save/latent_conditioner_input_scaler.pkl')
    out_latent_vectors, latent_vectors_scaler = latent_conditioner_scaler(out_latent_vectors, './model_save/latent_vectors_scaler.pkl')
    out_hierarchical_latent_vectors, xs_scaler = latent_conditioner_scaler(xs_vectors, './model_save/xs_scaler.pkl')

    print('LatentConditioner data loaded...')
    print('LatentConditioner scale: ')
    print(f'Input: {np.max(physical_param_input)} {np.min(physical_param_input)}')
    print(f'Main latent: {np.max(out_latent_vectors)} {np.min(out_latent_vectors)}')
    print(f'Hierarchical latent: {np.max(out_hierarchical_latent_vectors)} {np.min(out_hierarchical_latent_vectors)}')
    out_hierarchical_latent_vectors = out_hierarchical_latent_vectors.reshape([num_param, len(num_filter_enc)-1, latent_dim])

    # Debug: Print shapes of LatentConditioner dataset inputs
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
    print(f"LatentConditioner dataset inputs:")
    print(f"  - physical_param_input shape: {physical_param_input.shape}")
    print(f"  - out_latent_vectors shape: {out_latent_vectors.shape}")
    print(f"  - out_hierarchical_latent_vectors shape: {out_hierarchical_latent_vectors.shape}")
    
    # Get actual dataset size and calculate split sizes
    latent_conditioner_dataset_size = len(latent_conditioner_dataset)
    train_size = int(0.8 * latent_conditioner_dataset_size)
    val_size = latent_conditioner_dataset_size - train_size
    
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
    latent_conditioner_optimal_workers = 0#get_optimal_workers(latent_conditioner_dataset_size, False, latent_conditioner_batch_size)  # LatentConditioner data is not load_all
    
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
        print(f"   ✓ LatentConditioner DataLoader: {latent_conditioner_optimal_workers} workers ({latent_conditioner_dataset_size} samples)")

    size2 = len(num_filter_enc)-1


    # Get optimal device with comprehensive debugging
    print("\n🔍 === LATENT CONDITIONER INITIALIZATION DEBUGGING ===")
    device = safe_cuda_initialization()  # Get safe device
    print(f"Selected device: {device}")
    print("===================================================\n")
    
    if latent_conditioner_data_type=='image':
        try:
            print("Initializing LatentConditioner CNN image model...")
            latent_conditioner = LatentConditionerImg(latent_conditioner_filter, latent_dim_end, input_shape, latent_dim, size2, latent_conditioner_data_shape, dropout_rate=latent_conditioner_dropout_rate, use_attention=bool(use_spatial_attention)).to(device)
            print(f"✓ CNN model successfully initialized on {device}")
        except RuntimeError as e:
            print(f"❌ Error initializing LatentConditioner CNN image model: {e}")
            print("   Falling back to CPU-only model")
            device = torch.device('cpu')
            latent_conditioner = LatentConditionerImg(latent_conditioner_filter, latent_dim_end, input_shape, latent_dim, size2, latent_conditioner_data_shape, dropout_rate=latent_conditioner_dropout_rate, use_attention=bool(use_spatial_attention)).to(device)
            print(f"✓ CNN model fallback initialized on {device}")
    elif latent_conditioner_data_type=='image_vit':
        try:
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
            print(f"✓ ViT model fallback initialized on {device}")
    elif latent_conditioner_data_type=='csv':
        try:
            print("Initializing LatentConditioner MLP CSV model...")
            latent_conditioner = LatentConditioner(latent_conditioner_filter, latent_dim_end, input_shape, latent_dim, size2, dropout_rate=latent_conditioner_dropout_rate).to(device)
            print(f"✓ MLP model successfully initialized on {device}")
        except RuntimeError as e:
            print(f"❌ Error initializing LatentConditioner MLP CSV model: {e}")
            print("   Falling back to CPU-only model")
            device = torch.device('cpu')
            latent_conditioner = LatentConditioner(latent_conditioner_filter, latent_dim_end, input_shape, latent_dim, size2, dropout_rate=latent_conditioner_dropout_rate).to(device)
            print(f"✓ MLP model fallback initialized on {device}")
    else:
        raise NotImplementedError(f'Unrecognized latent_conditioner_data_type: {latent_conditioner_data_type}. Supported options: "image" (CNN), "image_vit" (ViT), "csv" (MLP)')

    print(latent_conditioner)

    # Print CUDA diagnostic information before LatentConditioner training
    if torch.cuda.is_available():
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
        LatentConditioner_loss = train_latent_conditioner(latent_conditioner_epoch, latent_conditioner_dataloader, latent_conditioner_validation_dataloader, latent_conditioner, latent_conditioner_lr, weight_decay=latent_conditioner_weight_decay, is_image_data=image, image_size=256)
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
    print(f"\n🔍 Pre-evaluation device check:")
    print(f"  Model device before: {next(latent_conditioner.parameters()).device}")
    
    eval_device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"  Target evaluation device: {eval_device}")
    
    try:
        latent_conditioner = latent_conditioner.to(eval_device)
        print(f"  ✓ Model moved to: {next(latent_conditioner.parameters()).device}")
    except Exception as e:
        print(f"  ❌ Failed to move model: {e}")
        eval_device = next(latent_conditioner.parameters()).device
        print(f"  Using current device: {eval_device}")
    
    device = eval_device  # Update device variable for consistency

    if VAE in globals():
        del VAE

    VAE_trained = torch.load('model_save/SimulGen-VAE', map_location= device, weights_only=False)
    VAE = VAE_trained.eval()

    # Optimize LatentConditioner evaluation DataLoader with intelligent worker detection
    latent_conditioner_eval_optimal_workers = 0 if len(latent_conditioner_dataset) < 1000 else min(2, torch.multiprocessing.cpu_count())
    
    latent_conditioner_dataloader_eval = torch.utils.data.DataLoader(
        latent_conditioner_dataset, 
        batch_size=1, 
        shuffle=False, 
        num_workers=latent_conditioner_eval_optimal_workers,
        pin_memory=True if latent_conditioner_eval_optimal_workers > 0 else False,
        persistent_workers=True if latent_conditioner_eval_optimal_workers > 0 else False,
        prefetch_factor=2 if latent_conditioner_eval_optimal_workers > 0 else None
    )

    for i, (x, y1, y2) in enumerate(latent_conditioner_dataloader_eval):
        x = x.to(device)

        y_pred1, y_pred2 = latent_conditioner(x)
        y_pred1 = y_pred1.cpu().detach().numpy()
        y1 = y1.cpu().detach().numpy()
        y_pred2 = y_pred2.cpu().detach().numpy()
        y2 = y2.cpu().detach().numpy()
        A = y2

        y2 = y2.reshape([1, -1])
        latent_predict = latent_vectors_scaler.inverse_transform(y1)
        xs_predict = xs_scaler.inverse_transform(y2)
        xs_predict = xs_predict.reshape([-1, 1, A.shape[-1]])
        latent_predict = torch.from_numpy(latent_predict)
        xs_predict = torch.from_numpy(xs_predict)
        xs_predict = xs_predict.to(device)
        xs_predict = list(xs_predict)
        latent_predict = latent_predict.to(device)
        target_output, _ = VAE.decoder(latent_predict, xs_predict, mode='fix')
        target_output_np = target_output.cpu().detach().numpy()
        target_output_np = target_output_np.swapaxes(1,2)
        target_output_np = target_output_np.reshape((-1, num_node))
        target_output_np = np.reshape(target_output_np, [num_time, num_node, 1])

        plt.figure()
        plt.title('Main latent')
        plt.plot(y1[0,:], '*', label = 'True')
        plt.plot(y_pred1[0,:], 'o', label = 'Predicted')
        plt.legend()

        plt.figure()
        plt.title('Hierarchical latent')
        plt.plot(y2[0,:], '*', label = 'True')
        plt.plot(y_pred2[0,0, :], 'o', label = 'Predicted')
        plt.legend()
        
        plt.figure()
        plt.title('Reconstruction')
        plt.plot(target_output_np[0,:,0], '.', label = 'Recon')
        plt.plot(new_x_train[0, :, int(num_time/2)], '.', label = 'True')
        plt.legend()

        plt.show()


if __name__ == "__main__":
    main()
