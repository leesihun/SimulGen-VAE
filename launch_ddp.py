#!/usr/bin/env python3
"""
DDP Launcher Script for SimulGen-VAE

This script makes it easy to launch distributed training with different configurations.
"""

import os
import sys
import subprocess
import argparse


def main():
    parser = argparse.ArgumentParser(description='Launch SimulGen-VAE DDP Training')
    parser.add_argument('--gpus', type=int, default=2, 
                       help='Number of GPUs to use (default: 2)')
    parser.add_argument('--preset', type=int, default=1, choices=[1, 2, 3, 4, 5],
                       help='Configuration preset (default: 1)')
    parser.add_argument('--plot', type=int, default=2, choices=[0, 1, 2],
                       help='Plotting option (default: 2)')
    parser.add_argument('--size', type=str, default='small', choices=['small', 'big'],
                       help='Model size (default: small)')
    parser.add_argument('--load_all', type=int, default=1, choices=[0, 1],
                       help='Load all data to memory (default: 1)')
    parser.add_argument('--train_latent_conditioner_only', type=int, default=0, choices=[0, 1],
                       help='Train only latent conditioner (default: 0)')
    parser.add_argument('--master_port', type=int, default=29500,
                       help='Master port for DDP communication (default: 29500)')
    
    args = parser.parse_args()
    
    # Check if CUDA is available
    try:
        import torch
        if not torch.cuda.is_available():
            print("CUDA not available. DDP training requires CUDA.")
            sys.exit(1)
        
        available_gpus = torch.cuda.device_count()
        if args.gpus > available_gpus:
            print(f"Requested {args.gpus} GPUs but only {available_gpus} available.")
            print(f"Using {available_gpus} GPUs instead.")
            args.gpus = available_gpus
            
    except ImportError:
        print("PyTorch not found. Please install PyTorch with CUDA support.")
        sys.exit(1)
    
    # Build the torchrun command
    cmd = [
        'torchrun',
        '--nproc_per_node', str(args.gpus),
        '--master_port', str(args.master_port),
        'SimulGen-VAE.py',
        '--use_ddp',
        '--preset', str(args.preset),
        '--plot', str(args.plot),
        '--size', args.size,
        '--load_all', str(args.load_all),
        '--train_latent_conditioner_only', str(args.train_latent_conditioner_only)
    ]
    
    print("="*60)
    print("SimulGen-VAE Distributed Training Launcher")
    print("="*60)
    print(f"GPUs: {args.gpus}")
    print(f"Preset: {args.preset}")
    print(f"Model size: {args.size}")
    print(f"Load all data: {bool(args.load_all)}")
    print(f"Train LC only: {bool(args.train_latent_conditioner_only)}")
    print(f"Master port: {args.master_port}")
    print("="*60)
    print("Command:")
    print(" ".join(cmd))
    print("="*60)
    
    # Ask for confirmation
    response = input("Launch training? [y/N]: ").strip().lower()
    if response not in ['y', 'yes']:
        print("Training cancelled.")
        sys.exit(0)
    
    # Set environment variables for better performance
    env = os.environ.copy()
    env['CUDA_LAUNCH_BLOCKING'] = '0'  # Allow async CUDA operations
    env['TORCH_CUDNN_V8_API_ENABLED'] = '1'  # Enable cuDNN v8 API
    env['NCCL_ASYNC_ERROR_HANDLING'] = '1'  # Better error handling for NCCL
    
    # Launch training
    try:
        print("Starting distributed training...")
        result = subprocess.run(cmd, env=env, check=True)
        print("\nTraining completed successfully!")
        
    except subprocess.CalledProcessError as e:
        print(f"\nTraining failed with exit code {e.returncode}")
        print("Check the error messages above for details.")
        sys.exit(e.returncode)
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user.")
        sys.exit(1)


if __name__ == "__main__":
    main()