#!/usr/bin/env python3
"""
Training Acceleration Script for SimulGen-VAE

This script demonstrates how to apply the optimizations for your specific case:
- Large batches per iteration
- Small dataset variety
- Focus on maximum training speed

Usage:
    python accelerate_training.py --scenario maximum_speed
    python accelerate_training.py --scenario small_variety_large_batch
"""

import argparse
import torch
from optimization_config import get_config, print_optimization_summary

def main():
    parser = argparse.ArgumentParser(description='Accelerated SimulGen-VAE Training')
    parser.add_argument('--scenario', default='maximum_speed', 
                       choices=['small_variety_large_batch', 'large_variety_small_batch', 
                               'memory_constrained', 'maximum_speed', 'safe_mode'],
                       help='Optimization scenario to use')
    parser.add_argument('--preset', default='1', help='Preset file to use')
    parser.add_argument('--plot', default='2', help='Plot setting')
    parser.add_argument('--train_pinn_only', default='0', help='Train PINN only')
    parser.add_argument('--size', default='small', choices=['small', 'large'], help='Model size')
    
    args = parser.parse_args()
    
    # Get optimization configuration
    config = get_config(args.scenario)
    print_optimization_summary(config)
    
    # Apply optimizations to load_all setting
    load_all = 1 if config['load_all'] else 0
    
    # Construct the optimized command
    cmd_args = [
        f"--preset={args.preset}",
        f"--plot={args.plot}",
        f"--train_pinn_only={args.train_pinn_only}",
        f"--size={args.size}",
        f"--load_all={load_all}"
    ]
    
    # Print the optimized command
    print("🚀 OPTIMIZED COMMAND:")
    print("="*60)
    print(f"python SimulGen-VAE.py {' '.join(cmd_args)}")
    print("="*60)
    
    # Print specific recommendations for your case
    print("\n📋 SPECIFIC RECOMMENDATIONS FOR YOUR CASE:")
    print("-" * 50)
    
    if args.scenario in ['maximum_speed', 'small_variety_large_batch']:
        print("✅ Perfect choice for your scenario!")
        print("   • All data preloaded to GPU (--load_all=1)")
        print("   • Model compilation enabled for consistent shapes")
        print("   • Single-threaded data loading (no overhead)")
        print("   • Aggressive memory usage")
        print("   • Mixed precision training")
        
        # Additional tips
        print("\n💡 ADDITIONAL TIPS:")
        print("   • Monitor GPU memory usage with nvidia-smi")
        print("   • If you get OOM errors, reduce batch size in condition.txt")
        print("   • Consider increasing batch size if you have extra GPU memory")
        print("   • The first epoch may be slower due to model compilation")
    
    print("\n🔧 MANUAL OPTIMIZATIONS YOU CAN APPLY:")
    print("   1. Increase batch size in input_data/condition.txt if GPU memory allows")
    print("   2. Reduce validation frequency (change epoch % 100 to epoch % 200)")
    print("   3. Disable TensorBoard logging for maximum speed")
    print("   4. Use --size=small for faster training if accuracy allows")
    
    print("\n⚡ EXPECTED SPEEDUP:")
    print("   • 2-4x faster training with these optimizations")
    print("   • Varies based on GPU and dataset size")
    print("   • Most gains from GPU preloading and model compilation")

if __name__ == "__main__":
    main() 