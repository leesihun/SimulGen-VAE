#!/usr/bin/env python3
"""Reconstruction Evaluation Module for SimulGenVAE

This module handles reconstruction evaluation, comparison plotting, and validation
of VAE and LatentConditioner outputs.

Author: SiHun Lee, Ph.D.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
from modules.decoder import reparameterize


class ReconstructionEvaluator:
    """Handles reconstruction evaluation and comparison plotting."""
    
    def __init__(self, VAE, device, num_time, debug_mode=0):
        """Initialize the reconstruction evaluator.
        
        Args:
            VAE: Trained VAE model
            device: torch device for computations
            num_time: Number of time steps
            debug_mode: Debug verbosity level
        """
        self.VAE = VAE
        self.device = device
        self.num_time = num_time
        self.debug_mode = debug_mode
        
    def evaluate_reconstruction_comparison(self, latent_conditioner, latent_conditioner_dataset, 
                                         original_data, latent_vectors_scaler, xs_scaler):
        """Compare VAE+LatentConditioner vs VAE-only reconstructions.
        
        Args:
            latent_conditioner: Trained latent conditioner model
            latent_conditioner_dataset: Dataset for latent conditioner evaluation
            original_data: Original training data (new_x_train)
            latent_vectors_scaler: Scaler for main latent vectors
            xs_scaler: Scaler for hierarchical latent vectors
        """
        # Create unshuffled dataloader for consistent comparison
        dataloader_test = torch.utils.data.DataLoader(
            latent_conditioner_dataset, 
            batch_size=1, 
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )
        
        # Create corresponding original data loader (unshuffled)
        original_dataset = torch.utils.data.TensorDataset(torch.from_numpy(original_data))
        original_dataloader = torch.utils.data.DataLoader(
            original_dataset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=False
        )
        
        if self.debug_mode >= 1:
            print(f"Evaluating {len(latent_conditioner_dataset)} samples...")
        
        for i, ((x_lc, y1_true, y2_true), (x_orig,)) in enumerate(zip(dataloader_test, original_dataloader)):
            if i >= 5:  # Limit to first 5 samples for visualization
                break
                
            # Move to device
            x_lc = x_lc.to(self.device)
            x_orig = x_orig.to(self.device)
            
            # Get predictions from latent conditioner
            y_pred1, y_pred2 = latent_conditioner(x_lc)
            
            # Convert to numpy for inverse scaling
            y_pred1_np = y_pred1.cpu().detach().numpy()
            y_pred2_np = y_pred2.cpu().detach().numpy()
            y1_true_np = y1_true.cpu().detach().numpy()
            y2_true_np = y2_true.cpu().detach().numpy()
            
            # Reconstruct using predicted latents (VAE+LatentConditioner)
            target_output_predicted = self._reconstruct_from_latents(
                y_pred1_np, y_pred2_np, latent_vectors_scaler, xs_scaler
            )
            
            # Reconstruct using true latents (VAE-only, should match original)
            target_output_true = self._reconstruct_from_latents(
                y1_true_np, y2_true_np, latent_vectors_scaler, xs_scaler
            )
            
            # Get original data for comparison
            original_data_sample = x_orig.cpu().detach().numpy()
            
            # Generate comparison plots
            self._plot_reconstruction_comparison(
                i, original_data_sample, target_output_predicted, 
                target_output_true, save_plots=True
            )
            
            if self.debug_mode >= 1:
                self._print_reconstruction_stats(
                    i, original_data_sample, target_output_predicted, target_output_true
                )
    
    def _reconstruct_from_latents(self, y_pred, y2_pred, latent_scaler, xs_scaler):
        """Reconstruct data from latent vectors."""
        # Inverse transform the predictions
        latent_predict = latent_scaler.inverse_transform(y_pred)
        xs_predict = xs_scaler.inverse_transform(y2_pred.reshape([1, -1]))
        xs_predict = xs_predict.reshape([-1, 1, y2_pred.shape[-1]])
        
        # Convert to tensors and move to device
        latent_predict = torch.from_numpy(latent_predict).to(self.device)
        xs_predict = torch.from_numpy(xs_predict).to(self.device)
        xs_predict = list(xs_predict)
        
        # Reconstruct using VAE decoder
        with torch.no_grad():
            target_output, _ = self.VAE.decoder(latent_predict, xs_predict, mode='fix')
            target_output_np = target_output.cpu().detach().numpy()
            target_output_np = target_output_np.swapaxes(1, 2)
        
        return target_output_np
    
    def _plot_reconstruction_comparison(self, sample_idx, original, predicted, true_recon, save_plots=False):
        """Plot comparison between original, predicted, and true reconstructions."""
        time_idx = int(self.num_time / 2)
        
        # Extract middle time slice and scale for visualization
        original_slice = original[0, :, time_idx] * 1e6
        predicted_slice = predicted[0, time_idx, :] * 1e6
        true_recon_slice = true_recon[0, time_idx, :] * 1e6
        
        plt.figure(figsize=(12, 6))
        
        # Plot comparison
        plt.title(f'Sample {sample_idx} - Reconstruction Comparison (t={time_idx})')
        plt.plot(original_slice, '.', label=f'Original [{original_slice.min():.1f}, {original_slice.max():.1f}]', alpha=0.8)
        plt.plot(predicted_slice, '.', label=f'VAE+LatentConditioner [{predicted_slice.min():.1f}, {predicted_slice.max():.1f}]', alpha=0.8)
        plt.plot(true_recon_slice, '.', label=f'VAE-only [{true_recon_slice.min():.1f}, {true_recon_slice.max():.1f}]', alpha=0.8)
        
        plt.xlabel('Node Index')
        plt.ylabel('Value (×1e6)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_plots:
            os.makedirs('checkpoints', exist_ok=True)
            plt.savefig(f'checkpoints/reconstruction_comparison_{sample_idx}.png', dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def _print_reconstruction_stats(self, sample_idx, original, predicted, true_recon):
        """Print reconstruction statistics for debugging."""
        time_idx = int(self.num_time / 2)
        
        original_slice = original[0, :, time_idx]
        predicted_slice = predicted[0, time_idx, :]
        true_recon_slice = true_recon[0, time_idx, :]
        
        # Calculate errors
        pred_error = np.mean((original_slice - predicted_slice) ** 2)
        true_error = np.mean((original_slice - true_recon_slice) ** 2)
        
        print(f"Sample {sample_idx} Reconstruction Stats:")
        print(f"  Original range: [{original_slice.min():.3e}, {original_slice.max():.3e}]")
        print(f"  VAE+LC MSE: {pred_error:.3e}")
        print(f"  VAE-only MSE: {true_error:.3e}")
        print(f"  VAE-only should be ≈0 (got {true_error:.1e})")
        print("")