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
        """Plot comparison between original, predicted, and true reconstructions with both temporal and nodal views."""
        # Create 2x2 subplot layout
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(f'Sample {sample_idx} - Dual-View Reconstruction Comparison', fontsize=16)
        
        # Get data dimensions
        num_nodes = original.shape[1]
        
        # Select representative indices for visualization
        time_indices = [int(self.num_time * 0.25), int(self.num_time * 0.5), int(self.num_time * 0.75)]
        node_indices = [int(num_nodes * 0.25), int(num_nodes * 0.5), int(num_nodes * 0.75)]
        
        # === NODAL VIEW (Top Row) ===
        # Top Left: Nodal distribution at middle timestep
        time_idx = int(self.num_time / 2)
        original_nodal = original[0, :, time_idx] * 1e6
        predicted_nodal = predicted[0, time_idx, :] * 1e6
        true_recon_nodal = true_recon[0, time_idx, :] * 1e6
        
        axes[0, 0].set_title(f'Nodal View - Spatial Distribution (t={time_idx})')
        axes[0, 0].plot(original_nodal, '.', label=f'Original [{original_nodal.min():.1f}, {original_nodal.max():.1f}]', alpha=0.8, markersize=1)
        axes[0, 0].plot(predicted_nodal, '.', label=f'VAE+LC [{predicted_nodal.min():.1f}, {predicted_nodal.max():.1f}]', alpha=0.8, markersize=1)
        axes[0, 0].plot(true_recon_nodal, '.', label=f'VAE-only [{true_recon_nodal.min():.1f}, {true_recon_nodal.max():.1f}]', alpha=0.8, markersize=1)
        axes[0, 0].set_xlabel('Node Index')
        axes[0, 0].set_ylabel('Value (×1e6)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Top Right: Multiple nodal snapshots
        axes[0, 1].set_title('Nodal View - Multiple Time Snapshots')
        colors = ['blue', 'green', 'red']
        for i, t_idx in enumerate(time_indices):
            orig_snap = original[0, :, t_idx] * 1e6
            pred_snap = predicted[0, t_idx, :] * 1e6
            axes[0, 1].plot(orig_snap, '--', color=colors[i], alpha=0.7, linewidth=1, label=f'Original t={t_idx}')
            axes[0, 1].plot(pred_snap, '-', color=colors[i], alpha=0.8, linewidth=1, label=f'VAE+LC t={t_idx}')
        axes[0, 1].set_xlabel('Node Index')
        axes[0, 1].set_ylabel('Value (×1e6)')
        axes[0, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0, 1].grid(True, alpha=0.3)
        
        # === TEMPORAL VIEW (Bottom Row) ===
        # Bottom Left: Temporal evolution at middle node
        node_idx = int(num_nodes / 2)
        original_temporal = original[0, node_idx, :] * 1e6
        predicted_temporal = predicted[0, :, node_idx] * 1e6
        true_recon_temporal = true_recon[0, :, node_idx] * 1e6
        
        axes[1, 0].set_title(f'Temporal View - Time Evolution (node={node_idx})')
        axes[1, 0].plot(original_temporal, '-', label=f'Original [{original_temporal.min():.1f}, {original_temporal.max():.1f}]', alpha=0.8)
        axes[1, 0].plot(predicted_temporal, '-', label=f'VAE+LC [{predicted_temporal.min():.1f}, {predicted_temporal.max():.1f}]', alpha=0.8)
        axes[1, 0].plot(true_recon_temporal, '-', label=f'VAE-only [{true_recon_temporal.min():.1f}, {true_recon_temporal.max():.1f}]', alpha=0.8)
        axes[1, 0].set_xlabel('Time Index')
        axes[1, 0].set_ylabel('Value (×1e6)')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Bottom Right: Multiple temporal traces at different nodes
        axes[1, 1].set_title('Temporal View - Multiple Node Traces')
        for i, n_idx in enumerate(node_indices):
            orig_trace = original[0, n_idx, :] * 1e6
            pred_trace = predicted[0, :, n_idx] * 1e6
            axes[1, 1].plot(orig_trace, '--', color=colors[i], alpha=0.7, linewidth=1, label=f'Original n={n_idx}')
            axes[1, 1].plot(pred_trace, '-', color=colors[i], alpha=0.8, linewidth=1, label=f'VAE+LC n={n_idx}')
        axes[1, 1].set_xlabel('Time Index')
        axes[1, 1].set_ylabel('Value (×1e6)')
        axes[1, 1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            os.makedirs('checkpoints', exist_ok=True)
            plt.savefig(f'checkpoints/reconstruction_dual_view_{sample_idx}.png', dpi=300, bbox_inches='tight')
        
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