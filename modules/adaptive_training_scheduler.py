"""
Adaptive Training Scheduler for Latent Conditioner E2E Training

This module provides advanced scheduling strategies to prevent loss stagnation
and improve convergence in end-to-end training scenarios.
"""

import torch
import torch.nn as nn
import math
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

class AdaptiveLearningRateScheduler:
    """
    Adaptive learning rate scheduler that monitors loss plateaus and adjusts accordingly.
    """
    
    def __init__(self, optimizer, initial_lr=0.001, patience=15, factor=0.7, 
                 min_lr=1e-7, plateau_threshold=1e-5, warmup_epochs=20):
        self.optimizer = optimizer
        self.initial_lr = initial_lr
        self.current_lr = initial_lr
        self.patience = patience
        self.factor = factor
        self.min_lr = min_lr
        self.plateau_threshold = plateau_threshold
        self.warmup_epochs = warmup_epochs
        
        # Loss tracking for plateau detection
        self.loss_history = deque(maxlen=patience + 5)
        self.best_loss = float('inf')
        self.plateau_count = 0
        self.epoch = 0
        
        # Warmup scheduler
        self.warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, total_iters=warmup_epochs
        )
        
        print(f"🔧 Adaptive LR Scheduler initialized:")
        print(f"   Initial LR: {initial_lr:.2e}, Min LR: {min_lr:.2e}")
        print(f"   Plateau patience: {patience}, Reduction factor: {factor}")
        print(f"   Warmup epochs: {warmup_epochs}")
    
    def step(self, loss):
        """Update learning rate based on loss progression."""
        self.epoch += 1
        self.loss_history.append(loss)
        
        # Warmup phase
        if self.epoch <= self.warmup_epochs:
            self.warmup_scheduler.step()
            self.current_lr = self.optimizer.param_groups[0]['lr']
            return self.current_lr
        
        # Check for improvement
        if loss < self.best_loss - self.plateau_threshold:
            self.best_loss = loss
            self.plateau_count = 0
        else:
            self.plateau_count += 1
        
        # Reduce LR on plateau
        if self.plateau_count >= self.patience:
            new_lr = max(self.current_lr * self.factor, self.min_lr)
            if new_lr < self.current_lr:
                print(f"📉 Plateau detected! Reducing LR: {self.current_lr:.2e} → {new_lr:.2e}")
                for param_group in self.optimizer.param_groups:
                    param_group['lr'] = new_lr
                self.current_lr = new_lr
                self.plateau_count = 0
        
        return self.current_lr
    
    def get_lr(self):
        return self.current_lr

class AdaptiveRegularizationScheduler:
    """
    Adaptive regularization weight scheduler that maintains guidance longer.
    """
    
    def __init__(self, initial_weight=0.001, total_epochs=5000, warmup_epochs=20, 
                 plateau_boost=2.0, min_weight_ratio=0.1):
        self.initial_weight = initial_weight
        self.total_epochs = total_epochs
        self.warmup_epochs = warmup_epochs
        self.plateau_boost = plateau_boost
        self.min_weight = initial_weight * min_weight_ratio
        
        # Loss tracking for adaptive adjustment
        self.loss_history = deque(maxlen=20)
        self.plateau_detected = False
        
        print(f"🎯 Adaptive Regularization Scheduler:")
        print(f"   Initial weight: {initial_weight:.6f}, Min weight: {self.min_weight:.6f}")
        print(f"   Plateau boost factor: {plateau_boost}x")
    
    def get_weight(self, epoch, current_loss=None):
        """Get regularization weight with adaptive adjustment."""
        
        # Track loss for plateau detection
        if current_loss is not None:
            self.loss_history.append(current_loss)
            
            # Simple plateau detection
            if len(self.loss_history) >= 15:
                recent_losses = list(self.loss_history)[-10:]
                improvement = (max(recent_losses) - min(recent_losses)) / max(recent_losses)
                self.plateau_detected = improvement < 0.01  # Less than 1% improvement
        
        if epoch < self.warmup_epochs:
            # Full regularization during warmup
            weight = self.initial_weight
        else:
            # Slower decay with plateau boost
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            
            # Use sigmoid decay instead of exponential for smoother transition
            decay_factor = 1 / (1 + math.exp(5 * (progress - 0.7)))  # Sigmoid centered at 70% progress
            weight = self.min_weight + (self.initial_weight - self.min_weight) * decay_factor
            
            # Boost regularization if plateau detected
            if self.plateau_detected and epoch > 50:
                weight *= self.plateau_boost
                weight = min(weight, self.initial_weight)  # Cap at initial weight
        
        return weight

class GradientHealthMonitor:
    """
    Monitor gradient health and provide adaptive clipping.
    """
    
    def __init__(self, model, base_clip_norm=5.0, adaptive_factor=2.0):
        self.model = model
        self.base_clip_norm = base_clip_norm
        self.adaptive_factor = adaptive_factor
        
        # Gradient statistics tracking
        self.grad_norms = deque(maxlen=100)
        self.grad_variance = deque(maxlen=50)
        
    def clip_and_monitor(self, epoch):
        """Adaptive gradient clipping based on training progress."""
        
        # Calculate current gradient norm
        total_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), max_norm=float('inf'), norm_type=2
        )
        
        self.grad_norms.append(total_norm.item())
        
        # Calculate adaptive clip norm
        if len(self.grad_norms) >= 10:
            recent_norms = list(self.grad_norms)[-10:]
            mean_norm = np.mean(recent_norms)
            std_norm = np.std(recent_norms)
            
            # Adaptive clipping: allow higher norms early in training
            progress_factor = min(epoch / 100, 1.0)  # Gradually become more restrictive
            adaptive_clip = self.base_clip_norm + (1 - progress_factor) * self.adaptive_factor * std_norm
            adaptive_clip = max(adaptive_clip, self.base_clip_norm)  # Never go below base
            
        else:
            adaptive_clip = self.base_clip_norm * 2  # More lenient initially
        
        # Apply adaptive clipping
        if total_norm > adaptive_clip:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=adaptive_clip)
            clipped_norm = adaptive_clip
        else:
            clipped_norm = total_norm
        
        # Monitor gradient health
        if epoch % 50 == 0 and len(self.grad_norms) >= 10:
            recent_norms = list(self.grad_norms)[-10:]
            print(f"🔍 Gradient Health (Epoch {epoch}):")
            print(f"   Current norm: {total_norm:.3f}, Clipped to: {clipped_norm:.3f}")
            print(f"   Recent avg: {np.mean(recent_norms):.3f} ± {np.std(recent_norms):.3f}")
        
        return clipped_norm

class LossStagnationDetector:
    """
    Detect and diagnose loss stagnation issues.
    """
    
    def __init__(self, patience=30, min_improvement=1e-6):
        self.patience = patience
        self.min_improvement = min_improvement
        
        self.train_losses = deque(maxlen=patience * 2)
        self.val_losses = deque(maxlen=patience * 2)
        self.best_val_loss = float('inf')
        self.stagnation_count = 0
        
    def update(self, train_loss, val_loss, epoch):
        """Update loss tracking and detect stagnation."""
        self.train_losses.append(train_loss)
        self.val_losses.append(val_loss)
        
        # Check for improvement
        if val_loss < self.best_val_loss - self.min_improvement:
            self.best_val_loss = val_loss
            self.stagnation_count = 0
        else:
            self.stagnation_count += 1
        
        # Diagnose stagnation
        if self.stagnation_count >= self.patience:
            return self._diagnose_stagnation(epoch)
        
        return None
    
    def _diagnose_stagnation(self, epoch):
        """Diagnose the cause of stagnation."""
        diagnosis = {
            'epoch': epoch,
            'stagnation_epochs': self.stagnation_count,
            'issues': []
        }
        
        if len(self.train_losses) >= 20 and len(self.val_losses) >= 20:
            recent_train = list(self.train_losses)[-10:]
            recent_val = list(self.val_losses)[-10:]
            
            train_trend = np.polyfit(range(len(recent_train)), recent_train, 1)[0]
            val_trend = np.polyfit(range(len(recent_val)), recent_val, 1)[0]
            
            # Check for overfitting
            if np.mean(recent_val) / np.mean(recent_train) > 2.0:
                diagnosis['issues'].append('overfitting')
            
            # Check for learning rate issues
            if abs(train_trend) < 1e-7:
                diagnosis['issues'].append('learning_rate_too_low')
            
            # Check for gradient issues
            if train_trend > 0:
                diagnosis['issues'].append('training_loss_increasing')
        
        return diagnosis

def create_loss_stagnation_plot(train_losses, val_losses, output_path='output/loss_analysis.png'):
    """Create diagnostic plot for loss stagnation analysis."""
    
    plt.figure(figsize=(15, 10))
    
    # Main loss plot
    plt.subplot(2, 2, 1)
    epochs = range(len(train_losses))
    plt.plot(epochs, train_losses, 'b-', label='Training Loss', alpha=0.7)
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss', alpha=0.7)
    plt.yscale('log')
    plt.xlabel('Epoch')
    plt.ylabel('Loss (log scale)')
    plt.title('Loss Progression')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Loss smoothed (moving average)
    plt.subplot(2, 2, 2)
    if len(train_losses) > 10:
        window = min(20, len(train_losses) // 10)
        train_smooth = np.convolve(train_losses, np.ones(window)/window, mode='valid')
        val_smooth = np.convolve(val_losses, np.ones(window)/window, mode='valid')
        smooth_epochs = range(window-1, len(train_losses))
        
        plt.plot(smooth_epochs, train_smooth, 'b-', label=f'Training (MA-{window})', linewidth=2)
        plt.plot(smooth_epochs, val_smooth, 'r-', label=f'Validation (MA-{window})', linewidth=2)
        plt.yscale('log')
        plt.xlabel('Epoch')
        plt.ylabel('Smoothed Loss (log scale)')
        plt.title('Smoothed Loss Trends')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    # Overfitting ratio
    plt.subplot(2, 2, 3)
    overfitting_ratios = [val/max(train, 1e-8) for train, val in zip(train_losses, val_losses)]
    plt.plot(epochs, overfitting_ratios, 'purple', linewidth=2)
    plt.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Perfect Fit')
    plt.axhline(y=2.0, color='red', linestyle='--', alpha=0.5, label='Overfitting Threshold')
    plt.xlabel('Epoch')
    plt.ylabel('Val/Train Loss Ratio')
    plt.title('Overfitting Analysis')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Loss improvement rate
    plt.subplot(2, 2, 4)
    if len(train_losses) > 5:
        train_improvement = [-np.log(max(train_losses[i]/train_losses[i-5], 1e-8)) 
                           for i in range(5, len(train_losses))]
        val_improvement = [-np.log(max(val_losses[i]/val_losses[i-5], 1e-8)) 
                         for i in range(5, len(val_losses))]
        
        improve_epochs = range(5, len(train_losses))
        plt.plot(improve_epochs, train_improvement, 'b-', label='Training Improvement Rate')
        plt.plot(improve_epochs, val_improvement, 'r-', label='Validation Improvement Rate')
        plt.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='No Improvement')
        plt.xlabel('Epoch')
        plt.ylabel('Improvement Rate (log scale)')
        plt.title('Learning Progress Rate')
        plt.legend()
        plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 Loss stagnation analysis plot saved to: {output_path}")
