"""
Loss Optimization Configuration for SimulGen-VAE

This configuration focuses on achieving lower VAE loss through:
1. Better data preprocessing
2. Improved loss balancing
3. Optimized training schedules
4. Enhanced regularization
"""

# Optimized hyperparameters for lower VAE loss
LOSS_OPTIMIZATION_CONFIG = {
    'data_preprocessing': {
        'scaling_range': (-0.9, 0.9),  # Wider range for better reconstruction
        'use_standard_scaler': False,   # MinMax generally better for VAEs
        'add_noise_regularization': True,
        'noise_std': 0.01,
    },
    
    'loss_balancing': {
        'adaptive_alpha': True,
        'initial_alpha': 1000,
        'final_alpha': 100,
        'alpha_decay_start': 0.2,  # Start decay at 20% of training
        'alpha_decay_end': 0.6,    # End decay at 60% of training
    },
    
    'kl_scheduling': {
        'init_beta': 1e-8,
        'target_beta': 5e-4,
        'warmup_start': 0.1,
        'warmup_end': 0.8,
        'use_cyclical_beta': False,  # Can help with posterior collapse
    },
    
    'optimizer_settings': {
        'learning_rate': 5e-5,      # Lower for more stable training
        'weight_decay': 1e-5,       # L2 regularization
        'use_warmup': True,
        'warmup_epochs': 100,
        'scheduler_type': 'cosine_warmup_restarts',
        'restart_period': 1000,
    },
    
    'training_stability': {
        'gradient_clip_norm': 2.0,   # Reduced from 5.0
        'use_early_stopping': True,
        'early_stopping_patience': 300,
        'early_stopping_min_delta': 1e-6,
        'save_best_model': True,
    },
    
    'architecture_tweaks': {
        'use_spectral_norm': True,
        'add_batch_norm': False,     # Can hurt VAE training
        'use_layer_norm': True,      # Better for VAEs
        'dropout_rate': 0.1,         # Light dropout
    },
    
    'loss_functions': {
        'reconstruction_loss': 'MSE',  # Your current setting
        'alternative_losses': ['smoothL1', 'Huber'],  # Can try these
        'use_perceptual_loss': False,  # Advanced option
    }
}

def get_optimized_training_params():
    """
    Returns optimized training parameters for lower VAE loss
    """
    return {
        'batch_size': 32,           # Your current setting
        'learning_rate': 5e-5,      # Reduced from 1e-4
        'epochs': 5002,             # Your current setting
        'alpha': 1000,              # Will be made adaptive
        'latent_dim': 32,           # Your current setting
        'hierarchical_dim': 8,      # Your current setting
        'use_mixed_precision': True,
        'compile_model': True,
        'load_all_to_gpu': True,    # For your use case
    }

def print_loss_optimization_tips():
    """
    Print actionable tips for reducing VAE loss
    """
    tips = [
        "🎯 IMMEDIATE ACTIONS:",
        "1. Use wider data scaling range (-0.9, 0.9) instead of (-0.7, 0.7)",
        "2. Lower learning rate to 5e-5 for more stable training",
        "3. Use adaptive alpha scheduling (start high, decay gradually)",
        "4. Implement better KL beta scheduling (start very low, increase slowly)",
        "",
        "📊 MONITORING:",
        "5. Watch reconstruction vs KL loss balance",
        "6. Use early stopping to prevent overfitting",
        "7. Monitor validation loss trends",
        "",
        "🔧 ADVANCED OPTIMIZATIONS:",
        "8. Add light weight decay (1e-5) for regularization",
        "9. Use cosine annealing with warm restarts",
        "10. Consider different loss functions (SmoothL1, Huber)",
        "",
        "⚠️  TROUBLESHOOTING:",
        "- If KL loss dominates: Lower beta values",
        "- If reconstruction poor: Increase alpha or lower beta",
        "- If overfitting: Add weight decay, reduce learning rate",
        "- If training unstable: Lower gradient clipping norm"
    ]
    
    for tip in tips:
        print(tip)

if __name__ == "__main__":
    print_loss_optimization_tips() 