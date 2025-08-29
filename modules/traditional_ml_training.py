"""Traditional ML Training Module

Training pipeline for Random Forest and SVM models in SimulGenVAE, designed to be
a drop-in replacement for deep learning training functions. Provides the same interface
and logging capabilities as the existing latent conditioner training pipeline.

Features:
- Compatible function signature with train_latent_conditioner()
- Cross-validation and hyperparameter tuning integrated into training
- TensorBoard logging for consistency with existing workflow
- Progress monitoring and early stopping equivalent
- Model checkpointing and best model selection
- Validation metrics calculation and reporting

Author: SiHun Lee, Ph.D.
Contact: kevin1007kr@gmail.com
Version: 1.0.0 (Traditional ML Integration)
"""

import torch
import numpy as np
import time
import os
from torch.utils.tensorboard import SummaryWriter
from sklearn.model_selection import train_test_split
from modules.traditional_ml_conditioner import TraditionalMLConditioner
from modules.feature_engineering import FeatureEngineer


def train_traditional_ml_conditioner(traditional_ml_epoch, traditional_ml_dataloader, 
                                   traditional_ml_validation_dataloader, config, 
                                   is_image_data=True, image_size=256):
    """Train Traditional ML latent conditioner with interface matching deep learning version.
    
    This function provides a drop-in replacement for train_latent_conditioner() but uses
    traditional ML models (Random Forest/SVM) instead of deep neural networks. Maintains
    the same logging, validation, and model saving patterns for seamless integration.
    
    Args:
        traditional_ml_epoch (int): Maximum number of epochs (used for CV folds instead)
        traditional_ml_dataloader (DataLoader): Training data loader
        traditional_ml_validation_dataloader (DataLoader): Validation data loader  
        config (dict): Configuration dictionary with traditional ML parameters
        is_image_data (bool): Whether input is image data (affects feature engineering)
        image_size (int): Image size for spatial feature extraction
        
    Returns:
        float: Final validation loss (MSE)
        
    Configuration Parameters (from config dict):
        traditional_ml_mode (str): 'rf' for Random Forest, 'svm' for SVM
        traditional_ml_n_components (int): PCA components for dimensionality reduction
        traditional_ml_cv_folds (int): Cross-validation folds for hyperparameter tuning
        traditional_ml_scaler (str): Scaler type ('standard', 'robust', 'minmax')
        traditional_ml_use_spatial_features (bool): Extract spatial features for images
        traditional_ml_feature_selection (bool): Apply statistical feature selection
        traditional_ml_k_best (int): Number of best features to select
        latent_dim_end (int): Main latent dimension (typically 32)
        latent_dim (int): Hierarchical latent dimension (typically 8)
    """
    
    print("=" * 60)
    print("TRADITIONAL ML LATENT CONDITIONER TRAINING")
    print("=" * 60)
    
    # Extract configuration parameters
    model_type = config.get('traditional_ml_mode', 'rf').lower()
    n_components = config.get('traditional_ml_n_components', 500)
    cv_folds = min(config.get('traditional_ml_cv_folds', 5), traditional_ml_epoch // 2)  # Use epoch limit
    scaler_type = config.get('traditional_ml_scaler', 'standard')
    use_spatial_features = config.get('traditional_ml_use_spatial_features', False)
    use_feature_selection = config.get('traditional_ml_feature_selection', True)
    k_best = config.get('traditional_ml_k_best', None)
    
    # Model architecture parameters
    latent_dim_end = config['latent_dim_end']
    latent_dim = config['latent_dim']
    size2 = len(config.get('num_filter_enc', [1024, 512, 256, 128])) - 1
    
    print(f"Configuration:")
    print(f"  Model type: {model_type.upper()}")
    print(f"  PCA components: {n_components}")
    print(f"  CV folds: {cv_folds}")
    print(f"  Scaler: {scaler_type}")
    print(f"  Spatial features: {use_spatial_features}")
    print(f"  Feature selection: {use_feature_selection}")
    print(f"  Output dimensions: {latent_dim_end} + {size2}×{latent_dim} = {latent_dim_end + size2*latent_dim}")
    
    # Initialize TensorBoard writer for consistency
    writer = SummaryWriter(log_dir='./TraditionalMLRuns', comment='TraditionalML')
    
    # Convert DataLoaders to numpy arrays
    print("\nPreparing training data...")
    train_inputs, train_main_targets, train_hier_targets = _extract_data_from_loader(traditional_ml_dataloader)
    val_inputs, val_main_targets, val_hier_targets = _extract_data_from_loader(traditional_ml_validation_dataloader)
    
    print(f"Training data: {train_inputs.shape[0]} samples")
    print(f"Validation data: {val_inputs.shape[0]} samples")
    
    # Initialize feature engineering pipeline
    print("\nInitializing feature engineering...")
    input_type = 'image' if is_image_data else 'csv'
    feature_engineer = FeatureEngineer(
        input_type=input_type,
        n_components=n_components,
        scaler_type=scaler_type,
        use_feature_selection=use_feature_selection,
        k_best=k_best,
        use_spatial_features=use_spatial_features,
        random_state=42
    )
    
    # Fit feature engineering pipeline and transform data
    print("\nApplying feature engineering...")
    start_time = time.time()
    
    # Combine targets for supervised feature selection
    combined_targets = np.concatenate([
        train_main_targets,
        train_hier_targets.reshape(train_hier_targets.shape[0], -1)
    ], axis=1)
    
    train_features = feature_engineer.fit_transform(train_inputs, combined_targets, verbose=True)
    val_features = feature_engineer.transform(val_inputs, verbose=True)
    
    feature_time = time.time() - start_time
    print(f"Feature engineering completed in {feature_time:.2f}s")
    
    # Initialize traditional ML model
    print(f"\nInitializing {model_type.upper()} model...")
    traditional_ml_model = TraditionalMLConditioner(
        model_type=model_type,
        latent_dim_end=latent_dim_end,
        latent_dim=latent_dim,
        size2=size2,
        n_components=n_components,
        cv_folds=cv_folds,
        random_state=42
    )
    
    # Training with cross-validation
    print(f"\nTraining {model_type.upper()} model with {cv_folds}-fold cross-validation...")
    start_time = time.time()
    
    training_metrics = traditional_ml_model.fit(
        train_features, train_main_targets, train_hier_targets, verbose=True
    )
    
    training_time = time.time() - start_time
    print(f"\nModel training completed in {training_time:.2f}s")
    
    # Validation evaluation
    print("\nEvaluating on validation set...")
    validation_metrics = traditional_ml_model.evaluate(
        val_features, val_main_targets, val_hier_targets, verbose=True
    )
    
    # Log metrics to TensorBoard (simulate epoch-based logging)
    epoch = 0  # Traditional ML is single-shot training
    
    # Training metrics
    writer.add_scalar('Loss/Train_Overall', training_metrics['overall_mse'], epoch)
    writer.add_scalar('Loss/Train_Main', training_metrics['main_mse'], epoch)
    writer.add_scalar('Loss/Train_Hierarchical', training_metrics['hierarchical_mse'], epoch)
    writer.add_scalar('R2/Train_Overall', training_metrics['overall_r2'], epoch)
    writer.add_scalar('R2/Train_Main', training_metrics['main_r2'], epoch)
    writer.add_scalar('R2/Train_Hierarchical', training_metrics['hierarchical_r2'], epoch)
    
    # Validation metrics
    writer.add_scalar('Loss/Val_Overall', validation_metrics['overall_mse'], epoch)
    writer.add_scalar('Loss/Val_Main', validation_metrics['main_mse'], epoch)
    writer.add_scalar('Loss/Val_Hierarchical', validation_metrics['hierarchical_mse'], epoch)
    writer.add_scalar('R2/Val_Overall', validation_metrics['overall_r2'], epoch)
    writer.add_scalar('R2/Val_Main', validation_metrics['main_r2'], epoch)
    writer.add_scalar('R2/Val_Hierarchical', validation_metrics['hierarchical_r2'], epoch)
    
    # Cross-validation score
    writer.add_scalar('Loss/CV_Score', training_metrics['cv_score'], epoch)
    
    # Feature importance (for Random Forest)
    if model_type == 'rf' and traditional_ml_model.feature_importance is not None:
        top_features = traditional_ml_model.get_feature_importance(top_n=20)
        for i, feature_idx in enumerate(top_features):
            writer.add_scalar(f'FeatureImportance/Feature_{feature_idx}', 
                            traditional_ml_model.feature_importance[feature_idx], epoch)
    
    # Save models
    print("\nSaving trained models...")
    
    # Create save directory
    os.makedirs('model_save', exist_ok=True)
    
    # Save traditional ML model
    model_save_path = 'model_save/traditional_ml_latent_conditioner'
    traditional_ml_model.save_model(model_save_path)
    
    # Save feature engineering pipeline
    feature_save_path = 'model_save/traditional_ml_feature_engineer'
    feature_engineer.save_pipeline(feature_save_path)
    
    # Create a wrapper function for inference compatibility
    _create_inference_wrapper(traditional_ml_model, feature_engineer, model_save_path)
    
    # Final performance summary
    print("\n" + "=" * 60)
    print("TRAINING SUMMARY")
    print("=" * 60)
    print(f"Model Type: {model_type.upper()}")
    print(f"Feature Engineering Time: {feature_time:.2f}s")
    print(f"Model Training Time: {training_time:.2f}s")
    print(f"Total Time: {feature_time + training_time:.2f}s")
    print()
    print("Training Performance:")
    print(f"  Overall MSE: {training_metrics['overall_mse']:.6f}")
    print(f"  Main Latent R²: {training_metrics['main_r2']:.4f}")
    print(f"  Hierarchical R²: {training_metrics['hierarchical_r2']:.4f}")
    print(f"  Cross-validation MSE: {training_metrics['cv_score']:.6f}")
    print()
    print("Validation Performance:")
    print(f"  Overall MSE: {validation_metrics['overall_mse']:.6f}")
    print(f"  Main Latent R²: {validation_metrics['main_r2']:.4f}")
    print(f"  Hierarchical R²: {validation_metrics['hierarchical_r2']:.4f}")
    print()
    print("Dimensionality Reduction:")
    print(f"  Input features: {train_inputs.shape[1]}")
    print(f"  Output features: {train_features.shape[1]}")
    print(f"  Reduction ratio: {train_features.shape[1]/train_inputs.shape[1]*100:.1f}%")
    
    if model_type == 'rf' and traditional_ml_model.feature_importance is not None:
        top_5_features = traditional_ml_model.get_feature_importance(top_n=5)
        print()
        print("Top 5 Most Important Features:")
        for i, feature_idx in enumerate(top_5_features):
            importance = traditional_ml_model.feature_importance[feature_idx]
            print(f"  {i+1}. Feature {feature_idx}: {importance:.4f}")
    
    print("=" * 60)
    
    # Close TensorBoard writer
    writer.close()
    
    # Return validation loss for compatibility with existing code
    return validation_metrics['overall_mse']


def _extract_data_from_loader(dataloader):
    """Extract all data from PyTorch DataLoader into numpy arrays.
    
    Args:
        dataloader (DataLoader): PyTorch DataLoader
        
    Returns:
        tuple: (inputs, main_targets, hierarchical_targets) as numpy arrays
    """
    inputs_list = []
    main_targets_list = []
    hier_targets_list = []
    
    with torch.no_grad():
        for batch_idx, (x, y_main, y_hier) in enumerate(dataloader):
            # Convert to numpy and move to CPU if needed
            inputs_list.append(x.cpu().numpy())
            main_targets_list.append(y_main.cpu().numpy())
            hier_targets_list.append(y_hier.cpu().numpy())
            
            if batch_idx % 10 == 0:
                print(f"  Processing batch {batch_idx+1}/{len(dataloader)}")
    
    # Concatenate all batches
    inputs = np.concatenate(inputs_list, axis=0)
    main_targets = np.concatenate(main_targets_list, axis=0)
    hier_targets = np.concatenate(hier_targets_list, axis=0)
    
    return inputs, main_targets, hier_targets


def _create_inference_wrapper(ml_model, feature_engineer, save_path):
    """Create inference wrapper for compatibility with existing evaluation code.
    
    Args:
        ml_model (TraditionalMLConditioner): Trained ML model
        feature_engineer (FeatureEngineer): Fitted feature engineering pipeline
        save_path (str): Path to save the wrapper
    """
    
    class TraditionalMLWrapper:
        """Wrapper class to provide PyTorch-like interface for traditional ML models."""
        
        def __init__(self, ml_model, feature_engineer):
            self.ml_model = ml_model
            self.feature_engineer = feature_engineer
            self.training = False  # For compatibility with eval() calls
        
        def eval(self):
            """Set model to evaluation mode (compatibility method)."""
            self.training = False
            return self
        
        def train(self, mode=True):
            """Set model to training mode (compatibility method)."""
            self.training = mode
            return self
        
        def forward(self, x):
            """Forward pass compatible with PyTorch models.
            
            Args:
                x (torch.Tensor or np.ndarray): Input features
                
            Returns:
                tuple: (main_latent, hierarchical_latent) as torch tensors
            """
            # Convert to numpy if needed
            if isinstance(x, torch.Tensor):
                x_np = x.detach().cpu().numpy()
                device = x.device
            else:
                x_np = x
                device = torch.device('cpu')
            
            # Apply feature engineering
            x_features = self.feature_engineer.transform(x_np)
            
            # Predict using ML model
            main_pred, hier_pred = self.ml_model.predict(x_features)
            
            # Convert back to torch tensors
            main_tensor = torch.tensor(main_pred, dtype=torch.float32, device=device)
            hier_tensor = torch.tensor(hier_pred, dtype=torch.float32, device=device)
            
            return main_tensor, hier_tensor
        
        def __call__(self, x):
            """Make the wrapper callable like a PyTorch model."""
            return self.forward(x)
        
        def to(self, device):
            """Device compatibility method (no-op for traditional ML)."""
            return self
        
        def cuda(self):
            """CUDA compatibility method (no-op for traditional ML)."""
            return self
        
        def cpu(self):
            """CPU compatibility method (no-op for traditional ML)."""
            return self
    
    # Create wrapper
    wrapper = TraditionalMLWrapper(ml_model, feature_engineer)
    
    # Save wrapper using torch.save for compatibility
    try:
        import pickle
        with open(f"{save_path}_wrapper.pkl", 'wb') as f:
            pickle.dump(wrapper, f)
        print(f"Inference wrapper saved to {save_path}_wrapper.pkl")
    except Exception as e:
        print(f"Warning: Could not save inference wrapper: {e}")


def load_traditional_ml_model(model_path):
    """Load traditional ML model with inference wrapper.
    
    Args:
        model_path (str): Path to saved model (without extension)
        
    Returns:
        TraditionalMLWrapper: Loaded model wrapper
    """
    try:
        import pickle
        with open(f"{model_path}_wrapper.pkl", 'rb') as f:
            wrapper = pickle.load(f)
        print(f"Traditional ML model loaded from {model_path}_wrapper.pkl")
        return wrapper
    except FileNotFoundError:
        print(f"Traditional ML wrapper not found at {model_path}_wrapper.pkl")
        
        # Try to reconstruct from individual components
        try:
            ml_model = TraditionalMLConditioner()
            ml_model.load_model(model_path)
            
            feature_engineer = FeatureEngineer()
            feature_engineer.load_pipeline(model_path.replace('traditional_ml_latent_conditioner', 'traditional_ml_feature_engineer'))
            
            wrapper = _create_inference_wrapper.__wrapped__(ml_model, feature_engineer, model_path)
            return wrapper
            
        except Exception as e:
            raise FileNotFoundError(f"Could not load traditional ML model: {e}")


if __name__ == "__main__":
    # Test/demo code
    print("Traditional ML Training Module")
    print("This module provides traditional ML training compatible with SimulGenVAE")
    print("Use train_traditional_ml_conditioner() as a drop-in replacement for train_latent_conditioner()")