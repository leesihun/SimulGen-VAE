"""Traditional ML Conditioner Module

Implements Random Forest and Support Vector Machine alternatives for latent conditioning
in SimulGenVAE. Designed for small datasets where traditional ML often outperforms deep learning.

Features:
- Multi-output regression for 56D total output (32D main + hierarchical latents)
- Cross-validation and hyperparameter tuning
- Feature importance analysis for Random Forest
- Compatible interface with existing deep learning models
- CPU-only execution with minimal memory requirements

Author: SiHun Lee, Ph.D.
Contact: kevin1007kr@gmail.com
Version: 1.0.0 (Traditional ML Integration)
"""

import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)


class TraditionalMLConditioner:
    """Traditional ML latent conditioner using Random Forest or SVM.
    
    This class provides a drop-in replacement for deep learning latent conditioners,
    specifically optimized for small datasets. Handles both main latent (32D) and
    hierarchical latent (3×8D) predictions using multi-output regression.
    
    Args:
        model_type (str): Model type - 'rf' for Random Forest, 'svm' for SVM
        latent_dim_end (int): Dimension of main latent space (typically 32)
        latent_dim (int): Dimension of hierarchical latent space (typically 8)
        size2 (int): Number of hierarchical layers (typically 3)
        n_components (int): Number of PCA components for feature reduction
        cv_folds (int): Number of cross-validation folds for hyperparameter tuning
        random_state (int): Random state for reproducibility
        
    Attributes:
        model_type (str): Selected model type
        latent_dim_end (int): Main latent dimensions
        latent_dim (int): Hierarchical latent dimensions
        size2 (int): Number of hierarchical layers
        total_output_dim (int): Total output dimensions (32 + 3×8 = 56)
        model (sklearn model): Fitted multi-output regression model
        best_params (dict): Best hyperparameters found during training
        feature_importance (np.ndarray): Feature importance scores (RF only)
        training_scores (dict): Training metrics and cross-validation scores
    """
    
    def __init__(self, model_type='rf', latent_dim_end=32, latent_dim=8, size2=3, 
                 n_components=500, cv_folds=5, random_state=42):
        self.model_type = model_type.lower()
        self.latent_dim_end = latent_dim_end
        self.latent_dim = latent_dim
        self.size2 = size2
        self.total_output_dim = latent_dim_end + (size2 * latent_dim)
        self.n_components = n_components
        self.cv_folds = cv_folds
        self.random_state = random_state
        
        # Initialize model containers
        self.model = None
        self.best_params = None
        self.feature_importance = None
        self.training_scores = {}
        
        # Validation
        if self.model_type not in ['rf', 'svm']:
            raise ValueError(f"Unsupported model_type: {model_type}. Use 'rf' or 'svm'")
            
        print(f"Initialized TraditionalML Conditioner:")
        print(f"  Model type: {self.model_type.upper()}")
        print(f"  Output dimensions: {self.total_output_dim} ({latent_dim_end} main + {size2}×{latent_dim} hierarchical)")
        print(f"  Cross-validation folds: {cv_folds}")
        print(f"  Random state: {random_state}")
    
    def _get_rf_param_grid(self):
        """Get Random Forest hyperparameter grid for small datasets."""
        return {
            'estimator__n_estimators': [100, 200, 300],
            'estimator__max_depth': [10, 15, 20, None],
            'estimator__min_samples_split': [5, 10, 15],
            'estimator__min_samples_leaf': [2, 4, 6],
            'estimator__max_features': ['sqrt', 'log2', 0.3]
        }
    
    def _get_svm_param_grid(self):
        """Get SVM hyperparameter grid for regression tasks."""
        return {
            'estimator__C': [0.1, 1.0, 10.0],
            'estimator__epsilon': [0.01, 0.1, 0.2],
            'estimator__gamma': ['scale', 'auto', 0.001, 0.01],
            'estimator__kernel': ['rbf', 'poly', 'linear']
        }
    
    def _create_base_model(self):
        """Create base model based on model type."""
        if self.model_type == 'rf':
            base_model = RandomForestRegressor(
                random_state=self.random_state,
                n_jobs=-1,  # Use all available cores
                verbose=0
            )
        else:  # svm
            base_model = SVR(
                cache_size=1000  # Increase cache for better performance
            )
        
        # Wrap in MultiOutputRegressor for multi-dimensional output
        return MultiOutputRegressor(base_model, n_jobs=-1)
    
    def fit(self, X, y_main, y_hierarchical, verbose=True):
        """Fit the traditional ML model with hyperparameter tuning.
        
        Args:
            X (np.ndarray): Input features, shape [N, features]
            y_main (np.ndarray): Main latent targets, shape [N, latent_dim_end]
            y_hierarchical (np.ndarray): Hierarchical latent targets, shape [N, size2, latent_dim]
            verbose (bool): Whether to print training progress
            
        Returns:
            dict: Training metrics including MSE, MAE, R² scores
        """
        if verbose:
            print(f"\nTraining {self.model_type.upper()} model...")
            print(f"  Input shape: {X.shape}")
            print(f"  Main latent shape: {y_main.shape}")
            print(f"  Hierarchical latent shape: {y_hierarchical.shape}")
        
        # Combine main and hierarchical latents into single output matrix
        # Flatten hierarchical: [N, size2, latent_dim] → [N, size2*latent_dim]
        y_hier_flat = y_hierarchical.reshape(y_hierarchical.shape[0], -1)
        y_combined = np.concatenate([y_main, y_hier_flat], axis=1)
        
        if verbose:
            print(f"  Combined output shape: {y_combined.shape}")
            print(f"  Expected total dimensions: {self.total_output_dim}")
        
        # Validate dimensions
        assert y_combined.shape[1] == self.total_output_dim, \
            f"Output dimension mismatch: {y_combined.shape[1]} != {self.total_output_dim}"
        
        # Create base model and parameter grid
        base_model = self._create_base_model()
        param_grid = self._get_rf_param_grid() if self.model_type == 'rf' else self._get_svm_param_grid()
        
        # Hyperparameter tuning with cross-validation
        if verbose:
            print(f"  Performing {self.cv_folds}-fold cross-validation...")
        
        grid_search = GridSearchCV(
            base_model,
            param_grid,
            cv=self.cv_folds,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1 if verbose else 0
        )
        
        # Fit with hyperparameter search
        grid_search.fit(X, y_combined)
        
        # Store best model and parameters
        self.model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_
        
        # Calculate training metrics
        y_pred = self.model.predict(X)
        
        # Overall metrics
        mse = mean_squared_error(y_combined, y_pred)
        mae = mean_absolute_error(y_combined, y_pred)
        r2 = r2_score(y_combined, y_pred)
        
        # Separate metrics for main and hierarchical
        y_main_pred = y_pred[:, :self.latent_dim_end]
        y_hier_pred = y_pred[:, self.latent_dim_end:]
        
        main_mse = mean_squared_error(y_main, y_main_pred)
        main_r2 = r2_score(y_main, y_main_pred)
        hier_mse = mean_squared_error(y_hier_flat, y_hier_pred)
        hier_r2 = r2_score(y_hier_flat, y_hier_pred)
        
        self.training_scores = {
            'overall_mse': mse,
            'overall_mae': mae,
            'overall_r2': r2,
            'main_mse': main_mse,
            'main_r2': main_r2,
            'hierarchical_mse': hier_mse,
            'hierarchical_r2': hier_r2,
            'cv_score': -grid_search.best_score_,  # Convert back from negative MSE
            'best_params': self.best_params
        }
        
        # Extract feature importance for Random Forest
        if self.model_type == 'rf':
            # Get feature importance from the first estimator (they should be similar)
            first_estimator = self.model.estimators_[0]
            self.feature_importance = first_estimator.feature_importances_
        
        if verbose:
            print(f"\nTraining completed!")
            print(f"  Best CV MSE: {self.training_scores['cv_score']:.6f}")
            print(f"  Training MSE: {mse:.6f} (Main: {main_mse:.6f}, Hierarchical: {hier_mse:.6f})")
            print(f"  Training R²: {r2:.4f} (Main: {main_r2:.4f}, Hierarchical: {hier_r2:.4f})")
            print(f"  Best parameters: {self.best_params}")
        
        return self.training_scores
    
    def predict(self, X):
        """Predict latent vectors for given inputs.
        
        Args:
            X (np.ndarray): Input features, shape [N, features]
            
        Returns:
            tuple: (main_latent, hierarchical_latent)
                - main_latent: shape [N, latent_dim_end]
                - hierarchical_latent: shape [N, size2, latent_dim]
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")
        
        # Predict combined output
        y_pred_combined = self.model.predict(X)
        
        # Split back into main and hierarchical components
        y_main_pred = y_pred_combined[:, :self.latent_dim_end]
        y_hier_flat = y_pred_combined[:, self.latent_dim_end:]
        
        # Reshape hierarchical latents back to [N, size2, latent_dim]
        y_hier_pred = y_hier_flat.reshape(y_hier_flat.shape[0], self.size2, self.latent_dim)
        
        return y_main_pred, y_hier_pred
    
    def evaluate(self, X, y_main, y_hierarchical, verbose=True):
        """Evaluate model performance on test data.
        
        Args:
            X (np.ndarray): Input features
            y_main (np.ndarray): True main latent vectors
            y_hierarchical (np.ndarray): True hierarchical latent vectors
            verbose (bool): Whether to print evaluation results
            
        Returns:
            dict: Evaluation metrics
        """
        y_main_pred, y_hier_pred = self.predict(X)
        
        # Calculate metrics
        y_hier_flat = y_hierarchical.reshape(y_hierarchical.shape[0], -1)
        y_hier_pred_flat = y_hier_pred.reshape(y_hier_pred.shape[0], -1)
        
        metrics = {
            'main_mse': mean_squared_error(y_main, y_main_pred),
            'main_mae': mean_absolute_error(y_main, y_main_pred),
            'main_r2': r2_score(y_main, y_main_pred),
            'hierarchical_mse': mean_squared_error(y_hier_flat, y_hier_pred_flat),
            'hierarchical_mae': mean_absolute_error(y_hier_flat, y_hier_pred_flat),
            'hierarchical_r2': r2_score(y_hier_flat, y_hier_pred_flat)
        }
        
        # Overall metrics
        y_true_combined = np.concatenate([y_main, y_hier_flat], axis=1)
        y_pred_combined = np.concatenate([y_main_pred, y_hier_pred_flat], axis=1)
        
        metrics.update({
            'overall_mse': mean_squared_error(y_true_combined, y_pred_combined),
            'overall_mae': mean_absolute_error(y_true_combined, y_pred_combined),
            'overall_r2': r2_score(y_true_combined, y_pred_combined)
        })
        
        if verbose:
            print(f"\nEvaluation Results:")
            print(f"  Overall - MSE: {metrics['overall_mse']:.6f}, MAE: {metrics['overall_mae']:.6f}, R²: {metrics['overall_r2']:.4f}")
            print(f"  Main Latent - MSE: {metrics['main_mse']:.6f}, R²: {metrics['main_r2']:.4f}")
            print(f"  Hierarchical - MSE: {metrics['hierarchical_mse']:.6f}, R²: {metrics['hierarchical_r2']:.4f}")
        
        return metrics
    
    def get_feature_importance(self, top_n=20):
        """Get feature importance for Random Forest models.
        
        Args:
            top_n (int): Number of top features to return
            
        Returns:
            np.ndarray: Feature indices sorted by importance (descending)
        """
        if self.model_type != 'rf':
            print("Feature importance only available for Random Forest models.")
            return None
            
        if self.feature_importance is None:
            print("Model not trained or feature importance not available.")
            return None
        
        # Get top features by importance
        feature_indices = np.argsort(self.feature_importance)[::-1]
        return feature_indices[:top_n]
    
    def save_model(self, filepath):
        """Save the trained model and metadata.
        
        Args:
            filepath (str): Path to save the model (without extension)
        """
        if self.model is None:
            raise ValueError("No trained model to save.")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model and metadata
        save_data = {
            'model': self.model,
            'model_type': self.model_type,
            'latent_dim_end': self.latent_dim_end,
            'latent_dim': self.latent_dim,
            'size2': self.size2,
            'total_output_dim': self.total_output_dim,
            'best_params': self.best_params,
            'feature_importance': self.feature_importance,
            'training_scores': self.training_scores,
            'random_state': self.random_state
        }
        
        joblib.dump(save_data, f"{filepath}_traditional_ml.pkl")
        print(f"Model saved to {filepath}_traditional_ml.pkl")
    
    def load_model(self, filepath):
        """Load a trained model and metadata.
        
        Args:
            filepath (str): Path to the saved model (without extension)
        """
        full_path = f"{filepath}_traditional_ml.pkl"
        
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Model file not found: {full_path}")
        
        save_data = joblib.load(full_path)
        
        # Restore model and metadata
        self.model = save_data['model']
        self.model_type = save_data['model_type']
        self.latent_dim_end = save_data['latent_dim_end']
        self.latent_dim = save_data['latent_dim']
        self.size2 = save_data['size2']
        self.total_output_dim = save_data['total_output_dim']
        self.best_params = save_data['best_params']
        self.feature_importance = save_data['feature_importance']
        self.training_scores = save_data['training_scores']
        self.random_state = save_data['random_state']
        
        print(f"Model loaded from {full_path}")
        print(f"  Model type: {self.model_type.upper()}")
        print(f"  Output dimensions: {self.total_output_dim}")
        print(f"  Training R²: {self.training_scores.get('overall_r2', 'N/A'):.4f}")
    
    def __repr__(self):
        """String representation of the model."""
        status = "Trained" if self.model is not None else "Untrained"
        return f"TraditionalMLConditioner({self.model_type.upper()}, {status}, {self.total_output_dim}D output)"