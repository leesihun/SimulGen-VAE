"""Feature Engineering Module for Traditional ML

Comprehensive feature preprocessing pipeline for traditional ML models in SimulGenVAE.
Handles dimensionality reduction, scaling, and feature selection for both image and CSV data.

Features:
- PCA-based dimensionality reduction for high-dimensional image data
- Multiple scaling options (Standard, Robust, MinMax) for different model requirements
- Feature selection using variance thresholding and statistical methods
- Optional spatial feature extraction (HOG, LBP) for enhanced image analysis
- Compatible with existing SimulGenVAE data preprocessing pipeline
- Persistent preprocessing pipelines for consistent train/test transforms

Author: SiHun Lee, Ph.D.
Contact: kevin1007kr@gmail.com
Version: 1.0.0 (Traditional ML Integration)
"""

import numpy as np
import joblib
import os
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler
from sklearn.feature_selection import VarianceThreshold, SelectKBest, f_regression
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore', category=FutureWarning)

try:
    from skimage.feature import hog, local_binary_pattern
    from skimage import exposure
    HAS_SKIMAGE = True
except ImportError:
    HAS_SKIMAGE = False
    print("Warning: scikit-image not available. Advanced image features disabled.")


class FeatureEngineer:
    """Comprehensive feature engineering pipeline for traditional ML.
    
    This class provides feature preprocessing specifically designed for SimulGenVAE's
    traditional ML integration. Handles the transformation from high-dimensional
    inputs (images: 65,536 features) to lower-dimensional representations suitable
    for Random Forest and SVM models.
    
    Args:
        input_type (str): Type of input data - 'image' or 'csv'
        n_components (int): Number of PCA components for dimensionality reduction
        scaler_type (str): Scaling method - 'standard', 'robust', or 'minmax'
        variance_threshold (float): Minimum variance threshold for feature selection
        use_feature_selection (bool): Whether to apply statistical feature selection
        k_best (int): Number of best features to select (if use_feature_selection=True)
        use_spatial_features (bool): Whether to extract spatial features (images only)
        random_state (int): Random state for reproducibility
        
    Attributes:
        pipeline (sklearn.Pipeline): Fitted preprocessing pipeline
        feature_names (list): Names of features after preprocessing
        pca_explained_variance (np.ndarray): Explained variance ratio of PCA components
        selected_features (np.ndarray): Indices of selected features
        preprocessing_stats (dict): Statistics from preprocessing pipeline
    """
    
    def __init__(self, input_type='image', n_components=500, scaler_type='standard',
                 variance_threshold=0.001, use_feature_selection=True, k_best=None,
                 use_spatial_features=False, random_state=42):
        
        self.input_type = input_type.lower()
        self.n_components = n_components
        self.scaler_type = scaler_type.lower()
        self.variance_threshold = variance_threshold
        self.use_feature_selection = use_feature_selection
        self.k_best = k_best
        self.use_spatial_features = use_spatial_features and HAS_SKIMAGE
        self.random_state = random_state
        
        # Initialize containers
        self.pipeline = None
        self.feature_names = []
        self.pca_explained_variance = None
        self.selected_features = None
        self.preprocessing_stats = {}
        
        # Validation
        if self.input_type not in ['image', 'csv']:
            raise ValueError(f"Unsupported input_type: {input_type}. Use 'image' or 'csv'")
        
        if self.scaler_type not in ['standard', 'robust', 'minmax']:
            raise ValueError(f"Unsupported scaler_type: {scaler_type}. Use 'standard', 'robust', or 'minmax'")
        
        if self.use_spatial_features and not HAS_SKIMAGE:
            print("Warning: Spatial features requested but scikit-image not available. Disabling spatial features.")
            self.use_spatial_features = False
        
        print(f"Initialized FeatureEngineer:")
        print(f"  Input type: {self.input_type}")
        print(f"  PCA components: {self.n_components}")
        print(f"  Scaler: {self.scaler_type}")
        print(f"  Spatial features: {self.use_spatial_features}")
        print(f"  Feature selection: {self.use_feature_selection}")
    
    def _create_scaler(self):
        """Create scaler based on specified type."""
        if self.scaler_type == 'standard':
            return StandardScaler()
        elif self.scaler_type == 'robust':
            return RobustScaler()
        else:  # minmax
            return MinMaxScaler()
    
    def _extract_spatial_features(self, images_2d):
        """Extract spatial features from 2D images.
        
        Args:
            images_2d (np.ndarray): Images in 2D format, shape [N, height, width]
            
        Returns:
            np.ndarray: Spatial features, shape [N, n_features]
        """
        if not self.use_spatial_features:
            return np.array([]).reshape(len(images_2d), 0)
        
        print("  Extracting spatial features (HOG + LBP)...")
        
        n_samples = len(images_2d)
        spatial_features = []
        
        for i, image in enumerate(images_2d):
            if i % 50 == 0:
                print(f"    Processing image {i+1}/{n_samples}")
            
            features = []
            
            # HOG features (Histogram of Oriented Gradients)
            try:
                hog_features = hog(
                    image, orientations=8, pixels_per_cell=(16, 16),
                    cells_per_block=(2, 2), visualize=False, 
                    feature_vector=True, block_norm='L2-Hys'
                )
                features.extend(hog_features)
            except Exception as e:
                print(f"    Warning: HOG extraction failed for image {i}: {e}")
                features.extend(np.zeros(72))  # Fallback HOG feature size
            
            # LBP features (Local Binary Pattern)
            try:
                lbp = local_binary_pattern(image, P=8, R=1, method='uniform')
                lbp_hist, _ = np.histogram(lbp.ravel(), bins=10, 
                                         range=(0, 10), density=True)
                features.extend(lbp_hist)
            except Exception as e:
                print(f"    Warning: LBP extraction failed for image {i}: {e}")
                features.extend(np.zeros(10))  # Fallback LBP feature size
            
            spatial_features.append(features)
        
        spatial_array = np.array(spatial_features)
        print(f"  Spatial features shape: {spatial_array.shape}")
        return spatial_array
    
    def _create_pipeline(self, X_sample):
        """Create preprocessing pipeline based on input characteristics."""
        steps = []
        
        # For image data, add spatial features first (if enabled)
        if self.input_type == 'image' and self.use_spatial_features:
            # Note: Spatial features are added externally, not in pipeline
            pass
        
        # Variance threshold for removing low-variance features
        if self.variance_threshold > 0:
            steps.append(('variance_threshold', VarianceThreshold(threshold=self.variance_threshold)))
        
        # PCA for dimensionality reduction (primarily for image data)
        if self.input_type == 'image' or (self.input_type == 'csv' and X_sample.shape[1] > 1000):
            # Adjust n_components based on available samples and features
            max_components = min(self.n_components, X_sample.shape[0] - 1, X_sample.shape[1])
            if max_components < self.n_components:
                print(f"  Adjusting PCA components from {self.n_components} to {max_components}")
                self.n_components = max_components
            
            steps.append(('pca', PCA(n_components=self.n_components, random_state=self.random_state)))
        
        # Feature selection (statistical)
        if self.use_feature_selection and self.k_best is not None:
            actual_k = min(self.k_best, self.n_components if 'pca' in [step[0] for step in steps] else X_sample.shape[1])
            steps.append(('feature_selection', SelectKBest(score_func=f_regression, k=actual_k)))
        
        # Scaling (always last step)
        steps.append(('scaler', self._create_scaler()))
        
        return Pipeline(steps)
    
    def fit_transform(self, X, y=None, verbose=True):
        """Fit preprocessing pipeline and transform training data.
        
        Args:
            X (np.ndarray): Input features, shape [N, features]
            y (np.ndarray, optional): Target values for supervised feature selection
            verbose (bool): Whether to print progress information
            
        Returns:
            np.ndarray: Transformed features, shape [N, n_output_features]
        """
        if verbose:
            print(f"\nFitting FeatureEngineer pipeline...")
            print(f"  Input shape: {X.shape}")
        
        X_processed = X.copy()
        
        # Handle spatial features for images
        if self.input_type == 'image' and self.use_spatial_features:
            # Reshape flattened images back to 2D for spatial feature extraction
            img_size = int(np.sqrt(X.shape[1]))
            if img_size * img_size == X.shape[1]:
                images_2d = X.reshape(X.shape[0], img_size, img_size)
                spatial_features = self._extract_spatial_features(images_2d)
                
                if spatial_features.shape[1] > 0:
                    X_processed = np.concatenate([X_processed, spatial_features], axis=1)
                    if verbose:
                        print(f"  After spatial features: {X_processed.shape}")
        
        # Create and fit pipeline
        self.pipeline = self._create_pipeline(X_processed)
        
        if verbose:
            print(f"  Pipeline steps: {[step[0] for step in self.pipeline.steps]}")
        
        # Fit and transform
        X_transformed = self.pipeline.fit_transform(X_processed, y)
        
        # Store preprocessing statistics
        self._extract_preprocessing_stats()
        
        if verbose:
            print(f"  Output shape: {X_transformed.shape}")
            print(f"  Dimensionality reduction: {X.shape[1]} → {X_transformed.shape[1]} ({X_transformed.shape[1]/X.shape[1]*100:.1f}%)")
        
        return X_transformed
    
    def transform(self, X, verbose=False):
        """Transform new data using fitted preprocessing pipeline.
        
        Args:
            X (np.ndarray): Input features to transform
            verbose (bool): Whether to print progress information
            
        Returns:
            np.ndarray: Transformed features
        """
        if self.pipeline is None:
            raise ValueError("Pipeline not fitted. Call fit_transform() first.")
        
        if verbose:
            print(f"Transforming data: {X.shape}")
        
        X_processed = X.copy()
        
        # Apply spatial features if used during fitting
        if self.input_type == 'image' and self.use_spatial_features:
            img_size = int(np.sqrt(X.shape[1]))
            if img_size * img_size == X.shape[1]:
                images_2d = X.reshape(X.shape[0], img_size, img_size)
                spatial_features = self._extract_spatial_features(images_2d)
                
                if spatial_features.shape[1] > 0:
                    X_processed = np.concatenate([X_processed, spatial_features], axis=1)
        
        X_transformed = self.pipeline.transform(X_processed)
        
        if verbose:
            print(f"  Transformed shape: {X_transformed.shape}")
        
        return X_transformed
    
    def _extract_preprocessing_stats(self):
        """Extract and store preprocessing statistics."""
        self.preprocessing_stats = {
            'input_features': None,
            'output_features': None,
            'pca_explained_variance_ratio': None,
            'pca_cumulative_variance': None,
            'selected_features': None
        }
        
        # Extract PCA statistics
        for step_name, transformer in self.pipeline.steps:
            if step_name == 'pca':
                self.pca_explained_variance = transformer.explained_variance_ratio_
                self.preprocessing_stats['pca_explained_variance_ratio'] = self.pca_explained_variance
                self.preprocessing_stats['pca_cumulative_variance'] = np.cumsum(self.pca_explained_variance)
                
                # Print PCA info
                total_variance = np.sum(self.pca_explained_variance)
                print(f"  PCA explained variance: {total_variance:.1%} ({self.n_components} components)")
                
            elif step_name == 'feature_selection':
                self.selected_features = transformer.get_support(indices=True)
                self.preprocessing_stats['selected_features'] = self.selected_features
                print(f"  Feature selection: {len(self.selected_features)} features selected")
    
    def get_feature_importance_pca(self, component_idx=0):
        """Get feature importance from PCA component loadings.
        
        Args:
            component_idx (int): PCA component index to analyze
            
        Returns:
            np.ndarray: Feature importance scores
        """
        if self.pipeline is None:
            raise ValueError("Pipeline not fitted.")
        
        pca_transformer = None
        for step_name, transformer in self.pipeline.steps:
            if step_name == 'pca':
                pca_transformer = transformer
                break
        
        if pca_transformer is None:
            print("No PCA step found in pipeline.")
            return None
        
        if component_idx >= len(pca_transformer.components_):
            raise ValueError(f"Component index {component_idx} out of range (0-{len(pca_transformer.components_)-1})")
        
        # Get absolute loadings for the specified component
        component_loadings = np.abs(pca_transformer.components_[component_idx])
        return component_loadings
    
    def plot_pca_variance(self):
        """Plot PCA explained variance (requires matplotlib)."""
        if self.pca_explained_variance is None:
            print("No PCA statistics available.")
            return
        
        try:
            import matplotlib.pyplot as plt
            
            cumvar = np.cumsum(self.pca_explained_variance)
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            # Individual variance
            ax1.bar(range(len(self.pca_explained_variance)), self.pca_explained_variance)
            ax1.set_xlabel('Principal Component')
            ax1.set_ylabel('Explained Variance Ratio')
            ax1.set_title('PCA - Individual Component Variance')
            
            # Cumulative variance
            ax2.plot(range(len(cumvar)), cumvar, 'b-', marker='o')
            ax2.axhline(y=0.95, color='r', linestyle='--', label='95% Variance')
            ax2.set_xlabel('Number of Components')
            ax2.set_ylabel('Cumulative Explained Variance')
            ax2.set_title('PCA - Cumulative Variance')
            ax2.legend()
            ax2.grid(True)
            
            plt.tight_layout()
            plt.show()
            
        except ImportError:
            print("Matplotlib not available for plotting.")
    
    def save_pipeline(self, filepath):
        """Save fitted preprocessing pipeline.
        
        Args:
            filepath (str): Path to save pipeline (without extension)
        """
        if self.pipeline is None:
            raise ValueError("No fitted pipeline to save.")
        
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        save_data = {
            'pipeline': self.pipeline,
            'input_type': self.input_type,
            'n_components': self.n_components,
            'scaler_type': self.scaler_type,
            'use_spatial_features': self.use_spatial_features,
            'preprocessing_stats': self.preprocessing_stats,
            'pca_explained_variance': self.pca_explained_variance,
            'selected_features': self.selected_features,
            'random_state': self.random_state
        }
        
        joblib.dump(save_data, f"{filepath}_feature_engineer.pkl")
        print(f"Feature engineering pipeline saved to {filepath}_feature_engineer.pkl")
    
    def load_pipeline(self, filepath):
        """Load fitted preprocessing pipeline.
        
        Args:
            filepath (str): Path to saved pipeline (without extension)
        """
        full_path = f"{filepath}_feature_engineer.pkl"
        
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Pipeline file not found: {full_path}")
        
        save_data = joblib.load(full_path)
        
        self.pipeline = save_data['pipeline']
        self.input_type = save_data['input_type']
        self.n_components = save_data['n_components']
        self.scaler_type = save_data['scaler_type']
        self.use_spatial_features = save_data['use_spatial_features']
        self.preprocessing_stats = save_data['preprocessing_stats']
        self.pca_explained_variance = save_data['pca_explained_variance']
        self.selected_features = save_data['selected_features']
        self.random_state = save_data['random_state']
        
        print(f"Feature engineering pipeline loaded from {full_path}")
        print(f"  Input type: {self.input_type}")
        print(f"  Output features: {self.preprocessing_stats.get('output_features', 'N/A')}")
    
    def __repr__(self):
        """String representation of the feature engineer."""
        status = "Fitted" if self.pipeline is not None else "Not Fitted"
        return f"FeatureEngineer({self.input_type}, {self.n_components} PCA, {status})"