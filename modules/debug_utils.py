import torch
import numpy as np

class NaNWatcher:
    """A utility class to monitor and log NaN occurrences during training"""
    
    def __init__(self):
        self.nan_counts = {
            'train_recon': 0,
            'train_kl': 0,
            'val_recon': 0,
            'val_kl': 0,
            'model_params': 0
        }
        self.total_batches = {'train': 0, 'val': 0}
    
    def log_nan(self, loss_type, epoch, batch_idx=None):
        """Log a NaN occurrence"""
        self.nan_counts[loss_type] += 1
        if batch_idx is not None:
            print(f"NaN #{self.nan_counts[loss_type]} in {loss_type} at epoch {epoch}, batch {batch_idx}")
        else:
            print(f"NaN #{self.nan_counts[loss_type]} in {loss_type} at epoch {epoch}")
    
    def log_batch(self, phase):
        """Log a completed batch"""
        self.total_batches[phase] += 1
    
    def get_nan_rate(self, loss_type):
        """Get the NaN rate for a specific loss type"""
        phase = 'train' if 'train' in loss_type else 'val'
        if self.total_batches[phase] == 0:
            return 0.0
        return self.nan_counts[loss_type] / self.total_batches[phase]
    
    def print_summary(self):
        """Print a summary of NaN occurrences"""
        print("\n" + "="*50)
        print("NaN Occurrence Summary:")
        print("="*50)
        for loss_type, count in self.nan_counts.items():
            rate = self.get_nan_rate(loss_type) if 'model' not in loss_type else 0
            print(f"{loss_type:15}: {count:6} occurrences (rate: {rate:.4f})")
        print(f"Total train batches: {self.total_batches['train']}")
        print(f"Total val batches: {self.total_batches['val']}")
        print("="*50)

def check_tensor_health(tensor, name="tensor", threshold=10.0):
    """Check if a tensor contains healthy values"""
    if tensor is None:
        return True, f"{name} is None"
    
    issues = []
    
    # Check for NaN
    if torch.isnan(tensor).any():
        nan_count = torch.isnan(tensor).sum().item()
        issues.append(f"Contains {nan_count} NaN values")
    
    # Check for Inf
    if torch.isinf(tensor).any():
        inf_count = torch.isinf(tensor).sum().item()
        issues.append(f"Contains {inf_count} Inf values")
    
    # Check for extreme values
    max_val = tensor.max().item()
    min_val = tensor.min().item()
    if abs(max_val) > threshold or abs(min_val) > threshold:
        issues.append(f"Extreme values: min={min_val:.2e}, max={max_val:.2e}")
    
    is_healthy = len(issues) == 0
    status = "healthy" if is_healthy else "; ".join(issues)
    
    return is_healthy, f"{name}: {status}"

def get_model_health_report(model):
    """Generate a comprehensive health report for model parameters"""
    report = []
    total_params = 0
    problematic_params = 0
    
    for name, param in model.named_parameters():
        if param is None:
            continue
            
        total_params += param.numel()
        is_healthy, status = check_tensor_health(param.data, name)
        
        if not is_healthy:
            problematic_params += 1
            report.append(status)
    
    summary = f"Model Health: {total_params - problematic_params}/{total_params} parameters healthy"
    if problematic_params > 0:
        summary += f" ({problematic_params} problematic)"
    
    return summary, report

# Add this to your training script to monitor NaN issues
nan_watcher = NaNWatcher() 