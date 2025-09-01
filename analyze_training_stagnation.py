#!/usr/bin/env python3
"""
Training Stagnation Analysis Tool for SimulGenVAE

This script analyzes training logs and provides recommendations for addressing
loss stagnation issues in latent conditioner end-to-end training.

Usage:
    python analyze_training_stagnation.py --log_dir ./output --plot_analysis
    python analyze_training_stagnation.py --csv_file ./output/e2e_training_data_20241201_143022.csv
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def analyze_loss_progression(losses, loss_name="Loss"):
    """Analyze loss progression and identify issues."""
    analysis = {
        'loss_name': loss_name,
        'total_epochs': len(losses),
        'final_loss': losses[-1] if losses else 0,
        'best_loss': min(losses) if losses else 0,
        'issues': [],
        'recommendations': []
    }
    
    if len(losses) < 10:
        analysis['issues'].append('insufficient_data')
        return analysis
    
    # Calculate improvement metrics
    early_loss = np.mean(losses[:min(50, len(losses)//4)])
    late_loss = np.mean(losses[-min(50, len(losses)//4):])
    improvement_ratio = early_loss / max(late_loss, 1e-8)
    
    # Calculate stagnation periods
    window_size = min(20, len(losses)//10)
    if window_size > 5:
        moving_avg = np.convolve(losses, np.ones(window_size)/window_size, mode='valid')
        stagnant_periods = 0
        for i in range(1, len(moving_avg)):
            improvement = (moving_avg[i-1] - moving_avg[i]) / max(moving_avg[i-1], 1e-8)
            if improvement < 1e-4:  # Less than 0.01% improvement
                stagnant_periods += 1
        
        stagnation_ratio = stagnant_periods / len(moving_avg)
        analysis['stagnation_ratio'] = stagnation_ratio
    else:
        analysis['stagnation_ratio'] = 0
    
    # Identify issues
    if improvement_ratio < 2.0:
        analysis['issues'].append('poor_initial_convergence')
        analysis['recommendations'].append('Increase initial learning rate or extend warmup period')
    
    if analysis['stagnation_ratio'] > 0.7:
        analysis['issues'].append('excessive_stagnation')
        analysis['recommendations'].append('Use adaptive learning rate scheduling')
    
    # Check for oscillations
    if len(losses) > 20:
        recent_losses = losses[-20:]
        loss_std = np.std(recent_losses)
        loss_mean = np.mean(recent_losses)
        if loss_std / loss_mean > 0.1:  # High relative standard deviation
            analysis['issues'].append('training_instability')
            analysis['recommendations'].append('Reduce learning rate or improve gradient clipping')
    
    # Check for plateau detection
    plateau_threshold = 1e-6
    plateau_epochs = 0
    best_so_far = float('inf')
    
    for loss in losses:
        if loss < best_so_far - plateau_threshold:
            best_so_far = loss
            plateau_epochs = 0
        else:
            plateau_epochs += 1
    
    if plateau_epochs > len(losses) * 0.3:  # More than 30% of training stuck
        analysis['issues'].append('long_plateau')
        analysis['recommendations'].append('Implement plateau detection with LR reduction')
    
    analysis['improvement_ratio'] = improvement_ratio
    analysis['plateau_epochs'] = plateau_epochs
    
    return analysis

def analyze_overfitting(train_losses, val_losses):
    """Analyze overfitting patterns."""
    if len(train_losses) != len(val_losses) or len(train_losses) < 10:
        return {'issues': ['insufficient_data'], 'recommendations': []}
    
    analysis = {'issues': [], 'recommendations': []}
    
    # Calculate overfitting ratios
    overfitting_ratios = [val/max(train, 1e-8) for train, val in zip(train_losses, val_losses)]
    final_ratio = overfitting_ratios[-1]
    max_ratio = max(overfitting_ratios)
    
    # Check for severe overfitting
    if final_ratio > 3.0:
        analysis['issues'].append('severe_overfitting')
        analysis['recommendations'].append('Increase regularization or reduce model complexity')
    elif final_ratio > 1.5:
        analysis['issues'].append('moderate_overfitting')
        analysis['recommendations'].append('Add dropout or weight decay')
    
    # Check for early overfitting
    if len(overfitting_ratios) > 50:
        early_ratios = overfitting_ratios[:50]
        if max(early_ratios) > 2.0:
            analysis['issues'].append('early_overfitting')
            analysis['recommendations'].append('Reduce initial learning rate or add early regularization')
    
    # Check for divergence
    recent_ratios = overfitting_ratios[-20:] if len(overfitting_ratios) >= 20 else overfitting_ratios
    if any(ratio > 5.0 for ratio in recent_ratios):
        analysis['issues'].append('training_divergence')
        analysis['recommendations'].append('Stop training and reduce learning rate significantly')
    
    analysis['final_overfitting_ratio'] = final_ratio
    analysis['max_overfitting_ratio'] = max_ratio
    
    return analysis

def analyze_learning_rate_schedule(learning_rates, losses):
    """Analyze learning rate scheduling effectiveness."""
    if len(learning_rates) < 10 or len(losses) < 10:
        return {'issues': ['insufficient_data'], 'recommendations': []}
    
    analysis = {'issues': [], 'recommendations': []}
    
    # Check for too rapid decay
    initial_lr = learning_rates[0]
    final_lr = learning_rates[-1]
    decay_ratio = initial_lr / max(final_lr, 1e-8)
    
    if decay_ratio > 1000:
        analysis['issues'].append('excessive_lr_decay')
        analysis['recommendations'].append('Use slower LR decay or higher minimum LR')
    
    # Check for insufficient decay
    if decay_ratio < 2.0 and len(losses) > 100:
        analysis['issues'].append('insufficient_lr_decay')
        analysis['recommendations'].append('Allow more aggressive LR scheduling')
    
    # Check for LR-loss correlation
    if len(learning_rates) == len(losses):
        # Find periods where LR changed significantly
        lr_changes = []
        for i in range(1, len(learning_rates)):
            if learning_rates[i] < learning_rates[i-1] * 0.9:  # 10% or more reduction
                lr_changes.append(i)
        
        # Check if loss improved after LR reductions
        improvements_after_reduction = 0
        for change_epoch in lr_changes:
            if change_epoch < len(losses) - 5:
                before_loss = np.mean(losses[max(0, change_epoch-5):change_epoch])
                after_loss = np.mean(losses[change_epoch:change_epoch+5])
                if after_loss < before_loss:
                    improvements_after_reduction += 1
        
        if len(lr_changes) > 0:
            improvement_rate = improvements_after_reduction / len(lr_changes)
            if improvement_rate < 0.5:
                analysis['issues'].append('ineffective_lr_scheduling')
                analysis['recommendations'].append('Try different LR scheduling strategy')
    
    analysis['decay_ratio'] = decay_ratio
    analysis['lr_changes'] = len(lr_changes) if 'lr_changes' in locals() else 0
    
    return analysis

def generate_comprehensive_report(csv_file):
    """Generate comprehensive training analysis report."""
    try:
        df = pd.read_csv(csv_file)
        print(f"📊 Analyzing training data from: {csv_file}")
        print(f"   Total epochs: {len(df)}")
        print(f"   Columns: {list(df.columns)}")
        print("=" * 80)
        
        # Analyze training loss
        if 'train_loss' in df.columns:
            train_analysis = analyze_loss_progression(df['train_loss'].tolist(), "Training Loss")
            print("🔍 TRAINING LOSS ANALYSIS:")
            print(f"   Final loss: {train_analysis['final_loss']:.6e}")
            print(f"   Best loss: {train_analysis['best_loss']:.6e}")
            print(f"   Improvement ratio: {train_analysis.get('improvement_ratio', 0):.2f}x")
            print(f"   Stagnation ratio: {train_analysis.get('stagnation_ratio', 0):.2f}")
            if train_analysis['issues']:
                print(f"   Issues: {', '.join(train_analysis['issues'])}")
            print()
        
        # Analyze validation loss
        if 'val_loss' in df.columns:
            val_analysis = analyze_loss_progression(df['val_loss'].tolist(), "Validation Loss")
            print("🎯 VALIDATION LOSS ANALYSIS:")
            print(f"   Final loss: {val_analysis['final_loss']:.6e}")
            print(f"   Best loss: {val_analysis['best_loss']:.6e}")
            print(f"   Improvement ratio: {val_analysis.get('improvement_ratio', 0):.2f}x")
            print(f"   Plateau epochs: {val_analysis.get('plateau_epochs', 0)}")
            if val_analysis['issues']:
                print(f"   Issues: {', '.join(val_analysis['issues'])}")
            print()
        
        # Analyze overfitting
        if 'train_loss' in df.columns and 'val_loss' in df.columns:
            overfitting_analysis = analyze_overfitting(df['train_loss'].tolist(), df['val_loss'].tolist())
            print("⚠️  OVERFITTING ANALYSIS:")
            print(f"   Final overfitting ratio: {overfitting_analysis.get('final_overfitting_ratio', 0):.2f}")
            print(f"   Max overfitting ratio: {overfitting_analysis.get('max_overfitting_ratio', 0):.2f}")
            if overfitting_analysis['issues']:
                print(f"   Issues: {', '.join(overfitting_analysis['issues'])}")
            print()
        
        # Analyze learning rate schedule
        if 'learning_rate' in df.columns:
            lr_analysis = analyze_learning_rate_schedule(df['learning_rate'].tolist(), 
                                                       df['train_loss'].tolist() if 'train_loss' in df.columns else [])
            print("📉 LEARNING RATE ANALYSIS:")
            print(f"   Initial LR: {df['learning_rate'].iloc[0]:.6e}")
            print(f"   Final LR: {df['learning_rate'].iloc[-1]:.6e}")
            print(f"   Decay ratio: {lr_analysis.get('decay_ratio', 0):.1f}x")
            print(f"   LR changes: {lr_analysis.get('lr_changes', 0)}")
            if lr_analysis['issues']:
                print(f"   Issues: {', '.join(lr_analysis['issues'])}")
            print()
        
        # Compile all recommendations
        all_recommendations = []
        for analysis in [train_analysis if 'train_analysis' in locals() else {},
                        val_analysis if 'val_analysis' in locals() else {},
                        overfitting_analysis if 'overfitting_analysis' in locals() else {},
                        lr_analysis if 'lr_analysis' in locals() else {}]:
            all_recommendations.extend(analysis.get('recommendations', []))
        
        if all_recommendations:
            print("💡 RECOMMENDATIONS:")
            for i, rec in enumerate(set(all_recommendations), 1):
                print(f"   {i}. {rec}")
            print()
        
        print("🔧 SPECIFIC SOLUTIONS FOR LOSS STAGNATION:")
        print("   1. Switch to improved E2E training: set use_improved_e2e=1 in condition.txt")
        print("   2. Reduce learning rate: try latent_conditioner_lr=0.001 or 0.003")
        print("   3. Increase regularization: try latent_reg_weight=0.002 or 0.005")
        print("   4. Extend warmup: the improved version uses 30 warmup epochs")
        print("   5. Use adaptive scheduling: automatically handled in improved version")
        print("=" * 80)
        
        return True
        
    except Exception as e:
        print(f"❌ Error analyzing CSV file: {e}")
        return False

def create_diagnostic_plots(csv_file, output_dir="./output"):
    """Create comprehensive diagnostic plots."""
    try:
        df = pd.read_csv(csv_file)
        
        plt.style.use('default')
        fig = plt.figure(figsize=(20, 16))
        
        # Create a 4x3 subplot grid
        gs = fig.add_gridspec(4, 3, hspace=0.3, wspace=0.3)
        
        epochs = df['epoch'] if 'epoch' in df.columns else range(len(df))
        
        # 1. Main loss plot
        ax1 = fig.add_subplot(gs[0, :2])
        if 'train_loss' in df.columns:
            ax1.plot(epochs, df['train_loss'], 'b-', linewidth=2, label='Training Loss', alpha=0.8)
        if 'val_loss' in df.columns:
            ax1.plot(epochs, df['val_loss'], 'r-', linewidth=2, label='Validation Loss', alpha=0.8)
        ax1.set_yscale('log')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss (log scale)')
        ax1.set_title('Loss Progression Analysis')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Overfitting analysis
        ax2 = fig.add_subplot(gs[0, 2])
        if 'train_loss' in df.columns and 'val_loss' in df.columns:
            overfitting_ratios = df['val_loss'] / df['train_loss'].clip(lower=1e-8)
            ax2.plot(epochs, overfitting_ratios, 'purple', linewidth=2)
            ax2.axhline(y=1.0, color='gray', linestyle='--', alpha=0.5, label='Perfect Fit')
            ax2.axhline(y=2.0, color='red', linestyle='--', alpha=0.5, label='Overfitting')
            ax2.set_xlabel('Epoch')
            ax2.set_ylabel('Val/Train Ratio')
            ax2.set_title('Overfitting Monitor')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
        
        # 3. Learning rate schedule
        ax3 = fig.add_subplot(gs[1, 0])
        if 'learning_rate' in df.columns:
            ax3.plot(epochs, df['learning_rate'], 'orange', linewidth=2)
            ax3.set_yscale('log')
            ax3.set_xlabel('Epoch')
            ax3.set_ylabel('Learning Rate (log)')
            ax3.set_title('LR Schedule')
            ax3.grid(True, alpha=0.3)
        
        # 4. Regularization weight
        ax4 = fig.add_subplot(gs[1, 1])
        if 'regularization_weight' in df.columns:
            ax4.plot(epochs, df['regularization_weight'], 'green', linewidth=2)
            ax4.set_yscale('log')
            ax4.set_xlabel('Epoch')
            ax4.set_ylabel('Reg Weight (log)')
            ax4.set_title('Regularization Schedule')
            ax4.grid(True, alpha=0.3)
        
        # 5. Loss improvement rate
        ax5 = fig.add_subplot(gs[1, 2])
        if 'train_loss' in df.columns:
            window = 10
            if len(df) > window:
                improvement_rate = []
                for i in range(window, len(df)):
                    old_loss = df['train_loss'].iloc[i-window]
                    new_loss = df['train_loss'].iloc[i]
                    rate = -np.log(max(new_loss/old_loss, 1e-8)) / window
                    improvement_rate.append(rate)
                
                ax5.plot(epochs[window:], improvement_rate, 'blue', linewidth=2)
                ax5.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
                ax5.set_xlabel('Epoch')
                ax5.set_ylabel('Improvement Rate')
                ax5.set_title(f'Loss Improvement Rate ({window}-epoch window)')
                ax5.grid(True, alpha=0.3)
        
        # 6. Reconstruction loss detail
        ax6 = fig.add_subplot(gs[2, :2])
        if 'train_recon_loss' in df.columns:
            ax6.plot(epochs, df['train_recon_loss'], 'g-', linewidth=2, label='Train Recon', alpha=0.8)
        if 'val_recon_loss' in df.columns:
            ax6.plot(epochs, df['val_recon_loss'], 'm-', linewidth=2, label='Val Recon', alpha=0.8)
        ax6.set_yscale('log')
        ax6.set_xlabel('Epoch')
        ax6.set_ylabel('Reconstruction Loss (log)')
        ax6.set_title('Reconstruction Loss Detail')
        ax6.legend()
        ax6.grid(True, alpha=0.3)
        
        # 7. Regularization loss detail
        ax7 = fig.add_subplot(gs[2, 2])
        if 'train_latent_reg_loss' in df.columns:
            ax7.plot(epochs, df['train_latent_reg_loss'], 'c-', linewidth=2, label='Train Reg', alpha=0.8)
        if 'val_latent_reg_loss' in df.columns:
            ax7.plot(epochs, df['val_latent_reg_loss'], 'y-', linewidth=2, label='Val Reg', alpha=0.8)
        ax7.set_yscale('log')
        ax7.set_xlabel('Epoch')
        ax7.set_ylabel('Regularization Loss (log)')
        ax7.set_title('Regularization Loss Detail')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        
        # 8. Loss distribution histogram
        ax8 = fig.add_subplot(gs[3, 0])
        if 'train_loss' in df.columns:
            ax8.hist(np.log10(df['train_loss']), bins=30, alpha=0.7, label='Train', color='blue')
        if 'val_loss' in df.columns:
            ax8.hist(np.log10(df['val_loss']), bins=30, alpha=0.7, label='Val', color='red')
        ax8.set_xlabel('Log10(Loss)')
        ax8.set_ylabel('Frequency')
        ax8.set_title('Loss Distribution')
        ax8.legend()
        ax8.grid(True, alpha=0.3)
        
        # 9. Stagnation periods
        ax9 = fig.add_subplot(gs[3, 1])
        if 'val_loss' in df.columns and len(df) > 20:
            # Calculate rolling improvement
            window = 20
            rolling_improvement = []
            for i in range(window, len(df)):
                old_val = df['val_loss'].iloc[i-window:i-window+5].mean()
                new_val = df['val_loss'].iloc[i-5:i].mean()
                improvement = (old_val - new_val) / old_val
                rolling_improvement.append(improvement)
            
            ax9.plot(epochs[window:], rolling_improvement, 'red', linewidth=2)
            ax9.axhline(y=0, color='gray', linestyle='--', alpha=0.5, label='No Improvement')
            ax9.axhline(y=0.01, color='green', linestyle='--', alpha=0.5, label='1% Improvement')
            ax9.set_xlabel('Epoch')
            ax9.set_ylabel('Rolling Improvement Rate')
            ax9.set_title(f'Validation Improvement ({window}-epoch window)')
            ax9.legend()
            ax9.grid(True, alpha=0.3)
        
        # 10. Training health summary
        ax10 = fig.add_subplot(gs[3, 2])
        ax10.axis('off')
        
        # Calculate health metrics
        health_text = "TRAINING HEALTH SUMMARY\n\n"
        
        if 'train_loss' in df.columns and 'val_loss' in df.columns:
            final_train = df['train_loss'].iloc[-1]
            final_val = df['val_loss'].iloc[-1]
            best_val = df['val_loss'].min()
            overfitting = final_val / final_train
            
            health_text += f"Final Train Loss: {final_train:.2e}\n"
            health_text += f"Final Val Loss: {final_val:.2e}\n"
            health_text += f"Best Val Loss: {best_val:.2e}\n"
            health_text += f"Overfitting Ratio: {overfitting:.2f}\n\n"
            
            # Health score
            score = 100
            if overfitting > 2.0:
                score -= 30
            if overfitting > 3.0:
                score -= 20
            
            improvement = df['train_loss'].iloc[0] / final_train
            if improvement < 10:
                score -= 25
            
            stagnant_epochs = 0
            best_so_far = float('inf')
            for val_loss in df['val_loss']:
                if val_loss < best_so_far * 0.999:
                    best_so_far = val_loss
                    stagnant_epochs = 0
                else:
                    stagnant_epochs += 1
            
            if stagnant_epochs > len(df) * 0.3:
                score -= 30
            
            score = max(score, 0)
            health_text += f"Health Score: {score}/100\n\n"
            
            if score < 50:
                health_text += "⚠️ POOR HEALTH\nConsider improved E2E training"
            elif score < 75:
                health_text += "⚡ MODERATE HEALTH\nSome optimization needed"
            else:
                health_text += "✅ GOOD HEALTH\nTraining progressing well"
        
        ax10.text(0.1, 0.9, health_text, transform=ax10.transAxes, fontsize=10,
                 verticalalignment='top', fontfamily='monospace')
        
        plt.suptitle(f'Comprehensive Training Analysis\nData: {os.path.basename(csv_file)}', 
                    fontsize=16, fontweight='bold')
        
        # Save plot
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, 'training_stagnation_analysis.png')
        plt.savefig(output_file, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"📊 Diagnostic plots saved to: {output_file}")
        return True
        
    except Exception as e:
        print(f"❌ Error creating diagnostic plots: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(description='Analyze training stagnation in SimulGenVAE')
    parser.add_argument('--csv_file', type=str, help='Path to training CSV file')
    parser.add_argument('--log_dir', type=str, default='./output', help='Directory containing training logs')
    parser.add_argument('--plot_analysis', action='store_true', help='Generate diagnostic plots')
    parser.add_argument('--auto_find', action='store_true', help='Automatically find latest CSV file')
    
    args = parser.parse_args()
    
    print("🔍 SimulGenVAE Training Stagnation Analysis Tool")
    print("=" * 60)
    
    csv_file = None
    
    # Find CSV file
    if args.csv_file:
        csv_file = args.csv_file
    elif args.auto_find or not args.csv_file:
        # Look for latest CSV file
        pattern = os.path.join(args.log_dir, "e2e_training_data_*.csv")
        csv_files = glob.glob(pattern)
        
        if csv_files:
            csv_file = max(csv_files, key=os.path.getctime)  # Most recent file
            print(f"📁 Auto-detected latest CSV file: {csv_file}")
        else:
            print(f"❌ No training CSV files found in {args.log_dir}")
            print("   Expected pattern: e2e_training_data_*.csv")
            return
    
    if not csv_file or not os.path.exists(csv_file):
        print(f"❌ CSV file not found: {csv_file}")
        return
    
    # Generate analysis report
    success = generate_comprehensive_report(csv_file)
    
    # Generate diagnostic plots if requested
    if args.plot_analysis and success:
        create_diagnostic_plots(csv_file, args.log_dir)
    
    print("\n🎯 Next Steps:")
    print("   1. Update your condition.txt with improved settings")
    print("   2. Run: python SimulGen-VAE.py --lc_only=1 --preset=1")
    print("   3. Monitor the improved training progress")
    print("   4. Re-run this analysis tool to compare results")

if __name__ == "__main__":
    main()
