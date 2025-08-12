#!/usr/bin/env python3
"""Test script to verify enhanced plotting functionality"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add modules path
sys.path.append('modules')

from plotter import temporal_plotter, nodal_plotter, dual_view_plotter

def test_plotting_functions():
    """Test the enhanced plotting functions with sample data"""
    print("Testing enhanced plotting functionality...")
    
    # Create sample data matching SimulGenVAE dimensions
    num_param = 10
    num_time = 200  # Temporal dimension (0~200)
    num_node = 1000  # Simplified nodal dimension (normally ~95k)
    
    # Generate synthetic data
    np.random.seed(42)
    test_data = np.random.randn(num_param, num_time, num_node) * 0.1
    
    # Add some structure to the data
    for i in range(num_param):
        for t in range(num_time):
            # Add temporal dynamics
            test_data[i, t, :] += 0.5 * np.sin(2 * np.pi * t / 50) * np.exp(-0.01 * t)
            # Add spatial patterns
            test_data[i, t, :] += 0.3 * np.sin(2 * np.pi * np.arange(num_node) / 100)
    
    print(f"Test data shape: {test_data.shape}")
    print(f"Data range: [{test_data.min():.3f}, {test_data.max():.3f}]")
    
    # Test 1: Temporal plotter
    print("\n1. Testing temporal_plotter...")
    plt.figure(figsize=(12, 6))
    temporal_plotter(test_data, axis=0, a=0, c=100, print_graph="0", n=3)
    plt.title("Test: Temporal Evolution (3 parameters at node 100)")
    plt.savefig("test_temporal.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Test 2: Nodal plotter
    print("2. Testing nodal_plotter...")
    plt.figure(figsize=(12, 6))
    nodal_plotter(test_data, axis=1, a=0, c=0, print_graph="0", n=3, time_idx=100)
    plt.title("Test: Nodal Distribution (3 time snapshots for param 0)")
    plt.savefig("test_nodal.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Test 3: Dual view plotter
    print("3. Testing dual_view_plotter...")
    fig = dual_view_plotter(test_data, param_idx=0, print_graph="0")
    fig.suptitle("Test: Dual View - Temporal vs Nodal")
    plt.savefig("test_dual_view.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\nTest completed successfully!")
    print("Generated test plots:")
    print("  - test_temporal.png")
    print("  - test_nodal.png") 
    print("  - test_dual_view.png")
    
    return True

if __name__ == "__main__":
    test_plotting_functions()