"""
Test script for specialized analysis plotting functions.

This script tests the newly implemented specialized plotting functions:
- plot_separability_analysis()
- plot_resolving_power()
- plot_loss_analysis()
- plot_comparison_grid()
"""

import numpy as np
import matplotlib.pyplot as plt
from src.visualization import (
    plot_separability_analysis,
    plot_resolving_power,
    plot_loss_analysis,
    plot_comparison_grid
)
from src.core import AnalysisResults, AnalysisParameters

# Set random seed for reproducibility
np.random.seed(42)

print("Testing specialized analysis plotting functions...")
print("=" * 60)

# Test 1: plot_separability_analysis
print("\n1. Testing plot_separability_analysis()...")
try:
    tau1 = 2.0  # Reference lifetime
    tau2_range = np.linspace(0.5, 10.0, 50)
    
    # Simulate separability measure (increases with lifetime difference)
    tau_diff = np.abs(tau2_range - tau1)
    f_values = tau_diff / (0.1 + 0.05 * tau_diff)  # Simulated separability
    
    params = {
        'time_bins': 512,
        'repetition_period': 25.0,
        'irf_sigma': 0.1
    }
    
    fig = plot_separability_analysis(
        tau1=tau1,
        tau2_range=tau2_range,
        f_values=f_values,
        threshold=2.0,
        params=params,
        save_path='test_separability.png'
    )
    plt.close(fig)
    print("   ✓ plot_separability_analysis() passed")
    print("   ✓ Saved to: test_separability.png")
except Exception as e:
    print(f"   ✗ plot_separability_analysis() failed: {e}")

# Test 2: plot_resolving_power
print("\n2. Testing plot_resolving_power()...")
try:
    tau_range = np.linspace(0.5, 10.0, 30)
    
    # Simulate minimum resolvable difference (typically increases with tau)
    min_resolvable_diff = 0.1 * tau_range + 0.05
    
    # Create comparison data
    comparison_data = {
        'Gaussian σ=0.1': {
            'tau': tau_range,
            'diff': 0.12 * tau_range + 0.08
        },
        'Gaussian σ=0.5': {
            'tau': tau_range,
            'diff': 0.15 * tau_range + 0.15
        }
    }
    
    params = {
        'time_bins': 512,
        'repetition_period': 25.0,
        'threshold': 2.0
    }
    
    fig = plot_resolving_power(
        tau_range=tau_range,
        min_resolvable_diff=min_resolvable_diff,
        params=params,
        comparison_data=comparison_data,
        save_path='test_resolving_power.png'
    )
    plt.close(fig)
    print("   ✓ plot_resolving_power() passed")
    print("   ✓ Saved to: test_resolving_power.png")
except Exception as e:
    print(f"   ✗ plot_resolving_power() failed: {e}")

# Test 3: plot_loss_analysis
print("\n3. Testing plot_loss_analysis()...")
try:
    tau_range = np.linspace(0.5, 10.0, 30)
    
    # Simulate F-values (ideal vs actual)
    f_values_ideal = 100 * tau_range**2  # Ideal case
    f_values_actual = f_values_ideal * (0.7 + 0.2 * np.exp(-tau_range/3))  # Actual with loss
    
    params = {
        'time_bins': 512,
        'repetition_period': 25.0,
        'irf_sigma': 0.25
    }
    
    fig = plot_loss_analysis(
        tau_range=tau_range,
        f_values_ideal=f_values_ideal,
        f_values_actual=f_values_actual,
        irf_label='Gaussian σ=0.25 ns',
        params=params,
        save_path='test_loss_analysis.png'
    )
    plt.close(fig)
    print("   ✓ plot_loss_analysis() passed")
    print("   ✓ Saved to: test_loss_analysis.png")
except Exception as e:
    print(f"   ✗ plot_loss_analysis() failed: {e}")

# Test 4: plot_comparison_grid
print("\n4. Testing plot_comparison_grid()...")
try:
    # Create sample AnalysisResults objects
    lifetimes = np.linspace(0.5, 10.0, 20)
    time_bins = 2**(np.arange(5) + 7)  # 128, 256, 512, 1024, 2048
    
    data_dict = {}
    
    for i, sigma in enumerate([0.01, 0.1, 0.5]):
        # Simulate F-values for different IRF conditions
        tau_grid, bins_grid = np.meshgrid(lifetimes, time_bins, indexing='ij')
        f_values = (tau_grid**2 * bins_grid) / (1 + sigma * 10)
        
        params = AnalysisParameters(
            repetition_period=25.0,
            lifetimes=lifetimes,
            time_bins=time_bins
        )
        
        results = AnalysisResults(
            f_values=f_values,
            parameters=params,
            metadata={'irf_sigma': sigma},
            analysis_type='gaussian_irf'
        )
        
        data_dict[f'Gaussian σ={sigma} ns'] = results
    
    fig = plot_comparison_grid(
        data_dict=data_dict,
        plot_type='heatmap',
        save_path='test_comparison_grid.png'
    )
    plt.close(fig)
    print("   ✓ plot_comparison_grid() passed")
    print("   ✓ Saved to: test_comparison_grid.png")
except Exception as e:
    print(f"   ✗ plot_comparison_grid() failed: {e}")

print("\n" + "=" * 60)
print("All specialized plotting function tests completed!")
print("\nGenerated test plots:")
print("  - test_separability.png")
print("  - test_resolving_power.png")
print("  - test_loss_analysis.png")
print("  - test_comparison_grid.png")
