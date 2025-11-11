#!/usr/bin/env python3
"""
Test script for core visualization functions.

This script tests the three core plotting functions implemented in task 5.1:
- plot_fisher_analysis()
- plot_monte_carlo_results()
- plot_irf_comparison()
"""

import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add src to path for imports
sys.path.insert(0, 'src')

from core import AnalysisResults, AnalysisParameters
from visualization import plot_fisher_analysis, plot_monte_carlo_results, plot_irf_comparison

def test_plot_fisher_analysis():
    """Test the plot_fisher_analysis function."""
    print("Testing plot_fisher_analysis...")
    
    # Create test data for 2D Fisher analysis (tau vs time_bins)
    tau_range = np.arange(0.5, 5.0, 0.5)
    time_bins_range = 2**(np.arange(6, 12))
    
    # Generate synthetic F-values (decreasing with more bins, varying with lifetime)
    f_values = np.zeros((len(tau_range), len(time_bins_range)))
    for i, tau in enumerate(tau_range):
        for j, tn in enumerate(time_bins_range):
            # Synthetic F-value that decreases with more bins and varies with lifetime
            f_values[i, j] = (1.0 / np.sqrt(tn)) * (1.0 + 0.1 * tau)
    
    # Create parameters and results objects
    params = AnalysisParameters(
        lifetimes=tau_range,
        time_bins=time_bins_range,
        repetition_period=25.0
    )
    
    results = AnalysisResults(
        f_values=f_values,
        parameters=params,
        metadata={'test': True},
        analysis_type='dirac_fisher_analysis'
    )
    
    # Test the plotting function
    try:
        fig = plot_fisher_analysis(results, log_scale=True, save_path='test_fisher_plot.png')
        plt.close(fig)  # Close to avoid display issues
        print("✓ plot_fisher_analysis test passed")
        return True
    except Exception as e:
        print(f"✗ plot_fisher_analysis test failed: {e}")
        return False


def test_plot_monte_carlo_results():
    """Test the plot_monte_carlo_results function."""
    print("Testing plot_monte_carlo_results...")
    
    # Create test Monte Carlo results
    tau_values = np.arange(0.5, 5.0, 0.5)
    
    # Synthetic Monte Carlo results
    results = {
        'tau_values': tau_values,
        'tau_fitted': tau_values + np.random.normal(0, 0.05, len(tau_values)),  # Small fitting errors
        'tau_errors': np.random.uniform(0.02, 0.1, len(tau_values)),
        'f_values': 1.0 / np.sqrt(tau_values),  # Synthetic F-values
        'f_value_errors': np.random.uniform(0.01, 0.05, len(tau_values)),
        'success_rate': 0.95
    }
    
    # Test precision plot
    try:
        fig = plot_monte_carlo_results(results, plot_type='precision', 
                                     save_path='test_mc_precision.png')
        plt.close(fig)
        print("✓ plot_monte_carlo_results (precision) test passed")
        
        # Test with comparison data
        comparison_data = {
            'tau_values': tau_values,
            'f_values': 1.1 / np.sqrt(tau_values)  # Slightly different for comparison
        }
        
        fig = plot_monte_carlo_results(results, plot_type='comparison',
                                     comparison_data=comparison_data,
                                     save_path='test_mc_comparison.png')
        plt.close(fig)
        print("✓ plot_monte_carlo_results (comparison) test passed")
        return True
        
    except Exception as e:
        print(f"✗ plot_monte_carlo_results test failed: {e}")
        return False


def test_plot_irf_comparison():
    """Test the plot_irf_comparison function."""
    print("Testing plot_irf_comparison...")
    
    # Create test time array
    time_array = np.linspace(0, 10, 1000)
    
    # Create different IRF shapes
    # Dirac (approximated as very narrow Gaussian)
    dirac_irf = np.exp(-(time_array - 5)**2 / (2 * 0.01**2))
    
    # Gaussian IRFs with different widths
    gaussian_01 = np.exp(-(time_array - 5)**2 / (2 * 0.1**2))
    gaussian_05 = np.exp(-(time_array - 5)**2 / (2 * 0.5**2))
    
    # Rectangular IRF (approximated)
    rectangular = np.where(np.abs(time_array - 5) <= 0.5, 1.0, 0.0)
    
    # Create IRF data dictionary
    irf_data = {
        'Dirac (σ≈0)': dirac_irf,
        'Gaussian σ=0.1 ns': gaussian_01,
        'Gaussian σ=0.5 ns': gaussian_05,
        'Rectangular': rectangular
    }
    
    # Test the plotting function
    try:
        fig = plot_irf_comparison(irf_data, time_array, normalize=True,
                                save_path='test_irf_comparison.png')
        plt.close(fig)
        print("✓ plot_irf_comparison test passed")
        return True
    except Exception as e:
        print(f"✗ plot_irf_comparison test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("Testing core visualization functions...")
    print("=" * 50)
    
    # Set matplotlib to non-interactive backend to avoid display issues
    plt.switch_backend('Agg')
    
    tests = [
        test_plot_fisher_analysis,
        test_plot_monte_carlo_results,
        test_plot_irf_comparison
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("✓ All core plotting functions implemented successfully!")
        return True
    else:
        print("✗ Some tests failed")
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)