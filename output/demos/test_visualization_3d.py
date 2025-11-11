#!/usr/bin/env python3
"""
Additional test for 3D Fisher analysis visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, 'src')

from core import AnalysisResults, AnalysisParameters
from visualization import plot_fisher_analysis

def test_3d_fisher_analysis():
    """Test plot_fisher_analysis with 3D data (tau vs time_bins vs sigma)."""
    print("Testing 3D Fisher analysis visualization...")
    
    # Create test data for 3D Fisher analysis
    tau_range = np.arange(0.5, 3.0, 0.5)
    time_bins_range = 2**(np.arange(6, 10))
    sigma_range = np.array([0.01, 0.1, 0.5, 1.0])
    
    # Generate synthetic 3D F-values
    f_values = np.zeros((len(tau_range), len(time_bins_range), len(sigma_range)))
    for i, tau in enumerate(tau_range):
        for j, tn in enumerate(time_bins_range):
            for k, sigma in enumerate(sigma_range):
                # F-value decreases with more bins and larger IRF width
                f_values[i, j, k] = (1.0 / np.sqrt(tn)) * (1.0 + 0.1 * tau) * (1.0 + sigma)
    
    # Create parameters and results
    params = AnalysisParameters(
        lifetimes=tau_range,
        time_bins=time_bins_range,
        irf_sigmas=sigma_range,
        repetition_period=25.0
    )
    
    results = AnalysisResults(
        f_values=f_values,
        parameters=params,
        metadata={'test': True},
        analysis_type='gaussian_fisher_analysis'
    )
    
    # Test plotting with tau_index
    try:
        fig = plot_fisher_analysis(results, tau_index=2, log_scale=True,
                                  save_path='test_3d_tau_slice.png')
        plt.close(fig)
        print("✓ 3D Fisher analysis (tau slice) test passed")
    except Exception as e:
        print(f"✗ 3D Fisher analysis (tau slice) test failed: {e}")
        return False
    
    # Test plotting with sigma_index
    try:
        fig = plot_fisher_analysis(results, sigma_index=1, log_scale=True,
                                  save_path='test_3d_sigma_slice.png')
        plt.close(fig)
        print("✓ 3D Fisher analysis (sigma slice) test passed")
    except Exception as e:
        print(f"✗ 3D Fisher analysis (sigma slice) test failed: {e}")
        return False
    
    # Test plotting multiple subplots
    try:
        fig = plot_fisher_analysis(results, log_scale=True,
                                  save_path='test_3d_multi.png')
        plt.close(fig)
        print("✓ 3D Fisher analysis (multi-panel) test passed")
    except Exception as e:
        print(f"✗ 3D Fisher analysis (multi-panel) test failed: {e}")
        return False
    
    return True


if __name__ == "__main__":
    plt.switch_backend('Agg')
    success = test_3d_fisher_analysis()
    sys.exit(0 if success else 1)
