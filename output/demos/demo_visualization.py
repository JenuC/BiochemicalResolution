#!/usr/bin/env python3
"""
Demonstration of the three core visualization functions.

This script demonstrates the usage of:
1. plot_fisher_analysis() - Fisher information visualization
2. plot_monte_carlo_results() - Monte Carlo simulation results
3. plot_irf_comparison() - IRF shape comparison

Run this to see example outputs of all three plotting functions.
"""

import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, 'src')

from core import AnalysisResults, AnalysisParameters
from visualization import (plot_fisher_analysis, plot_monte_carlo_results, 
                          plot_irf_comparison, setup_plot_style)

def demo_fisher_analysis():
    """Demonstrate Fisher information analysis plotting."""
    print("Generating Fisher analysis plot...")
    
    # Simulate realistic Fisher analysis data
    tau_range = np.arange(0.2, 15, 0.4)
    time_bins_range = 2**(np.arange(9)+2)
    
    # Generate F-values that decrease with more bins (better precision)
    f_values = np.zeros((len(tau_range), len(time_bins_range)))
    for i, tau in enumerate(tau_range):
        for j, tn in enumerate(time_bins_range):
            # Realistic F-value calculation
            f_values[i, j] = np.sqrt(75000) / (tau * np.sqrt(tn * 0.5))
    
    params = AnalysisParameters(
        lifetimes=tau_range,
        time_bins=time_bins_range,
        repetition_period=25.0
    )
    
    results = AnalysisResults(
        f_values=f_values,
        parameters=params,
        metadata={'description': 'Dirac IRF Fisher Information Analysis'},
        analysis_type='dirac_fisher_analysis'
    )
    
    fig = plot_fisher_analysis(results, log_scale=True, 
                              title='Fisher Information Analysis - Dirac IRF',
                              save_path='demo_fisher.png')
    plt.close(fig)
    print("✓ Saved: demo_fisher.png")


def demo_monte_carlo():
    """Demonstrate Monte Carlo results plotting."""
    print("Generating Monte Carlo results plots...")
    
    # Simulate Monte Carlo results
    tau_values = np.arange(1.0, 10.0, 1.0)
    
    # Simulate fitted values with realistic noise
    tau_fitted = tau_values + np.random.normal(0, 0.1, len(tau_values))
    tau_errors = np.abs(np.random.normal(0.1, 0.02, len(tau_values)))
    
    # Calculate F-values (precision measure)
    f_values_mc = tau_errors * np.sqrt(75000) / tau_values
    f_value_errors = f_values_mc * 0.1  # 10% uncertainty
    
    mc_results = {
        'tau_values': tau_values,
        'tau_fitted': tau_fitted,
        'tau_errors': tau_errors,
        'f_values': f_values_mc,
        'f_value_errors': f_value_errors,
        'success_rate': 0.98
    }
    
    # Generate Fisher information comparison data
    f_values_fisher = tau_errors * np.sqrt(75000) / tau_values * 0.95  # Slightly better
    
    comparison_data = {
        'tau_values': tau_values,
        'f_values': f_values_fisher
    }
    
    # Create precision plot
    fig = plot_monte_carlo_results(mc_results, plot_type='precision',
                                  comparison_data=comparison_data,
                                  title='Monte Carlo Precision Analysis',
                                  save_path='demo_mc_precision.png')
    plt.close(fig)
    print("✓ Saved: demo_mc_precision.png")
    
    # Create comparison plot
    fig = plot_monte_carlo_results(mc_results, plot_type='comparison',
                                  comparison_data=comparison_data,
                                  title='MC vs Fisher Information',
                                  save_path='demo_mc_comparison.png')
    plt.close(fig)
    print("✓ Saved: demo_mc_comparison.png")


def demo_irf_comparison():
    """Demonstrate IRF comparison plotting."""
    print("Generating IRF comparison plot...")
    
    # Create realistic time array
    T = 25.0  # Repetition period
    tn = 1024  # Number of time bins
    time_array = np.linspace(0, T, tn)
    
    # Create different IRF shapes
    t0 = T / 2  # Center position
    
    # Dirac (approximated as very narrow Gaussian)
    dirac_irf = np.exp(-(time_array - t0)**2 / (2 * 0.01**2))
    
    # Gaussian IRFs with different widths
    gaussian_01 = np.exp(-(time_array - t0)**2 / (2 * 0.1**2))
    gaussian_025 = np.exp(-(time_array - t0)**2 / (2 * 0.25**2))
    gaussian_05 = np.exp(-(time_array - t0)**2 / (2 * 0.5**2))
    gaussian_10 = np.exp(-(time_array - t0)**2 / (2 * 1.0**2))
    
    # Rectangular IRF
    width = 1.0
    rectangular = np.where(np.abs(time_array - t0) <= width/2, 1.0, 0.0)
    
    irf_data = {
        'Dirac (σ≈0)': dirac_irf,
        'Gaussian σ=0.1 ns': gaussian_01,
        'Gaussian σ=0.25 ns': gaussian_025,
        'Gaussian σ=0.5 ns': gaussian_05,
        'Gaussian σ=1.0 ns': gaussian_10,
        'Rectangular (w=1.0 ns)': rectangular
    }
    
    fig = plot_irf_comparison(irf_data, time_array, normalize=True,
                            title='Instrument Response Function Comparison',
                            save_path='demo_irf_comparison.png')
    plt.close(fig)
    print("✓ Saved: demo_irf_comparison.png")


def main():
    """Run all demonstrations."""
    print("=" * 60)
    print("Core Visualization Functions Demonstration")
    print("=" * 60)
    print()
    
    # Set matplotlib backend
    plt.switch_backend('Agg')
    
    # Set up consistent plot style
    setup_plot_style()
    
    # Run demonstrations
    demo_fisher_analysis()
    print()
    
    demo_monte_carlo()
    print()
    
    demo_irf_comparison()
    print()
    
    print("=" * 60)
    print("✓ All demonstration plots generated successfully!")
    print()
    print("Generated files:")
    print("  - demo_fisher.png")
    print("  - demo_mc_precision.png")
    print("  - demo_mc_comparison.png")
    print("  - demo_irf_comparison.png")
    print("=" * 60)


if __name__ == "__main__":
    main()
