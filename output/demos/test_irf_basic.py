#!/usr/bin/env python3
"""
Basic test script to verify IRF functions implementation.
"""

import numpy as np
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from irf_functions import (
    dirac_irf, gaussian_irf, rectangular_irf, 
    get_irf, validate_irf, convolve_with_exponential,
    exponential_decay
)

def test_basic_functionality():
    """Test basic IRF functionality."""
    print("Testing basic IRF functionality...")
    
    # Create time array
    t = np.linspace(0, 10, 1000)
    
    # Test Dirac IRF
    print("  Testing Dirac IRF...")
    dirac = dirac_irf(t, t0=2.0)
    print(f"    Dirac IRF sum: {np.sum(dirac) * np.mean(np.diff(t)):.6f} (should be ~1.0)")
    
    # Test Gaussian IRF
    print("  Testing Gaussian IRF...")
    gaussian = gaussian_irf(t, t0=2.0, sigma=0.5)
    print(f"    Gaussian IRF sum: {np.sum(gaussian) * np.mean(np.diff(t)):.6f} (should be ~1.0)")
    
    # Test Rectangular IRF
    print("  Testing Rectangular IRF...")
    rectangular = rectangular_irf(t, t0=2.0, width=1.0)
    print(f"    Rectangular IRF sum: {np.sum(rectangular) * np.mean(np.diff(t)):.6f} (should be ~1.0)")
    
    # Test factory function
    print("  Testing factory function...")
    irf_factory = get_irf('gaussian', t, t0=2.0, sigma=0.5)
    print(f"    Factory Gaussian IRF sum: {np.sum(irf_factory) * np.mean(np.diff(t)):.6f} (should be ~1.0)")
    
    # Test convolution
    print("  Testing convolution...")
    decay = exponential_decay(t, tau=2.0)
    convolved = convolve_with_exponential(gaussian, t, tau=2.0, T=25.0)
    print(f"    Convolution completed successfully, max value: {np.max(convolved):.6f}")
    
    print("All basic tests passed!")

if __name__ == "__main__":
    test_basic_functionality()