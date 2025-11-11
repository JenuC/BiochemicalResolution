"""
Test script to verify the Gaussian IRF notebook can be executed.
Tests imports and basic functionality without running the full analysis.
"""

import sys
import numpy as np
from pathlib import Path

# Add src to path
sys.path.insert(0, 'src')

print("=" * 70)
print("Testing Gaussian IRF Notebook Dependencies")
print("=" * 70)

# Test 1: Import modules
print("\n1. Testing module imports...")
try:
    from fisher_information import (
        gaussian_irf_analysis,
        dirac_irf_analysis,
        calculate_f_value
    )
    from visualization import (
        setup_plot_style,
        export_figure
    )
    from core import AnalysisParameters
    print("   ✓ All modules imported successfully")
except ImportError as e:
    print(f"   ✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Verify functions exist
print("\n2. Testing function availability...")
functions = [
    ('gaussian_irf_analysis', gaussian_irf_analysis),
    ('dirac_irf_analysis', dirac_irf_analysis),
    ('calculate_f_value', calculate_f_value),
    ('setup_plot_style', setup_plot_style),
    ('AnalysisParameters', AnalysisParameters)
]

for name, func in functions:
    if callable(func) or isinstance(func, type):
        print(f"   ✓ {name} is available")
    else:
        print(f"   ✗ {name} is not callable")
        sys.exit(1)

# Test 3: Test single F-value calculation
print("\n3. Testing single F-value calculation...")
try:
    # Test Dirac IRF
    F_dirac = calculate_f_value(
        tau=2.5, T=25.0, time_bins=256,
        irf_type='dirac', num_photons=75000
    )
    print(f"   ✓ Dirac IRF F-value: {F_dirac:.6f}")
    
    # Test Gaussian IRF
    F_gaussian = calculate_f_value(
        tau=2.5, T=25.0, time_bins=256,
        irf_type='gaussian', irf_params={'sigma': 0.25},
        num_photons=75000
    )
    print(f"   ✓ Gaussian IRF F-value: {F_gaussian:.6f}")
    
    # Verify Gaussian > Dirac (worse precision)
    if F_gaussian > F_dirac:
        print(f"   ✓ Gaussian F-value > Dirac F-value (expected)")
    else:
        print(f"   ✗ Unexpected: Gaussian F-value <= Dirac F-value")
        sys.exit(1)
        
except Exception as e:
    print(f"   ✗ Calculation failed: {e}")
    sys.exit(1)

# Test 4: Test parameter arrays
print("\n4. Testing parameter arrays...")
try:
    tau_range = np.arange(0.2, 15.0, 0.4)
    time_bins_range = 2 ** (np.arange(9) + 2)
    sigma_range = np.array([0.01, 0.1, 0.25, 0.5, 1.0, 2.0])
    
    print(f"   ✓ Lifetime range: {len(tau_range)} values")
    print(f"   ✓ Time bins range: {len(time_bins_range)} values")
    print(f"   ✓ Sigma range: {len(sigma_range)} values")
    print(f"   ✓ Total calculations: {len(tau_range) * len(time_bins_range) * len(sigma_range)}")
except Exception as e:
    print(f"   ✗ Parameter array creation failed: {e}")
    sys.exit(1)

# Test 5: Test data directory
print("\n5. Testing data directory...")
try:
    output_dir = Path('data/generated')
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"   ✓ Data directory exists: {output_dir}")
except Exception as e:
    print(f"   ✗ Data directory creation failed: {e}")
    sys.exit(1)

# Test 6: Test AnalysisParameters
print("\n6. Testing AnalysisParameters...")
try:
    params = AnalysisParameters(
        repetition_period=25.0,
        lifetimes=tau_range,
        time_bins=time_bins_range,
        irf_sigmas=sigma_range,
        num_photons=75000
    )
    print(f"   ✓ AnalysisParameters created successfully")
    print(f"   ✓ Repetition period: {params.repetition_period} ns")
    print(f"   ✓ Number of photons: {params.num_photons}")
except Exception as e:
    print(f"   ✗ AnalysisParameters creation failed: {e}")
    sys.exit(1)

# Test 7: Test small parameter sweep
print("\n7. Testing small parameter sweep...")
try:
    # Test with minimal parameters
    tau_test = np.array([1.0, 2.0])
    bins_test = np.array([64, 256])
    sigma_test = np.array([0.1, 0.5])
    
    print(f"   Testing {len(tau_test)} × {len(bins_test)} × {len(sigma_test)} = {len(tau_test) * len(bins_test) * len(sigma_test)} calculations...")
    
    results = gaussian_irf_analysis(
        tau_range=tau_test,
        time_bins_range=bins_test,
        sigma_range=sigma_test,
        T=25.0,
        num_photons=75000,
        progress=False
    )
    
    print(f"   ✓ Small parameter sweep completed")
    print(f"   ✓ Results shape: {results.f_values.shape}")
    print(f"   ✓ Successful calculations: {results.metadata['successful_calculations']}")
    print(f"   ✓ Failed calculations: {results.metadata['failed_calculations']}")
    
    # Verify shape
    expected_shape = (len(tau_test), len(bins_test), len(sigma_test))
    if results.f_values.shape == expected_shape:
        print(f"   ✓ Results shape matches expected: {expected_shape}")
    else:
        print(f"   ✗ Shape mismatch: got {results.f_values.shape}, expected {expected_shape}")
        sys.exit(1)
        
except Exception as e:
    print(f"   ✗ Parameter sweep failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Summary
print("\n" + "=" * 70)
print("All Tests Passed! ✓")
print("=" * 70)
print("\nThe Gaussian IRF notebook is ready to execute.")
print("Run the notebook to generate full results and figures.")
print("\nEstimated full execution time: 5-10 minutes")
print("=" * 70)
