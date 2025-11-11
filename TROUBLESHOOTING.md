# Troubleshooting Guide

This guide addresses common issues and their solutions when using the Instrument Response Function Analysis package.

## Installation Issues

### Issue: `uv` command not found

**Symptoms:**
```bash
'uv' is not recognized as an internal or external command
```

**Solution:**
Install uv using pip or your system package manager:
```bash
# Using pip
pip install uv

# Or follow installation guide
# https://docs.astral.sh/uv/getting-started/installation/
```

**Alternative:**
Use standard pip instead of uv:
```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Issue: Package import errors

**Symptoms:**
```python
ModuleNotFoundError: No module named 'src'
```

**Solution:**
Install the package in development mode:
```bash
pip install -e .
```

Or add the src directory to your Python path:
```python
import sys
import os
sys.path.insert(0, os.path.abspath('src'))
```

### Issue: Dependency conflicts

**Symptoms:**
```
ERROR: pip's dependency resolver does not currently take into account all the packages that are installed
```

**Solution:**
Create a fresh virtual environment:
```bash
# Remove old environment
rm -rf venv  # Windows: rmdir /s venv

# Create new environment
python -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

## Notebook Execution Issues

### Issue: Kernel dies when running notebooks

**Symptoms:**
- Jupyter kernel crashes during execution
- "Kernel Restarting" message appears

**Possible Causes & Solutions:**

1. **Memory Issues:**
   - Reduce parameter sweep ranges
   - Decrease number of Monte Carlo iterations
   - Close other applications

2. **Numerical Instability:**
   - Check for extreme parameter values
   - Ensure lifetimes are positive
   - Verify time bins are reasonable (e.g., 4-1024)

3. **Missing Dependencies:**
   ```bash
   pip install jupyter ipykernel
   python -m ipykernel install --user
   ```

### Issue: Plots not displaying in notebooks

**Symptoms:**
- Empty output cells where plots should appear
- No error messages

**Solution:**
Add matplotlib backend configuration:
```python
%matplotlib inline
import matplotlib.pyplot as plt
```

Or try different backend:
```python
%matplotlib notebook  # Interactive plots
%matplotlib widget    # ipympl widgets
```

### Issue: "Proactor event loop" warning on Windows

**Symptoms:**
```
RuntimeWarning: Proactor event loop does not implement add_reader family of methods
```

**Solution:**
This is a harmless warning related to Windows event loop. To suppress:
```python
import warnings
warnings.filterwarnings('ignore', category=RuntimeWarning)
```

Or use the recommended event loop policy:
```python
import asyncio
asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
```

## Numerical Issues

### Issue: Fisher information matrix is not positive definite

**Symptoms:**
```
ValueError: Fisher information matrix is not positive definite
```

**Possible Causes:**
- Lifetime too short relative to time bin width
- Repetition period too short
- IRF width comparable to or larger than lifetime

**Solution:**
Adjust parameters to physically reasonable ranges:
```python
params = AnalysisParameters(
    repetition_period=25.0,  # Should be >> max lifetime
    lifetimes=np.arange(0.5, 10, 0.5),  # Avoid very short lifetimes
    time_bins=2**(np.arange(5, 10)),  # Reasonable bin counts
    irf_sigmas=np.array([0.01, 0.1, 0.5])  # IRF << lifetime
)
```

### Issue: Monte Carlo fitting fails to converge

**Symptoms:**
```
OptimizeWarning: Covariance of the parameters could not be estimated
```

**Possible Causes:**
- Insufficient photon counts
- Poor initial guess
- Extreme parameter values

**Solutions:**

1. **Increase photon counts:**
   ```python
   params.num_photons = 100000  # More photons = better fits
   ```

2. **Improve initial guess:**
   ```python
   # Use Fisher information to guide initial guess
   initial_tau = true_tau * 0.9  # Start close to true value
   ```

3. **Add bounds to fitting:**
   ```python
   from scipy.optimize import curve_fit
   bounds = ([0, 0], [np.inf, np.inf])  # Positive parameters only
   ```

### Issue: NaN or Inf values in results

**Symptoms:**
- Results contain NaN or Inf
- Plots show empty or strange values

**Possible Causes:**
- Division by zero
- Numerical overflow/underflow
- Invalid parameter combinations

**Solution:**
Add validation checks:
```python
# Check for valid inputs
assert np.all(np.isfinite(tau_range)), "Lifetimes must be finite"
assert np.all(tau_range > 0), "Lifetimes must be positive"

# Check outputs
if not np.all(np.isfinite(F_values)):
    print("Warning: Non-finite values detected")
    F_values = np.nan_to_num(F_values, nan=0.0, posinf=1e10, neginf=0.0)
```

## Data File Issues

### Issue: Cannot load .mat files

**Symptoms:**
```
OSError: Unable to open file (file signature not found)
```

**Solution:**
Ensure scipy is installed and up to date:
```bash
pip install --upgrade scipy
```

For MATLAB v7.3+ files, use h5py:
```python
import h5py
with h5py.File('data.mat', 'r') as f:
    data = f['variable_name'][:]
```

### Issue: .npy files have wrong shape

**Symptoms:**
```
ValueError: could not broadcast input array from shape (37,9) into shape (37,9,6)
```

**Solution:**
Check parameter ranges match expected dimensions:
```python
# Verify shapes
print(f"Lifetimes: {len(tau_range)}")
print(f"Time bins: {len(time_bins_range)}")
print(f"Sigmas: {len(sigma_range)}")
print(f"Expected shape: ({len(tau_range)}, {len(time_bins_range)}, {len(sigma_range)})")
```

## Performance Issues

### Issue: Analysis takes too long

**Symptoms:**
- Parameter sweeps run for hours
- Notebook cells never complete

**Solutions:**

1. **Reduce parameter space:**
   ```python
   # Use coarser sampling
   lifetimes = np.arange(0.5, 10, 1.0)  # Fewer points
   time_bins = 2**np.array([5, 7, 9])   # Fewer bins
   ```

2. **Use parallel processing:**
   ```python
   from multiprocessing import Pool
   with Pool() as pool:
       results = pool.map(analysis_function, parameter_list)
   ```

3. **Profile code to find bottlenecks:**
   ```python
   import cProfile
   cProfile.run('your_analysis_function()')
   ```

### Issue: High memory usage

**Symptoms:**
- System becomes unresponsive
- Out of memory errors

**Solutions:**

1. **Process in batches:**
   ```python
   # Instead of all at once
   for tau_batch in np.array_split(tau_range, 10):
       results_batch = analyze(tau_batch)
       save_results(results_batch)
   ```

2. **Use memory-efficient data types:**
   ```python
   # Use float32 instead of float64 if precision allows
   F_values = np.zeros((n, m), dtype=np.float32)
   ```

3. **Clear intermediate results:**
   ```python
   import gc
   del large_array
   gc.collect()
   ```

## Validation Issues

### Issue: Results don't match Mathematica

**Symptoms:**
- Different numerical values
- Plots look different

**Expected Behavior:**
Some differences are expected and acceptable:

1. **Floating-point precision:** Differences < 1e-6 are normal
2. **Parameter ranges:** Ensure same ranges are used
3. **Random seeds:** Monte Carlo results vary without fixed seeds

**Solution:**
```python
# Fix random seed for reproducibility
np.random.seed(42)

# Check parameter ranges match
print("Python tau range:", tau_range)
print("Mathematica tau range:", [0.2, 0.6, ..., 14.6])

# Compare with tolerance
np.allclose(python_result, mathematica_result, rtol=1e-5, atol=1e-8)
```

## Common Usage Errors

### Issue: Wrong parameter units

**Symptoms:**
- Results seem off by orders of magnitude
- Unphysical values

**Solution:**
Ensure consistent units (nanoseconds):
```python
# Correct
repetition_period = 25.0  # ns
lifetime = 2.5            # ns
irf_sigma = 0.1           # ns

# Incorrect
repetition_period = 25e-9  # This is seconds, not ns!
```

### Issue: Incorrect IRF parameter interpretation

**Symptoms:**
- Gaussian IRF results don't make sense
- Unexpected behavior with IRF width

**Solution:**
Understand IRF parameters:
```python
# Gaussian IRF
sigma = 0.1  # Standard deviation in ns (FWHM ≈ 2.355 * sigma)

# Rectangular IRF
width = 0.2  # Full width in ns

# Dirac IRF
# No parameters - instantaneous response
```

## Getting Help

If you encounter an issue not covered here:

1. **Check the validation report:** `VALIDATION_REPORT.md`
2. **Review notebook examples:** All notebooks include detailed explanations
3. **Check parameter ranges:** Ensure physically reasonable values
4. **Enable debug output:**
   ```python
   import logging
   logging.basicConfig(level=logging.DEBUG)
   ```
5. **Create minimal reproducible example:**
   ```python
   # Simplest case that shows the problem
   import numpy as np
   from src.fisher_information import calculate_fisher_matrix
   
   result = calculate_fisher_matrix(
       tau=2.0, T=25.0, time_bins=64,
       irf_type='dirac', irf_params={}
   )
   print(result)
   ```

## Reporting Bugs

When reporting issues, please include:

1. Python version: `python --version`
2. Package versions: `pip list`
3. Operating system
4. Minimal code to reproduce
5. Full error message and traceback
6. Expected vs actual behavior

## Additional Resources

- **Validation Report:** See `VALIDATION_REPORT.md` for numerical accuracy validation
- **API Documentation:** See docstrings in source files
- **Examples:** Check `notebooks/` directory for comprehensive examples
- **Literature:** Kollner & Wolfrum (1995) for theoretical background
