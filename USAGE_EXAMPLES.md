# Usage Examples and Parameter Guidelines

This document provides practical examples and guidelines for using the Instrument Response Function Analysis package.

## Table of Contents

1. [Basic Fisher Information Analysis](#basic-fisher-information-analysis)
2. [Gaussian IRF Analysis](#gaussian-irf-analysis)
3. [Monte Carlo Simulations](#monte-carlo-simulations)
4. [Custom Visualization](#custom-visualization)
5. [Parameter Guidelines](#parameter-guidelines)
6. [Advanced Usage](#advanced-usage)

## Basic Fisher Information Analysis

### Example 1: Single Lifetime Analysis

Calculate Fisher information for a single lifetime with Dirac IRF:

```python
import numpy as np
from src.fisher_information import calculate_fisher_matrix

# Parameters
tau = 2.5  # Fluorescence lifetime (ns)
T = 25.0   # Repetition period (ns)
time_bins = 256  # Number of time bins

# Calculate Fisher information
F = calculate_fisher_matrix(
    tau=tau,
    T=T,
    time_bins=time_bins,
    irf_type='dirac',
    irf_params={}
)

# Extract precision (Cramer-Rao lower bound)
precision = 1.0 / np.sqrt(F[0, 0])
print(f"Minimum achievable precision: {precision:.4f} ns")
```

### Example 2: Parameter Sweep

Analyze how precision varies with lifetime and time bins:

```python
from src.fisher_information import dirac_irf_analysis
from src.visualization import plot_fisher_analysis
import matplotlib.pyplot as plt

# Define parameter ranges
tau_range = np.arange(0.5, 10, 0.5)  # 0.5 to 10 ns
time_bins_range = 2**np.arange(5, 11)  # 32 to 1024 bins

# Run analysis
results = dirac_irf_analysis(
    tau_range=tau_range,
    time_bins_range=time_bins_range,
    T=25.0
)

# Visualize results
plot_fisher_analysis(results)
plt.savefig('fisher_analysis.png', dpi=300)
plt.show()
```

### Example 3: Kollner-Wolfrum Validation

Reproduce Figure 1 from Kollner and Wolfrum (1995):

```python
from src.fisher_information import kollner_wolfrum_reference
import matplotlib.pyplot as plt

# Parameters from the paper
T = 25.0  # ns
tau_range = np.linspace(0.2, 15, 100)
tn_values = [4, 8, 16, 32, 64, 128, 256, 512, 1024]

# Calculate for each time bin count
plt.figure(figsize=(10, 6))
for tn in tn_values:
    F_values = [kollner_wolfrum_reference(tau, T, tn) for tau in tau_range]
    plt.plot(tau_range, F_values, label=f'tn={tn}')

plt.xlabel('Lifetime τ (ns)')
plt.ylabel('Fisher Information F')
plt.legend()
plt.grid(True, alpha=0.3)
plt.title('Kollner-Wolfrum Figure 1 Reproduction')
plt.savefig('kollner_wolfrum_validation.png', dpi=300)
plt.show()
```

## Gaussian IRF Analysis

### Example 4: Effect of IRF Width

Compare precision for different Gaussian IRF widths:

```python
from src.fisher_information import gaussian_irf_analysis
from src.visualization import plot_irf_comparison

# Parameters
tau_range = np.arange(0.5, 10, 0.5)
time_bins = 256
sigma_values = [0.01, 0.1, 0.5, 1.0]  # IRF widths (ns)

# Analyze each IRF width
results = {}
for sigma in sigma_values:
    results[sigma] = gaussian_irf_analysis(
        tau_range=tau_range,
        time_bins_range=[time_bins],
        sigma_range=[sigma],
        T=25.0
    )

# Plot comparison
plot_irf_comparison(results)
plt.savefig('irf_width_comparison.png', dpi=300)
plt.show()
```

### Example 5: Optimal IRF Width

Find the optimal IRF width for a given lifetime:

```python
import numpy as np
from src.fisher_information import calculate_fisher_matrix

def find_optimal_irf_width(tau, T=25.0, time_bins=256):
    """Find IRF width that maximizes Fisher information."""
    sigma_range = np.logspace(-2, 0, 50)  # 0.01 to 1 ns
    F_values = []
    
    for sigma in sigma_range:
        F = calculate_fisher_matrix(
            tau=tau, T=T, time_bins=time_bins,
            irf_type='gaussian',
            irf_params={'sigma': sigma}
        )
        F_values.append(F[0, 0])
    
    optimal_idx = np.argmax(F_values)
    optimal_sigma = sigma_range[optimal_idx]
    
    return optimal_sigma, F_values

# Example: Find optimal IRF for tau=2.5 ns
tau = 2.5
optimal_sigma, F_values = find_optimal_irf_width(tau)
print(f"Optimal IRF width for τ={tau} ns: σ={optimal_sigma:.3f} ns")
```

## Monte Carlo Simulations

### Example 6: Basic Monte Carlo Validation

Validate Fisher information predictions with Monte Carlo:

```python
from src.monte_carlo import generate_measurement_data, fit_exponential_decay
import numpy as np

# True parameters
tau_true = 2.5  # ns
T = 25.0
time_bins = 256
num_photons = 50000
iterations = 1000

# Generate and fit data
tau_estimates = []
for i in range(iterations):
    # Generate synthetic data
    data = generate_measurement_data(
        tau=tau_true,
        T=T,
        time_bins=time_bins,
        num_photons=num_photons,
        irf_type='dirac'
    )
    
    # Fit exponential decay
    tau_fit, _ = fit_exponential_decay(data, initial_guess=[tau_true, num_photons])
    tau_estimates.append(tau_fit)

# Calculate empirical precision
tau_estimates = np.array(tau_estimates)
empirical_std = np.std(tau_estimates)
print(f"True lifetime: {tau_true} ns")
print(f"Mean estimate: {np.mean(tau_estimates):.3f} ns")
print(f"Empirical precision: {empirical_std:.4f} ns")

# Compare with Fisher information prediction
from src.fisher_information import calculate_fisher_matrix
F = calculate_fisher_matrix(tau_true, T, time_bins, 'dirac', {})
fisher_precision = 1.0 / np.sqrt(num_photons * F[0, 0])
print(f"Fisher prediction: {fisher_precision:.4f} ns")
print(f"Agreement: {empirical_std / fisher_precision:.2f}x")
```

### Example 7: IRF Parameter Uncertainty

Analyze the effect of unknown IRF parameters:

```python
from src.monte_carlo import monte_carlo_analysis

# Scenario 1: Known IRF width
results_known = monte_carlo_analysis(
    tau_range=[2.5],
    irf_params={'sigma': 0.1, 'known': True},
    iterations=1000
)

# Scenario 2: Unknown IRF width (must be fitted)
results_unknown = monte_carlo_analysis(
    tau_range=[2.5],
    irf_params={'sigma': 0.1, 'known': False},
    iterations=1000
)

print(f"Precision with known IRF: {results_known['precision'][0]:.4f} ns")
print(f"Precision with unknown IRF: {results_unknown['precision'][0]:.4f} ns")
print(f"Precision loss: {results_unknown['precision'][0] / results_known['precision'][0]:.2f}x")
```

## Custom Visualization

### Example 8: Publication-Ready Figures

Create customized plots with consistent styling:

```python
from src.visualization import plot_fisher_analysis
import matplotlib.pyplot as plt

# Set publication style
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    'font.size': 12,
    'font.family': 'serif',
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 11,
    'figure.figsize': (8, 6),
    'figure.dpi': 300
})

# Generate analysis
from src.fisher_information import dirac_irf_analysis
results = dirac_irf_analysis(
    tau_range=np.arange(0.5, 10, 0.5),
    time_bins_range=2**np.arange(5, 10)
)

# Create figure
fig, ax = plt.subplots()
plot_fisher_analysis(results, ax=ax)
ax.set_title('Fisher Information Analysis for Dirac IRF')
plt.tight_layout()
plt.savefig('publication_figure.pdf', bbox_inches='tight')
plt.savefig('publication_figure.png', dpi=300, bbox_inches='tight')
```

### Example 9: Interactive Parameter Exploration

Create interactive plots for parameter exploration:

```python
from ipywidgets import interact, FloatSlider, IntSlider
import matplotlib.pyplot as plt
from src.fisher_information import calculate_fisher_matrix

def plot_precision_vs_lifetime(sigma, time_bins):
    """Interactive plot of precision vs lifetime."""
    tau_range = np.linspace(0.5, 10, 100)
    precisions = []
    
    for tau in tau_range:
        F = calculate_fisher_matrix(
            tau=tau, T=25.0, time_bins=time_bins,
            irf_type='gaussian',
            irf_params={'sigma': sigma}
        )
        precision = 1.0 / np.sqrt(F[0, 0])
        precisions.append(precision)
    
    plt.figure(figsize=(10, 6))
    plt.plot(tau_range, precisions, linewidth=2)
    plt.xlabel('Lifetime τ (ns)')
    plt.ylabel('Precision (ns)')
    plt.title(f'Measurement Precision (σ={sigma} ns, bins={time_bins})')
    plt.grid(True, alpha=0.3)
    plt.show()

# Create interactive widget
interact(
    plot_precision_vs_lifetime,
    sigma=FloatSlider(min=0.01, max=2.0, step=0.01, value=0.1, description='IRF σ (ns)'),
    time_bins=IntSlider(min=32, max=1024, step=32, value=256, description='Time bins')
)
```

## Parameter Guidelines

### Typical FLIM Parameters

```python
# Standard FLIM setup
TYPICAL_PARAMS = {
    'repetition_period': 25.0,  # ns (40 MHz laser)
    'lifetimes': {
        'autofluorescence': 0.5,  # ns
        'NADH_free': 0.4,         # ns
        'NADH_bound': 2.5,        # ns
        'FAD_free': 2.8,          # ns
        'FAD_bound': 0.4,         # ns
        'GFP': 2.5,               # ns
        'mCherry': 1.5,           # ns
    },
    'time_bins': 256,            # Typical TCSPC
    'photon_counts': 50000,      # Per pixel
    'irf_width': 0.1,            # ns (typical for good system)
}
```

### Parameter Selection Guidelines

#### Repetition Period (T)
- **Rule of thumb:** T > 5 × max(lifetime)
- **Typical values:** 12.5 ns (80 MHz), 25 ns (40 MHz), 50 ns (20 MHz)
- **Trade-off:** Longer T → better separation, but lower count rate

```python
def recommend_repetition_period(max_lifetime):
    """Recommend repetition period for given lifetime."""
    recommended_T = 5 * max_lifetime
    standard_periods = [12.5, 25.0, 50.0, 100.0]
    return min([T for T in standard_periods if T >= recommended_T])

# Example
max_tau = 3.0  # ns
T = recommend_repetition_period(max_tau)
print(f"For max lifetime {max_tau} ns, use T = {T} ns")
```

#### Time Bins
- **Minimum:** 32 bins (for simple analysis)
- **Typical:** 256 bins (good balance)
- **Maximum:** 1024 bins (high precision, slower)
- **Rule of thumb:** bin_width ≈ 0.1 × shortest_lifetime

```python
def recommend_time_bins(T, min_lifetime):
    """Recommend number of time bins."""
    desired_bin_width = 0.1 * min_lifetime
    recommended_bins = int(T / desired_bin_width)
    # Round to nearest power of 2
    return 2 ** int(np.log2(recommended_bins))

# Example
bins = recommend_time_bins(T=25.0, min_lifetime=0.5)
print(f"Recommended time bins: {bins}")
```

#### IRF Width
- **Excellent system:** σ < 0.05 ns (FWHM < 120 ps)
- **Good system:** σ ≈ 0.1 ns (FWHM ≈ 240 ps)
- **Moderate system:** σ ≈ 0.5 ns (FWHM ≈ 1.2 ns)
- **Rule of thumb:** σ < 0.2 × shortest_lifetime

#### Photon Counts
- **Minimum:** 1,000 photons (rough estimate)
- **Good:** 10,000 photons (reliable fitting)
- **Excellent:** 50,000+ photons (high precision)
- **Diminishing returns:** > 100,000 photons

```python
def estimate_required_photons(desired_precision, tau, T, time_bins, irf_sigma):
    """Estimate photon count needed for desired precision."""
    from src.fisher_information import calculate_fisher_matrix
    
    F = calculate_fisher_matrix(
        tau=tau, T=T, time_bins=time_bins,
        irf_type='gaussian',
        irf_params={'sigma': irf_sigma}
    )
    
    # Cramer-Rao bound: precision = 1 / sqrt(N * F)
    required_N = (1.0 / (desired_precision ** 2)) / F[0, 0]
    return int(np.ceil(required_N))

# Example
N = estimate_required_photons(
    desired_precision=0.05,  # Want 50 ps precision
    tau=2.5, T=25.0, time_bins=256, irf_sigma=0.1
)
print(f"Required photon count: {N:,}")
```

## Advanced Usage

### Example 10: Multi-Exponential Analysis

Analyze systems with multiple lifetime components:

```python
from src.fisher_information import calculate_fisher_matrix_multiexp

# Two-component system (e.g., NADH free/bound)
tau1, tau2 = 0.4, 2.5  # ns
fraction1 = 0.3  # 30% free, 70% bound

# Calculate Fisher information for both components
F = calculate_fisher_matrix_multiexp(
    taus=[tau1, tau2],
    fractions=[fraction1, 1-fraction1],
    T=25.0,
    time_bins=256,
    irf_type='gaussian',
    irf_params={'sigma': 0.1}
)

# Precision for each component
precision1 = 1.0 / np.sqrt(F[0, 0])
precision2 = 1.0 / np.sqrt(F[1, 1])
print(f"Precision for τ1={tau1} ns: {precision1:.4f} ns")
print(f"Precision for τ2={tau2} ns: {precision2:.4f} ns")
```

### Example 11: Separability Analysis

Determine if two lifetimes can be distinguished:

```python
from src.visualization import plot_separability_analysis

def calculate_separability(tau1, tau2, T, time_bins, irf_sigma, num_photons):
    """Calculate separability metric for two lifetimes."""
    from src.fisher_information import calculate_fisher_matrix
    
    # Fisher information for each lifetime
    F1 = calculate_fisher_matrix(tau1, T, time_bins, 'gaussian', {'sigma': irf_sigma})
    F2 = calculate_fisher_matrix(tau2, T, time_bins, 'gaussian', {'sigma': irf_sigma})
    
    # Precision for each
    sigma1 = 1.0 / np.sqrt(num_photons * F1[0, 0])
    sigma2 = 1.0 / np.sqrt(num_photons * F2[0, 0])
    
    # Separability: difference relative to combined uncertainty
    separability = abs(tau2 - tau1) / np.sqrt(sigma1**2 + sigma2**2)
    
    return separability

# Example: NADH free vs bound
sep = calculate_separability(
    tau1=0.4, tau2=2.5,  # NADH free/bound
    T=25.0, time_bins=256,
    irf_sigma=0.1, num_photons=50000
)
print(f"Separability: {sep:.1f}σ")
if sep > 3:
    print("✓ Lifetimes are well separated")
else:
    print("⚠ Lifetimes may be difficult to distinguish")
```

### Example 12: Batch Processing

Process multiple datasets efficiently:

```python
from pathlib import Path
import pickle

def batch_analyze_datasets(data_dir, output_dir):
    """Analyze all datasets in a directory."""
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    for data_file in data_dir.glob('*.npy'):
        print(f"Processing {data_file.name}...")
        
        # Load data
        data = np.load(data_file)
        
        # Analyze
        results = analyze_dataset(data)
        
        # Save results
        output_file = output_dir / f"{data_file.stem}_results.pkl"
        with open(output_file, 'wb') as f:
            pickle.dump(results, f)
        
        print(f"  Saved to {output_file}")

def analyze_dataset(data):
    """Analyze a single dataset."""
    from src.monte_carlo import fit_exponential_decay
    
    # Fit data
    tau_fit, amplitude = fit_exponential_decay(data)
    
    # Calculate quality metrics
    residuals = calculate_residuals(data, tau_fit, amplitude)
    chi_squared = np.sum(residuals**2)
    
    return {
        'tau': tau_fit,
        'amplitude': amplitude,
        'chi_squared': chi_squared,
        'residuals': residuals
    }

# Usage
batch_analyze_datasets('data/measurements', 'data/results')
```

## Best Practices

1. **Always validate parameters:**
   ```python
   assert tau > 0, "Lifetime must be positive"
   assert T > 5 * tau, "Repetition period should be > 5× lifetime"
   assert irf_sigma < 0.5 * tau, "IRF width should be < 0.5× lifetime"
   ```

2. **Use appropriate tolerances:**
   ```python
   # For numerical comparisons
   np.allclose(a, b, rtol=1e-5, atol=1e-8)
   ```

3. **Save intermediate results:**
   ```python
   # For long computations
   np.save('intermediate_results.npy', results)
   ```

4. **Document your analysis:**
   ```python
   analysis_metadata = {
       'date': '2025-11-11',
       'parameters': params.__dict__,
       'software_version': '1.0',
       'notes': 'Analysis of NADH binding experiment'
   }
   ```

5. **Visualize before and after:**
   ```python
   # Always plot raw data and fits
   plt.figure(figsize=(12, 4))
   plt.subplot(131)
   plt.plot(data, label='Raw data')
   plt.subplot(132)
   plt.plot(fit, label='Fit')
   plt.subplot(133)
   plt.plot(residuals, label='Residuals')
   ```

## Additional Resources

- See `TROUBLESHOOTING.md` for common issues
- See `VALIDATION_REPORT.md` for numerical validation
- See notebooks for complete worked examples
- See source code docstrings for detailed API documentation
