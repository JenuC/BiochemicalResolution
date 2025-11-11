# Specialized Analysis Plotting Functions

This document provides detailed documentation for the specialized analysis plotting functions added to the visualization module in Task 5.2.

## Overview

The specialized plotting functions provide advanced visualization capabilities for biochemical resolution analysis in fluorescence lifetime imaging microscopy (FLIM). These functions help researchers understand:

1. **Separability** - Can we distinguish between two different lifetimes?
2. **Resolving Power** - What is the minimum lifetime difference we can detect?
3. **Information Loss** - How much precision do we lose with non-ideal IRFs?
4. **Comparison** - How do different measurement conditions compare?

## Functions

### 1. `plot_separability_analysis()`

**Purpose**: Analyzes the ability to distinguish between two fluorescence lifetimes based on Fisher information.

**Biochemical Context**: In FLIM, different molecular species or environmental conditions produce different fluorescence lifetimes. For example:
- NADH free (~0.4 ns) vs bound (~2.5 ns)
- Different protein conformational states
- pH-sensitive fluorophores in different environments

**Parameters**:
- `tau1` (float): Reference fluorescence lifetime (ns)
- `tau2_range` (np.ndarray): Array of comparison lifetimes (ns)
- `f_values` (np.ndarray): Array of F-values or separability measures
- `threshold` (float): Threshold value for "resolvable" lifetimes (default: 2.0σ)
- `params` (dict, optional): Analysis parameters for annotation
- `show_threshold` (bool): Whether to show threshold line
- `title` (str, optional): Custom title
- `save_path` (str, optional): Path to save figure

**Returns**: matplotlib Figure object

**Example**:
```python
from src.visualization import plot_separability_analysis
import numpy as np

tau1 = 0.4  # NADH free state
tau2_range = np.linspace(0.2, 5.0, 50)
separability = calculate_separability(tau1, tau2_range)  # Your calculation

fig = plot_separability_analysis(
    tau1=tau1,
    tau2_range=tau2_range,
    f_values=separability,
    threshold=2.0,
    params={'time_bins': 512, 'repetition_period': 25.0},
    save_path='nadh_separability.png'
)
```

**Output**: Two-panel plot showing:
- Top: Separability vs lifetime ratio (τ₂/τ₁)
- Bottom: Separability vs absolute lifetime difference

Regions are shaded to indicate resolvable vs non-resolvable lifetimes.

---

### 2. `plot_resolving_power()`

**Purpose**: Visualizes the minimum resolvable lifetime difference across a range of reference lifetimes.

**Biochemical Context**: Understanding resolving power is critical for experimental design. For example:
- Can we distinguish FRET states with similar lifetimes?
- What is the minimum lifetime change we can detect for a biosensor?
- How does IRF width affect our ability to resolve similar species?

**Parameters**:
- `tau_range` (np.ndarray): Array of reference lifetime values (ns)
- `min_resolvable_diff` (np.ndarray): Minimum resolvable lifetime difference at each tau (ns)
- `params` (dict, optional): Measurement parameters
- `comparison_data` (dict, optional): Additional datasets for comparison
  - Format: `{'label': {'tau': array, 'diff': array}, ...}`
- `log_scale` (bool): Whether to use logarithmic scale
- `title` (str, optional): Custom title
- `save_path` (str, optional): Path to save figure

**Returns**: matplotlib Figure object

**Example**:
```python
from src.visualization import plot_resolving_power
import numpy as np

tau_range = np.linspace(0.5, 10.0, 30)
min_diff = calculate_min_resolvable_diff(tau_range)  # Your calculation

# Compare different IRF conditions
comparison = {
    'Gaussian σ=0.1': {'tau': tau_range, 'diff': min_diff_gauss_01},
    'Gaussian σ=0.5': {'tau': tau_range, 'diff': min_diff_gauss_05}
}

fig = plot_resolving_power(
    tau_range=tau_range,
    min_resolvable_diff=min_diff,
    comparison_data=comparison,
    params={'time_bins': 512, 'threshold': 2.0},
    save_path='resolving_power.png'
)
```

**Output**: Two-panel plot showing:
- Left: Absolute resolving power (Δτ in ns)
- Right: Relative resolving power (Δτ/τ as percentage)

---

### 3. `plot_loss_analysis()`

**Purpose**: Quantifies information loss due to non-ideal IRF characteristics.

**Biochemical Context**: Real measurement systems have finite IRF widths due to:
- Detector response time
- Electronic jitter
- Optical dispersion

Understanding the information loss helps:
- Optimize system design
- Set realistic expectations for measurement precision
- Justify equipment upgrades

**Parameters**:
- `tau_range` (np.ndarray): Array of lifetime values (ns)
- `f_values_ideal` (np.ndarray): F-values for ideal (Dirac) IRF
- `f_values_actual` (np.ndarray): F-values for actual IRF
- `irf_label` (str): Label for the actual IRF condition
- `params` (dict, optional): IRF parameters
- `show_percentage` (bool): Whether to show percentage loss
- `title` (str, optional): Custom title
- `save_path` (str, optional): Path to save figure

**Returns**: matplotlib Figure object

**Example**:
```python
from src.visualization import plot_loss_analysis
import numpy as np

tau_range = np.linspace(0.5, 10.0, 30)
f_ideal = calculate_f_dirac(tau_range)
f_actual = calculate_f_gaussian(tau_range, sigma=0.25)

fig = plot_loss_analysis(
    tau_range=tau_range,
    f_values_ideal=f_ideal,
    f_values_actual=f_actual,
    irf_label='Gaussian σ=0.25 ns',
    params={'time_bins': 512, 'irf_sigma': 0.25},
    save_path='loss_analysis.png'
)
```

**Output**: Four-panel comprehensive analysis:
1. Top: Direct F-value comparison (ideal vs actual)
2. Middle: Information loss percentage or ratio
3. Bottom-left: Precision degradation factor (σ_actual/σ_ideal)
4. Bottom-right: Summary statistics and interpretation

**Key Metrics**:
- **Information Loss**: `Loss = 1 - (F_actual / F_ideal)`
- **Precision Factor**: `Factor = √(F_ideal / F_actual) = σ_actual / σ_ideal`
- Factor > 1 means increased uncertainty
- Loss > 0 means reduced precision

---

### 4. `plot_comparison_grid()`

**Purpose**: Creates a grid of subplots comparing multiple analysis conditions side-by-side.

**Biochemical Context**: Useful for:
- Comparing different IRF conditions
- Evaluating parameter optimization
- Presenting comprehensive results in publications

**Parameters**:
- `data_dict` (dict): Dictionary with condition labels as keys and AnalysisResults as values
  - Example: `{'Dirac': results1, 'Gaussian σ=0.1': results2, ...}`
- `plot_type` (str): Type of plot ('heatmap', 'line', 'contour')
- `tau_index` (int, optional): Index of lifetime for line plots
- `log_scale` (bool): Whether to use logarithmic scale
- `title` (str, optional): Custom title
- `save_path` (str, optional): Path to save figure

**Returns**: matplotlib Figure object

**Example**:
```python
from src.visualization import plot_comparison_grid
from src.core import AnalysisResults, AnalysisParameters

# Create results for different conditions
data_dict = {
    'Dirac IRF': results_dirac,
    'Gaussian σ=0.1': results_gauss_01,
    'Gaussian σ=0.5': results_gauss_05
}

fig = plot_comparison_grid(
    data_dict=data_dict,
    plot_type='heatmap',
    title='Fisher Information Comparison',
    save_path='comparison_grid.png'
)
```

**Output**: Grid layout with up to 3 columns, showing heatmaps or line plots for each condition.

---

## Usage Examples

### Complete Workflow Example

```python
import numpy as np
from src.fisher_information import calculate_f_value
from src.visualization import (
    plot_separability_analysis,
    plot_resolving_power,
    plot_loss_analysis
)

# Define measurement parameters
T = 25.0  # Repetition period (ns)
time_bins = 512

# 1. Separability Analysis
tau_ref = 2.0
tau_range = np.linspace(0.5, 10.0, 50)

separability = []
for tau in tau_range:
    f1 = calculate_f_value(tau_ref, T, time_bins, 'dirac')
    f2 = calculate_f_value(tau, T, time_bins, 'dirac')
    sep = np.abs(tau - tau_ref) * np.sqrt((f1 + f2) / 2)
    separability.append(sep)

plot_separability_analysis(
    tau1=tau_ref,
    tau2_range=tau_range,
    f_values=np.array(separability),
    save_path='separability.png'
)

# 2. Resolving Power
min_diff = []
for tau in tau_range:
    f = calculate_f_value(tau, T, time_bins, 'dirac')
    min_diff.append(2.0 / np.sqrt(f))

plot_resolving_power(
    tau_range=tau_range,
    min_resolvable_diff=np.array(min_diff),
    save_path='resolving_power.png'
)

# 3. Information Loss
f_ideal = [calculate_f_value(tau, T, time_bins, 'dirac') for tau in tau_range]
f_actual = [calculate_f_value(tau, T, time_bins, 'gaussian', {'sigma': 0.25}) 
            for tau in tau_range]

plot_loss_analysis(
    tau_range=tau_range,
    f_values_ideal=np.array(f_ideal),
    f_values_actual=np.array(f_actual),
    irf_label='Gaussian σ=0.25 ns',
    save_path='loss_analysis.png'
)
```

---

## Interpretation Guidelines

### Separability Analysis
- **Threshold = 2σ**: Standard criterion for distinguishing two populations
- **Threshold = 3σ**: More conservative criterion for high confidence
- Lifetimes are considered resolvable when separability ≥ threshold

### Resolving Power
- **Absolute**: Minimum Δτ in nanoseconds
- **Relative**: Minimum Δτ/τ as percentage
- Lower values indicate better resolving power
- Typically improves with more time bins and narrower IRF

### Information Loss
- **0% loss**: Ideal case (Dirac IRF)
- **10-30% loss**: Typical for good TCSPC systems
- **>50% loss**: Significant degradation, consider system optimization
- Precision factor shows how much uncertainty increases

---

## Requirements Addressed

This implementation addresses the following requirements from the specification:

- **Requirement 4.1**: Consistent styling and formatting across all plots
- **Requirement 4.3**: Interactive plotting capabilities for parameter exploration

The specialized plotting functions provide:
1. ✓ Biochemical resolution analysis capabilities
2. ✓ Resolving power quantification
3. ✓ Information loss analysis
4. ✓ Multi-condition comparison
5. ✓ Publication-ready figures with consistent styling
6. ✓ Comprehensive documentation and examples

---

## Testing

All functions have been tested with:
- Unit tests in `test_specialized_plots.py`
- Demonstration examples in `demo_specialized_analysis.py`
- Real-world scenarios (NADH analysis, IRF comparison)

Run tests with:
```bash
python test_specialized_plots.py
python demo_specialized_analysis.py
```

---

## References

1. Kollner, M., & Wolfrum, J. (1992). "How many photons are necessary for fluorescence-lifetime measurements?" Chemical Physics Letters, 200(1-2), 199-204.

2. Lakowicz, J. R. (2006). "Principles of Fluorescence Spectroscopy" (3rd ed.). Springer.

3. Becker, W. (2005). "Advanced Time-Correlated Single Photon Counting Techniques". Springer.
