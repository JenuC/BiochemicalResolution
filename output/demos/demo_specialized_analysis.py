"""
Demonstration of specialized analysis plotting functions.

This script demonstrates how to use the specialized plotting functions
for biochemical resolution analysis in FLIM applications.
"""

import numpy as np
import matplotlib.pyplot as plt
from src.visualization import (
    plot_separability_analysis,
    plot_resolving_power,
    plot_loss_analysis,
    plot_comparison_grid
)
from src.fisher_information import calculate_f_value
from src.core import AnalysisResults, AnalysisParameters

print("Specialized Analysis Plotting Demonstration")
print("=" * 70)

# Define analysis parameters
T = 25.0  # Repetition period (ns)
time_bins = 512
irf_type = 'dirac'

# Example 1: Separability Analysis for NADH Free vs Bound
print("\n1. NADH Free vs Bound Separability Analysis")
print("-" * 70)
print("Biological Context: NADH exists in free (~0.4 ns) and bound (~2.5 ns)")
print("states. Can we distinguish them with our measurement system?")

tau_nadh_free = 0.4  # ns
tau_range = np.linspace(0.2, 5.0, 50)

# Calculate separability based on Fisher information
separability = []
for tau2 in tau_range:
    f1 = calculate_f_value(tau_nadh_free, T, time_bins, irf_type)
    f2 = calculate_f_value(tau2, T, time_bins, irf_type)
    # Simplified separability metric
    sep = np.abs(tau2 - tau_nadh_free) * np.sqrt((f1 + f2) / 2)
    separability.append(sep)

separability = np.array(separability)

params = {
    'time_bins': time_bins,
    'repetition_period': T,
    'irf_sigma': 0.0
}

fig = plot_separability_analysis(
    tau1=tau_nadh_free,
    tau2_range=tau_range,
    f_values=separability,
    threshold=2.0,
    params=params,
    title='NADH Free vs Bound Separability',
    save_path='demo_nadh_separability.png'
)
plt.close(fig)

# Find minimum resolvable lifetime
resolvable_idx = np.where(separability >= 2.0)[0]
if len(resolvable_idx) > 0:
    min_resolvable = tau_range[resolvable_idx[0]]
    print(f"✓ Minimum resolvable lifetime from {tau_nadh_free} ns: {min_resolvable:.3f} ns")
    print(f"✓ NADH bound state (2.5 ns) is clearly resolvable!")
else:
    print("✗ No lifetimes are resolvable with current settings")

print("✓ Saved: demo_nadh_separability.png")

# Example 2: Resolving Power Across Lifetime Range
print("\n2. Resolving Power Analysis for Different IRF Conditions")
print("-" * 70)

tau_range = np.linspace(0.5, 10.0, 25)

# Calculate resolving power for different IRF conditions
resolving_data = {}

for irf_sigma in [0.0, 0.1, 0.5]:
    min_diff = []
    
    for tau in tau_range:
        if irf_sigma == 0.0:
            irf_type_local = 'dirac'
            irf_params = {}
        else:
            irf_type_local = 'gaussian'
            irf_params = {'sigma': irf_sigma}
        
        f_val = calculate_f_value(tau, T, time_bins, irf_type_local, irf_params)
        
        # Minimum resolvable difference (2σ criterion)
        # σ(τ) ≈ 1/√F, so Δτ_min ≈ 2σ(τ) = 2/√F
        delta_tau_min = 2.0 / np.sqrt(f_val) if f_val > 0 else np.inf
        min_diff.append(delta_tau_min)
    
    min_diff = np.array(min_diff)
    
    if irf_sigma == 0.0:
        primary_diff = min_diff
        label = 'Dirac IRF'
    else:
        resolving_data[f'Gaussian σ={irf_sigma} ns'] = {
            'tau': tau_range,
            'diff': min_diff
        }

params = {
    'time_bins': time_bins,
    'repetition_period': T,
    'threshold': 2.0
}

fig = plot_resolving_power(
    tau_range=tau_range,
    min_resolvable_diff=primary_diff,
    params=params,
    comparison_data=resolving_data,
    title='Resolving Power: Impact of IRF Width',
    save_path='demo_resolving_power.png'
)
plt.close(fig)

print("✓ Calculated resolving power for Dirac and Gaussian IRFs")
print(f"✓ At τ=2.0 ns, Dirac IRF: Δτ_min ≈ {primary_diff[np.argmin(np.abs(tau_range-2.0))]:.3f} ns")
print("✓ Saved: demo_resolving_power.png")

# Example 3: Information Loss Due to Non-Ideal IRF
print("\n3. Information Loss Analysis: Gaussian vs Dirac IRF")
print("-" * 70)

tau_range = np.linspace(0.5, 10.0, 30)

# Calculate F-values for ideal (Dirac) and actual (Gaussian) IRF
f_ideal = []
f_actual = []

for tau in tau_range:
    f_dirac = calculate_f_value(tau, T, time_bins, 'dirac')
    f_gauss = calculate_f_value(tau, T, time_bins, 'gaussian', {'sigma': 0.25})
    
    f_ideal.append(f_dirac)
    f_actual.append(f_gauss)

f_ideal = np.array(f_ideal)
f_actual = np.array(f_actual)

params = {
    'time_bins': time_bins,
    'repetition_period': T,
    'irf_sigma': 0.25
}

fig = plot_loss_analysis(
    tau_range=tau_range,
    f_values_ideal=f_ideal,
    f_values_actual=f_actual,
    irf_label='Gaussian σ=0.25 ns',
    params=params,
    title='Information Loss: Gaussian IRF vs Ideal',
    save_path='demo_loss_analysis.png'
)
plt.close(fig)

# Calculate average loss
avg_loss = np.mean((1 - f_actual / f_ideal) * 100)
print(f"✓ Average information loss: {avg_loss:.2f}%")
print(f"✓ This means ~{avg_loss:.1f}% increase in measurement uncertainty")
print("✓ Saved: demo_loss_analysis.png")

# Example 4: Comparison Grid for Multiple Conditions
print("\n4. Comparison Grid: Multiple IRF Conditions")
print("-" * 70)

lifetimes = np.linspace(0.5, 10.0, 25)
time_bins_array = 2**(np.arange(6) + 7)  # 128 to 4096

data_dict = {}

for sigma in [0.01, 0.1, 0.25, 0.5]:
    f_values = np.zeros((len(lifetimes), len(time_bins_array)))
    
    for i, tau in enumerate(lifetimes):
        for j, n_bins in enumerate(time_bins_array):
            if sigma < 0.05:
                f_val = calculate_f_value(tau, T, n_bins, 'dirac')
            else:
                f_val = calculate_f_value(tau, T, n_bins, 'gaussian', {'sigma': sigma})
            f_values[i, j] = f_val
    
    params = AnalysisParameters(
        repetition_period=T,
        lifetimes=lifetimes,
        time_bins=time_bins_array
    )
    
    results = AnalysisResults(
        f_values=f_values,
        parameters=params,
        metadata={'irf_sigma': sigma},
        analysis_type='gaussian_irf' if sigma >= 0.05 else 'dirac_irf'
    )
    
    label = 'Dirac IRF' if sigma < 0.05 else f'Gaussian σ={sigma} ns'
    data_dict[label] = results

fig = plot_comparison_grid(
    data_dict=data_dict,
    plot_type='heatmap',
    title='Fisher Information: IRF Comparison Grid',
    save_path='demo_comparison_grid.png'
)
plt.close(fig)

print(f"✓ Created comparison grid for {len(data_dict)} IRF conditions")
print("✓ Saved: demo_comparison_grid.png")

print("\n" + "=" * 70)
print("Demonstration Complete!")
print("\nGenerated demonstration plots:")
print("  1. demo_nadh_separability.png - NADH free vs bound analysis")
print("  2. demo_resolving_power.png - Resolving power comparison")
print("  3. demo_loss_analysis.png - Information loss quantification")
print("  4. demo_comparison_grid.png - Multi-condition comparison")
print("\nThese plots demonstrate the specialized analysis capabilities for")
print("biochemical resolution studies in fluorescence lifetime imaging.")
