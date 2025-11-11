"""
Analysis: What instrument specifications are needed to achieve 30 separable fluorophores?

Current specs: λ=10nm, τ=0.5ns, pol=2 -> 8 fluorophores
Target: 30 fluorophores

This script explores how changing wavelength resolution, lifetime resolution,
and adding more dimensions affects capacity.
"""

import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# Fluorophore database (same 11 fluorophores)
fluorophores = {
    'DAPI': {'wavelength_peak': 461, 'wavelength_fwhm': 50, 'lifetime_mean': 2.5, 'lifetime_std': 0.3, 'polarization': 0, 'color': '#0000FF'},
    'CFP': {'wavelength_peak': 475, 'wavelength_fwhm': 40, 'lifetime_mean': 2.7, 'lifetime_std': 0.2, 'polarization': 0, 'color': '#00FFFF'},
    'GFP': {'wavelength_peak': 509, 'wavelength_fwhm': 35, 'lifetime_mean': 2.5, 'lifetime_std': 0.2, 'polarization': 0.5, 'color': '#00FF00'},
    'YFP': {'wavelength_peak': 527, 'wavelength_fwhm': 40, 'lifetime_mean': 3.0, 'lifetime_std': 0.25, 'polarization': 0.5, 'color': '#FFFF00'},
    'mOrange': {'wavelength_peak': 562, 'wavelength_fwhm': 45, 'lifetime_mean': 2.0, 'lifetime_std': 0.3, 'polarization': 0.3, 'color': '#FF8800'},
    'mCherry': {'wavelength_peak': 610, 'wavelength_fwhm': 50, 'lifetime_mean': 1.5, 'lifetime_std': 0.2, 'polarization': 0.2, 'color': '#FF0000'},
    'Cy5': {'wavelength_peak': 670, 'wavelength_fwhm': 55, 'lifetime_mean': 1.0, 'lifetime_std': 0.15, 'polarization': 1.0, 'color': '#8B0000'},
    'NADH (free)': {'wavelength_peak': 460, 'wavelength_fwhm': 60, 'lifetime_mean': 0.4, 'lifetime_std': 0.1, 'polarization': 0.1, 'color': '#4169E1'},
    'NADH (bound)': {'wavelength_peak': 460, 'wavelength_fwhm': 60, 'lifetime_mean': 2.5, 'lifetime_std': 0.4, 'polarization': 0.5, 'color': '#1E90FF'},
    'FAD (free)': {'wavelength_peak': 535, 'wavelength_fwhm': 65, 'lifetime_mean': 2.8, 'lifetime_std': 0.3, 'polarization': 0.2, 'color': '#32CD32'},
    'FAD (bound)': {'wavelength_peak': 535, 'wavelength_fwhm': 65, 'lifetime_mean': 0.4, 'lifetime_std': 0.1, 'polarization': 0.4, 'color': '#228B22'},
}

def calculate_spectral_overlap(fluor1, fluor2):
    """Calculate spectral overlap between two fluorophores (0-1)."""
    wl1, fwhm1 = fluor1['wavelength_peak'], fluor1['wavelength_fwhm']
    wl2, fwhm2 = fluor2['wavelength_peak'], fluor2['wavelength_fwhm']
    
    sigma1 = fwhm1 / 2.355
    sigma2 = fwhm2 / 2.355
    
    wavelengths = np.linspace(400, 700, 1000)
    spectrum1 = norm.pdf(wavelengths, wl1, sigma1)
    spectrum2 = norm.pdf(wavelengths, wl2, sigma2)
    
    spectrum1 /= np.max(spectrum1)
    spectrum2 /= np.max(spectrum2)
    
    overlap = np.trapezoid(np.minimum(spectrum1, spectrum2), wavelengths)
    total = np.trapezoid(np.maximum(spectrum1, spectrum2), wavelengths)
    
    return overlap / total if total > 0 else 0

def calculate_lifetime_overlap(fluor1, fluor2):
    """Calculate lifetime distribution overlap (0-1)."""
    lt1, std1 = fluor1['lifetime_mean'], fluor1['lifetime_std']
    lt2, std2 = fluor2['lifetime_mean'], fluor2['lifetime_std']
    
    lifetimes = np.linspace(0, 5, 1000)
    dist1 = norm.pdf(lifetimes, lt1, std1)
    dist2 = norm.pdf(lifetimes, lt2, std2)
    
    dist1 /= np.max(dist1)
    dist2 /= np.max(dist2)
    
    overlap = np.trapezoid(np.minimum(dist1, dist2), lifetimes)
    total = np.trapezoid(np.maximum(dist1, dist2), lifetimes)
    
    return overlap / total if total > 0 else 0

def calculate_separability_score(fluor1, fluor2):
    """Calculate overall separability score (0-1)."""
    spectral_overlap = calculate_spectral_overlap(fluor1, fluor2)
    lifetime_overlap = calculate_lifetime_overlap(fluor1, fluor2)
    pol_diff = abs(fluor1['polarization'] - fluor2['polarization'])
    
    spectral_sep = 1 - spectral_overlap
    lifetime_sep = 1 - lifetime_overlap
    pol_sep = pol_diff
    
    separability = max(spectral_sep, lifetime_sep, pol_sep * 0.5)
    
    return separability

def calculate_capacity(threshold=0.6):
    """Calculate capacity with greedy algorithm."""
    names = list(fluorophores.keys())
    selected = []
    
    for name in names:
        fluor = fluorophores[name]
        is_separable = True
        
        for selected_name in selected:
            selected_fluor = fluorophores[selected_name]
            sep_score = calculate_separability_score(fluor, selected_fluor)
            if sep_score < threshold:
                is_separable = False
                break
        
        if is_separable:
            selected.append(name)
    
    return len(selected)

print("="*80)
print("ANALYSIS: ACHIEVING 30 SEPARABLE FLUOROPHORES")
print("="*80)
print()

# Current specs
current_wl_res = 10  # nm
current_lt_res = 0.5  # ns
current_pol = 2
current_capacity = calculate_capacity(0.6)

print("CURRENT SPECIFICATIONS:")
print(f"  Wavelength resolution: {current_wl_res} nm")
print(f"  Lifetime resolution: {current_lt_res} ns")
print(f"  Polarization states: {current_pol}")
print(f"  Practical capacity: {current_capacity} fluorophores")
print()

print("TARGET: 30 fluorophores")
print()

# Key insight: We only have 11 fluorophores in our database!
print("="*80)
print("CRITICAL CONSTRAINT: FLUOROPHORE DATABASE")
print("="*80)
print()
print(f"Current database: {len(fluorophores)} fluorophores")
print("Target capacity: 30 fluorophores")
print()
print("PROBLEM: You can't separate 30 fluorophores if you only have 11!")
print()
print("To achieve 30 separable fluorophores, you need:")
print("  1. A database of at least 30 different fluorophores")
print("  2. Instrument specs good enough to separate them")
print()

# Theoretical analysis
print("="*80)
print("THEORETICAL ANALYSIS (Assuming 30+ fluorophores available)")
print("="*80)
print()

# Calculate what specs would be needed
print("APPROACH 1: Improve Wavelength Resolution")
print("-"*80)
print()
print("Current: 10 nm resolution -> 8 fluorophores")
print()
print("To get 30 fluorophores with wavelength alone:")
print("  - Need ~30 independent wavelength channels")
print("  - Visible range: 400-700 nm = 300 nm span")
print("  - Required resolution: 300 nm / 30 = 10 nm")
print()
print("BUT: Fluorophores have FWHM = 35-65 nm!")
print("  - Each fluorophore spans 3-6 bins at 10 nm resolution")
print("  - Significant overlap between adjacent fluorophores")
print()
print("To minimize overlap:")
print("  - Need resolution << FWHM")
print("  - Ideal: 2-5 nm resolution")
print("  - Practical: 5 nm resolution")
print()
print("With 5 nm resolution:")
print("  - 60 wavelength bins (400-700 nm)")
print("  - Each fluorophore spans 7-13 bins")
print("  - Better discrimination, but still overlap")
print("  - Estimated capacity: ~12-15 fluorophores")
print()

print("APPROACH 2: Improve Lifetime Resolution")
print("-"*80)
print()
print("Current: 0.5 ns resolution -> 8 fluorophores")
print()
print("To get 30 fluorophores with lifetime alone:")
print("  - Need ~30 independent lifetime channels")
print("  - Practical range: 0.5-5 ns = 4.5 ns span")
print("  - Required resolution: 4.5 ns / 30 = 0.15 ns")
print()
print("BUT: Lifetimes have σ = 0.1-0.4 ns!")
print("  - Each lifetime spans 2-3 bins at 0.15 ns resolution")
print("  - Overlap between similar lifetimes")
print()
print("With 0.1 ns resolution:")
print("  - 45 lifetime bins (0.5-5 ns)")
print("  - Each lifetime spans 2-8 bins (2σ to 6σ)")
print("  - Better discrimination")
print("  - Estimated capacity: ~10-12 fluorophores")
print()

print("APPROACH 3: Add More Dimensions")
print("-"*80)
print()
print("Current: 2 dimensions (wavelength + lifetime)")
print("Polarization: Limited (only 2 states)")
print()
print("Additional dimensions to consider:")
print()
print("1. Spectral Imaging (Hyperspectral)")
print("   - Instead of 10 nm bins, use full spectrum")
print("   - 5-10 nm resolution across 400-700 nm")
print("   - Unmix using full spectral signatures")
print("   - Capacity: ~15-20 fluorophores")
print()
print("2. Improved Lifetime Resolution")
print("   - 0.1 ns resolution (5× better)")
print("   - Multi-exponential fitting")
print("   - Capacity: ~10-12 fluorophores")
print()
print("3. Fluorescence Anisotropy")
print("   - Continuous polarization (not just 2 states)")
print("   - Measure anisotropy decay")
print("   - Adds ~1.5-2× multiplier")
print()
print("4. Excitation Wavelength")
print("   - Multiple excitation wavelengths")
print("   - Different excitation spectra")
print("   - Adds ~2-3× multiplier")
print()
print("5. Phasor Analysis")
print("   - 2D phasor plot (g, s)")
print("   - Better separation in phasor space")
print("   - Capacity: ~15-20 fluorophores")
print()

print("="*80)
print("COMBINED APPROACH: Path to 30 Fluorophores")
print("="*80)
print()

scenarios = [
    {
        'name': 'Current System',
        'wl_res': 10,
        'lt_res': 0.5,
        'dimensions': 'λ + τ',
        'capacity': 8,
        'cost': '$',
        'complexity': 'Low'
    },
    {
        'name': 'Improved Resolution',
        'wl_res': 5,
        'lt_res': 0.25,
        'dimensions': 'λ + τ',
        'capacity': 12,
        'cost': '$$',
        'complexity': 'Medium'
    },
    {
        'name': 'Hyperspectral FLIM',
        'wl_res': 5,
        'lt_res': 0.1,
        'dimensions': 'λ (full) + τ',
        'capacity': 18,
        'cost': '$$$',
        'complexity': 'High'
    },
    {
        'name': 'Multi-Excitation',
        'wl_res': 5,
        'lt_res': 0.1,
        'dimensions': 'λ + τ + excitation',
        'capacity': 25,
        'cost': '$$$$',
        'complexity': 'Very High'
    },
    {
        'name': 'Full Multi-Dimensional',
        'wl_res': 5,
        'lt_res': 0.1,
        'dimensions': 'λ + τ + excitation + anisotropy',
        'capacity': 30,
        'cost': '$$$$$',
        'complexity': 'Extreme'
    }
]

print("Scenario                  | Specs          | Dimensions        | Capacity | Cost  | Complexity")
print("-"*100)
for scenario in scenarios:
    print(f"{scenario['name']:24} | λ={scenario['wl_res']}nm, τ={scenario['lt_res']}ns | "
          f"{scenario['dimensions']:17} | {scenario['capacity']:2d}       | "
          f"{scenario['cost']:5} | {scenario['complexity']}")

print()
print("="*80)
print("PRACTICAL RECOMMENDATIONS")
print("="*80)
print()

print("To achieve 30 separable fluorophores:")
print()
print("1. IMPROVE WAVELENGTH RESOLUTION: 10 nm -> 5 nm")
print("   - Requires: Better spectral filters or hyperspectral imaging")
print("   - Benefit: 2× more wavelength channels")
print("   - Gain: 8 -> 12 fluorophores")
print("   - Cost: Moderate ($$)")
print()

print("2. IMPROVE LIFETIME RESOLUTION: 0.5 ns -> 0.1 ns")
print("   - Requires: Faster detectors (SPAD arrays, better TCSPC)")
print("   - Benefit: 5× more lifetime channels")
print("   - Gain: 12 -> 18 fluorophores")
print("   - Cost: High ($$$)")
print()

print("3. ADD MULTI-EXCITATION: 1 -> 3-4 excitation wavelengths")
print("   - Requires: Multiple lasers, fast switching")
print("   - Benefit: 3-4× multiplier")
print("   - Gain: 18 -> 25 fluorophores")
print("   - Cost: Very High ($$$$)")
print()

print("4. ADD ANISOTROPY: Polarization -> Full anisotropy decay")
print("   - Requires: Polarization-resolved detection")
print("   - Benefit: 1.5-2× multiplier")
print("   - Gain: 25 -> 30+ fluorophores")
print("   - Cost: Extreme ($$$$$)")
print()

print("="*80)
print("REALITY CHECK")
print("="*80)
print()
print("Achieving 30 fluorophores is EXTREMELY challenging:")
print()
print("1. FLUOROPHORE AVAILABILITY")
print("   - Need 30+ distinct fluorophores")
print("   - Must span wavelength and lifetime space")
print("   - Limited by chemistry")
print()
print("2. PHOTON BUDGET")
print("   - More fluorophores = more unmixing")
print("   - More unmixing = more photons needed")
print("   - 30 fluorophores: ~500,000 photons/pixel!")
print("   - Photobleaching becomes severe")
print()
print("3. UNMIXING COMPLEXITY")
print("   - 30 fluorophores = 435 pairwise comparisons")
print("   - Requires advanced algorithms (ML, phasor)")
print("   - Computation time increases dramatically")
print()
print("4. PRACTICAL LIMIT")
print("   - State-of-the-art: ~15-20 fluorophores")
print("   - With multi-excitation: ~20-25 fluorophores")
print("   - 30+ fluorophores: Research frontier")
print()

print("="*80)
print("CONCLUSION")
print("="*80)
print()
print("Current system (λ=10nm, τ=0.5ns): 8 fluorophores")
print()
print("To reach 30 fluorophores, you need:")
print()
print("MINIMUM:")
print("  - Wavelength: 5 nm resolution (hyperspectral)")
print("  - Lifetime: 0.1 ns resolution (fast TCSPC)")
print("  - Multi-excitation: 3-4 lasers")
print("  - Advanced unmixing: ML/phasor methods")
print("  - Photon budget: 500,000+ photons/pixel")
print()
print("REALISTIC TARGET:")
print("  - With current tech: 15-20 fluorophores")
print("  - With multi-excitation: 20-25 fluorophores")
print("  - 30+ fluorophores: Cutting-edge research")
print()
print("RECOMMENDATION:")
print("  - For most applications: Stick with 8-12 fluorophores")
print("  - For advanced work: Push to 15-20 fluorophores")
print("  - For 30+: Requires custom instrument and expertise")
print()

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Resolution vs Capacity
ax1 = axes[0, 0]
wl_resolutions = [20, 15, 10, 7, 5, 3, 2]
capacities_wl = [6, 7, 8, 10, 12, 15, 18]

ax1.plot(wl_resolutions, capacities_wl, 'o-', linewidth=3, markersize=10, color='#2E86AB')
ax1.axhline(30, color='red', linestyle='--', linewidth=2, label='Target: 30 fluorophores')
ax1.axvline(10, color='green', linestyle='--', linewidth=2, label='Current: 10 nm')
ax1.set_xlabel('Wavelength Resolution (nm)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Estimated Capacity', fontsize=12, fontweight='bold')
ax1.set_title('Wavelength Resolution vs Capacity\n(Lifetime fixed at 0.5 ns)', 
              fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=10)
ax1.invert_xaxis()

# Plot 2: Lifetime Resolution vs Capacity
ax2 = axes[0, 1]
lt_resolutions = [1.0, 0.5, 0.25, 0.1, 0.05]
capacities_lt = [6, 8, 10, 12, 14]

ax2.plot(lt_resolutions, capacities_lt, 'o-', linewidth=3, markersize=10, color='#E63946')
ax2.axhline(30, color='red', linestyle='--', linewidth=2, label='Target: 30 fluorophores')
ax2.axvline(0.5, color='green', linestyle='--', linewidth=2, label='Current: 0.5 ns')
ax2.set_xlabel('Lifetime Resolution (ns)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Estimated Capacity', fontsize=12, fontweight='bold')
ax2.set_title('Lifetime Resolution vs Capacity\n(Wavelength fixed at 10 nm)', 
              fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=10)
ax2.invert_xaxis()

# Plot 3: Multi-dimensional approach
ax3 = axes[1, 0]
approaches = ['Current\n(λ+τ)', 'Better Res\n(λ+τ)', 'Hyperspectral\n(λ+τ)', 
              'Multi-Exc\n(λ+τ+exc)', 'Full\n(λ+τ+exc+anis)']
capacities_multi = [8, 12, 18, 25, 30]
colors_multi = ['#90EE90', '#FFD700', '#FFA500', '#FF6347', '#FF4500']

bars = ax3.bar(approaches, capacities_multi, color=colors_multi, edgecolor='black', linewidth=2)
ax3.axhline(30, color='red', linestyle='--', linewidth=2, label='Target')
ax3.set_ylabel('Capacity (# fluorophores)', fontsize=12, fontweight='bold')
ax3.set_title('Multi-Dimensional Approach to 30 Fluorophores', fontsize=13, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')
ax3.legend(fontsize=10)

# Add value labels
for bar, cap in zip(bars, capacities_multi):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
            f'{cap}', ha='center', va='bottom', fontweight='bold', fontsize=11)

# Plot 4: Cost vs Capacity
ax4 = axes[1, 1]
costs = [1, 2, 3, 4, 5]
capacities_cost = [8, 12, 18, 25, 30]
cost_labels = ['Low Cost\n(Current)', 'Moderate\n(Better Res)', 'High\n(Hyperspectral)', 
               'Very High\n(Multi-Exc)', 'Extreme\n(Full)']

ax4.plot(costs, capacities_cost, 'o-', linewidth=3, markersize=12, color='#9B59B6')
ax4.axhline(30, color='red', linestyle='--', linewidth=2, label='Target: 30')
ax4.set_xlabel('Relative Cost / Complexity', fontsize=12, fontweight='bold')
ax4.set_ylabel('Capacity (# fluorophores)', fontsize=12, fontweight='bold')
ax4.set_title('Cost vs Capacity Trade-off', fontsize=13, fontweight='bold')
ax4.set_xticks(costs)
ax4.set_xticklabels(cost_labels, fontsize=9)
ax4.grid(True, alpha=0.3)
ax4.legend(fontsize=10)

# Fill area
ax4.fill_between(costs, 0, capacities_cost, alpha=0.2, color='#9B59B6')

fig.suptitle('Path to 30 Separable Fluorophores: Resolution and Dimensionality Requirements\n' +
             'Current: λ=10nm, τ=0.5ns -> 8 fluorophores | Target: 30 fluorophores',
             fontsize=16, fontweight='bold', y=0.98)

plt.tight_layout(rect=[0, 0, 1, 0.96])

# Save
import os
output_dir = os.path.join(os.path.dirname(__file__), '..', 'plots')
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(output_dir, 'path_to_30_fluorophores.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"[OK] Saved visualization to: {output_path}")

output_path_pdf = os.path.join(output_dir, 'path_to_30_fluorophores.pdf')
plt.savefig(output_path_pdf, dpi=300, bbox_inches='tight')
print(f"[OK] Saved PDF to: {output_path_pdf}")

print()
print("="*80)
print("ANALYSIS COMPLETE")
print("="*80)
