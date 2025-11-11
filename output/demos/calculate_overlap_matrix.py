"""
Calculate detailed overlap percentages for the 8 selected fluorophores.

Shows:
1. Spectral overlap (emission spectra)
2. Lifetime overlap (lifetime distributions)
3. Combined separability scores
4. Detailed pairwise analysis
"""

import numpy as np
from scipy.stats import norm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
# import seaborn as sns  # Not needed

# The 8 selected fluorophores (from 60% threshold analysis)
selected_fluorophores = {
    'DAPI': {
        'wavelength_peak': 461, 
        'wavelength_fwhm': 50,
        'lifetime_mean': 2.5, 
        'lifetime_std': 0.3,
        'polarization': 0,
        'color': '#0000FF'
    },
    'GFP': {
        'wavelength_peak': 509, 
        'wavelength_fwhm': 35,
        'lifetime_mean': 2.5, 
        'lifetime_std': 0.2,
        'polarization': 0.5,
        'color': '#00FF00'
    },
    'YFP': {
        'wavelength_peak': 527, 
        'wavelength_fwhm': 40,
        'lifetime_mean': 3.0, 
        'lifetime_std': 0.25,
        'polarization': 0.5,
        'color': '#FFFF00'
    },
    'mOrange': {
        'wavelength_peak': 562, 
        'wavelength_fwhm': 45,
        'lifetime_mean': 2.0, 
        'lifetime_std': 0.3,
        'polarization': 0.3,
        'color': '#FF8800'
    },
    'mCherry': {
        'wavelength_peak': 610, 
        'wavelength_fwhm': 50,
        'lifetime_mean': 1.5, 
        'lifetime_std': 0.2,
        'polarization': 0.2,
        'color': '#FF0000'
    },
    'Cy5': {
        'wavelength_peak': 670, 
        'wavelength_fwhm': 55,
        'lifetime_mean': 1.0, 
        'lifetime_std': 0.15,
        'polarization': 1.0,
        'color': '#8B0000'
    },
    'NADH (free)': {
        'wavelength_peak': 460, 
        'wavelength_fwhm': 60,
        'lifetime_mean': 0.4, 
        'lifetime_std': 0.1,
        'polarization': 0.1,
        'color': '#4169E1'
    },
    'FAD (bound)': {
        'wavelength_peak': 535, 
        'wavelength_fwhm': 65,
        'lifetime_mean': 0.4, 
        'lifetime_std': 0.1,
        'polarization': 0.4,
        'color': '#228B22'
    },
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
    
    return separability, spectral_overlap, lifetime_overlap

print("="*90)
print("OVERLAP ANALYSIS FOR 8 SELECTED FLUOROPHORES")
print("="*90)
print()
print("Selected panel (60% separability threshold):")
for i, name in enumerate(selected_fluorophores.keys(), 1):
    props = selected_fluorophores[name]
    print(f"  {i}. {name:15} - {props['wavelength_peak']}nm (FWHM={props['wavelength_fwhm']}nm), "
          f"{props['lifetime_mean']:.1f}ns (sigma={props['lifetime_std']:.2f}ns)")
print()

# Calculate all pairwise overlaps
names = list(selected_fluorophores.keys())
n = len(names)

spectral_overlap_matrix = np.zeros((n, n))
lifetime_overlap_matrix = np.zeros((n, n))
separability_matrix = np.zeros((n, n))

print("="*90)
print("PAIRWISE OVERLAP ANALYSIS")
print("="*90)
print()

overlap_data = []

for i, name1 in enumerate(names):
    for j, name2 in enumerate(names):
        if i < j:  # Only upper triangle
            fluor1 = selected_fluorophores[name1]
            fluor2 = selected_fluorophores[name2]
            
            sep, spec_overlap, life_overlap = calculate_separability_score(fluor1, fluor2)
            
            spectral_overlap_matrix[i, j] = spec_overlap
            spectral_overlap_matrix[j, i] = spec_overlap
            
            lifetime_overlap_matrix[i, j] = life_overlap
            lifetime_overlap_matrix[j, i] = life_overlap
            
            separability_matrix[i, j] = sep
            separability_matrix[j, i] = sep
            
            # Store for sorting
            overlap_data.append({
                'pair': f"{name1} - {name2}",
                'name1': name1,
                'name2': name2,
                'spectral_overlap': spec_overlap * 100,
                'lifetime_overlap': life_overlap * 100,
                'separability': sep * 100,
                'wl_diff': abs(fluor1['wavelength_peak'] - fluor2['wavelength_peak']),
                'lt_diff': abs(fluor1['lifetime_mean'] - fluor2['lifetime_mean'])
            })

# Sort by separability (lowest = most overlap)
overlap_data.sort(key=lambda x: x['separability'])

print("MOST OVERLAPPING PAIRS (Lowest Separability):")
print("-"*90)
print(f"{'Pair':<30} | {'Spectral':<10} | {'Lifetime':<10} | {'Separability':<12} | {'Status'}")
print("-"*90)

for data in overlap_data[:5]:  # Top 5 most overlapping
    status = "Good" if data['separability'] > 80 else "Moderate" if data['separability'] > 60 else "Poor"
    print(f"{data['pair']:<30} | {data['spectral_overlap']:>6.1f}%   | {data['lifetime_overlap']:>6.1f}%   | "
          f"{data['separability']:>8.1f}%    | {status}")

print()
print("LEAST OVERLAPPING PAIRS (Highest Separability):")
print("-"*90)
print(f"{'Pair':<30} | {'Spectral':<10} | {'Lifetime':<10} | {'Separability':<12} | {'Status'}")
print("-"*90)

for data in overlap_data[-5:]:  # Top 5 least overlapping
    status = "Excellent" if data['separability'] > 95 else "Very Good" if data['separability'] > 85 else "Good"
    print(f"{data['pair']:<30} | {data['spectral_overlap']:>6.1f}%   | {data['lifetime_overlap']:>6.1f}%   | "
          f"{data['separability']:>8.1f}%    | {status}")

print()
print("="*90)
print("COMPLETE PAIRWISE OVERLAP TABLE")
print("="*90)
print()

# Print complete table
print("SPECTRAL OVERLAP (%):")
print("-"*90)
print(f"{'':15}", end="")
for name in names:
    print(f"{name[:8]:>8}", end=" ")
print()
print("-"*90)

for i, name1 in enumerate(names):
    print(f"{name1[:15]:15}", end="")
    for j, name2 in enumerate(names):
        if i == j:
            print(f"{'---':>8}", end=" ")
        else:
            print(f"{spectral_overlap_matrix[i,j]*100:>7.1f}%", end=" ")
    print()

print()
print("LIFETIME OVERLAP (%):")
print("-"*90)
print(f"{'':15}", end="")
for name in names:
    print(f"{name[:8]:>8}", end=" ")
print()
print("-"*90)

for i, name1 in enumerate(names):
    print(f"{name1[:15]:15}", end="")
    for j, name2 in enumerate(names):
        if i == j:
            print(f"{'---':>8}", end=" ")
        else:
            print(f"{lifetime_overlap_matrix[i,j]*100:>7.1f}%", end=" ")
    print()

print()
print("SEPARABILITY SCORE (%):")
print("-"*90)
print(f"{'':15}", end="")
for name in names:
    print(f"{name[:8]:>8}", end=" ")
print()
print("-"*90)

for i, name1 in enumerate(names):
    print(f"{name1[:15]:15}", end="")
    for j, name2 in enumerate(names):
        if i == j:
            print(f"{'---':>8}", end=" ")
        else:
            print(f"{separability_matrix[i,j]*100:>7.1f}%", end=" ")
    print()

print()
print("="*90)
print("SUMMARY STATISTICS")
print("="*90)
print()

# Calculate statistics (excluding diagonal)
spectral_overlaps = spectral_overlap_matrix[np.triu_indices(n, k=1)] * 100
lifetime_overlaps = lifetime_overlap_matrix[np.triu_indices(n, k=1)] * 100
separabilities = separability_matrix[np.triu_indices(n, k=1)] * 100

print("SPECTRAL OVERLAP:")
print(f"  Mean: {np.mean(spectral_overlaps):.1f}%")
print(f"  Median: {np.median(spectral_overlaps):.1f}%")
print(f"  Min: {np.min(spectral_overlaps):.1f}%")
print(f"  Max: {np.max(spectral_overlaps):.1f}%")
print(f"  Std Dev: {np.std(spectral_overlaps):.1f}%")
print()

print("LIFETIME OVERLAP:")
print(f"  Mean: {np.mean(lifetime_overlaps):.1f}%")
print(f"  Median: {np.median(lifetime_overlaps):.1f}%")
print(f"  Min: {np.min(lifetime_overlaps):.1f}%")
print(f"  Max: {np.max(lifetime_overlaps):.1f}%")
print(f"  Std Dev: {np.std(lifetime_overlaps):.1f}%")
print()

print("SEPARABILITY:")
print(f"  Mean: {np.mean(separabilities):.1f}%")
print(f"  Median: {np.median(separabilities):.1f}%")
print(f"  Min: {np.min(separabilities):.1f}% (worst pair)")
print(f"  Max: {np.max(separabilities):.1f}% (best pair)")
print(f"  Std Dev: {np.std(separabilities):.1f}%")
print()

# Count pairs by separability
excellent = np.sum(separabilities > 95)
very_good = np.sum((separabilities > 85) & (separabilities <= 95))
good = np.sum((separabilities > 70) & (separabilities <= 85))
moderate = np.sum(separabilities <= 70)

print("SEPARABILITY DISTRIBUTION:")
print(f"  Excellent (>95%): {excellent} pairs ({excellent/len(separabilities)*100:.1f}%)")
print(f"  Very Good (85-95%): {very_good} pairs ({very_good/len(separabilities)*100:.1f}%)")
print(f"  Good (70-85%): {good} pairs ({good/len(separabilities)*100:.1f}%)")
print(f"  Moderate (<70%): {moderate} pairs ({moderate/len(separabilities)*100:.1f}%)")
print()

print("="*90)
print("KEY INSIGHTS")
print("="*90)
print()

# Find most problematic pair
worst_idx = np.argmin(separabilities)
worst_pair = overlap_data[worst_idx]
print(f"MOST OVERLAPPING PAIR: {worst_pair['pair']}")
print(f"  Spectral overlap: {worst_pair['spectral_overlap']:.1f}%")
print(f"  Lifetime overlap: {worst_pair['lifetime_overlap']:.1f}%")
print(f"  Separability: {worst_pair['separability']:.1f}%")
print(f"  Wavelength difference: {worst_pair['wl_diff']} nm")
print(f"  Lifetime difference: {worst_pair['lt_diff']:.1f} ns")
print()

# Find best pair
best_idx = np.argmax(separabilities)
best_pair = overlap_data[best_idx]
print(f"LEAST OVERLAPPING PAIR: {best_pair['pair']}")
print(f"  Spectral overlap: {best_pair['spectral_overlap']:.1f}%")
print(f"  Lifetime overlap: {best_pair['lifetime_overlap']:.1f}%")
print(f"  Separability: {best_pair['separability']:.1f}%")
print(f"  Wavelength difference: {best_pair['wl_diff']} nm")
print(f"  Lifetime difference: {best_pair['lt_diff']:.1f} ns")
print()

print("OVERALL ASSESSMENT:")
print(f"  Average pairwise separability: {np.mean(separabilities):.1f}%")
print(f"  Minimum pairwise separability: {np.min(separabilities):.1f}%")
print(f"  All pairs above 60% threshold: {'YES' if np.min(separabilities) > 60 else 'NO'}")
print(f"  All pairs above 85% threshold: {'YES' if np.min(separabilities) > 85 else 'NO'}")
print()

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 14))

# Plot 1: Spectral overlap heatmap
ax1 = axes[0, 0]
im1 = ax1.imshow(spectral_overlap_matrix * 100, cmap='YlOrRd', vmin=0, vmax=100)
ax1.set_xticks(range(n))
ax1.set_yticks(range(n))
ax1.set_xticklabels([name[:10] for name in names], rotation=45, ha='right', fontsize=9)
ax1.set_yticklabels([name[:10] for name in names], fontsize=9)
ax1.set_title('Spectral Overlap (%)\n(Lower = Better Separation)', fontsize=12, fontweight='bold')
plt.colorbar(im1, ax=ax1, label='Overlap %')

# Add text annotations
for i in range(n):
    for j in range(n):
        if i != j:
            text = ax1.text(j, i, f'{spectral_overlap_matrix[i, j]*100:.0f}',
                          ha="center", va="center", color="black", fontsize=8)

# Plot 2: Lifetime overlap heatmap
ax2 = axes[0, 1]
im2 = ax2.imshow(lifetime_overlap_matrix * 100, cmap='YlOrRd', vmin=0, vmax=100)
ax2.set_xticks(range(n))
ax2.set_yticks(range(n))
ax2.set_xticklabels([name[:10] for name in names], rotation=45, ha='right', fontsize=9)
ax2.set_yticklabels([name[:10] for name in names], fontsize=9)
ax2.set_title('Lifetime Overlap (%)\n(Lower = Better Separation)', fontsize=12, fontweight='bold')
plt.colorbar(im2, ax=ax2, label='Overlap %')

# Add text annotations
for i in range(n):
    for j in range(n):
        if i != j:
            text = ax2.text(j, i, f'{lifetime_overlap_matrix[i, j]*100:.0f}',
                          ha="center", va="center", color="black", fontsize=8)

# Plot 3: Separability heatmap
ax3 = axes[1, 0]
im3 = ax3.imshow(separability_matrix * 100, cmap='RdYlGn', vmin=60, vmax=100)
ax3.set_xticks(range(n))
ax3.set_yticks(range(n))
ax3.set_xticklabels([name[:10] for name in names], rotation=45, ha='right', fontsize=9)
ax3.set_yticklabels([name[:10] for name in names], fontsize=9)
ax3.set_title('Separability Score (%)\n(Higher = Better, >60% = Good)', fontsize=12, fontweight='bold')
plt.colorbar(im3, ax=ax3, label='Separability %')

# Add text annotations
for i in range(n):
    for j in range(n):
        if i != j:
            text = ax3.text(j, i, f'{separability_matrix[i, j]*100:.0f}',
                          ha="center", va="center", color="black", fontsize=8, fontweight='bold')

# Plot 4: Distribution histograms
ax4 = axes[1, 1]
ax4.hist(spectral_overlaps, bins=15, alpha=0.5, label='Spectral Overlap', color='red', edgecolor='black')
ax4.hist(lifetime_overlaps, bins=15, alpha=0.5, label='Lifetime Overlap', color='blue', edgecolor='black')
ax4.axvline(np.mean(spectral_overlaps), color='red', linestyle='--', linewidth=2, label=f'Mean Spectral: {np.mean(spectral_overlaps):.1f}%')
ax4.axvline(np.mean(lifetime_overlaps), color='blue', linestyle='--', linewidth=2, label=f'Mean Lifetime: {np.mean(lifetime_overlaps):.1f}%')
ax4.set_xlabel('Overlap (%)', fontsize=11, fontweight='bold')
ax4.set_ylabel('Number of Pairs', fontsize=11, fontweight='bold')
ax4.set_title('Overlap Distribution\n(28 pairwise comparisons)', fontsize=12, fontweight='bold')
ax4.legend(fontsize=9)
ax4.grid(True, alpha=0.3)

fig.suptitle('Overlap Analysis for 8 Selected Fluorophores\n' +
             f'Mean Separability: {np.mean(separabilities):.1f}% | Min: {np.min(separabilities):.1f}% | All pairs >60%: YES',
             fontsize=14, fontweight='bold', y=0.98)

plt.tight_layout(rect=[0, 0, 1, 0.96])

# Save
import os
output_dir = os.path.join(os.path.dirname(__file__), '..', 'plots')
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(output_dir, 'overlap_analysis_8_fluorophores.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"[OK] Saved visualization to: {output_path}")

output_path_pdf = os.path.join(output_dir, 'overlap_analysis_8_fluorophores.pdf')
plt.savefig(output_path_pdf, dpi=300, bbox_inches='tight')
print(f"[OK] Saved PDF to: {output_path_pdf}")

print()
print("="*90)
print("ANALYSIS COMPLETE")
print("="*90)
