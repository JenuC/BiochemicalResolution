"""
Plot showing the relationship between separability confidence threshold
and the number of fluorophores that can be separated.

Shows the trade-off between:
- Low threshold (10-30%): More fluorophores, but higher crosstalk
- Medium threshold (40-60%): Balanced approach
- High threshold (70-90%): Fewer fluorophores, but excellent separation
"""

import matplotlib
matplotlib.use('Agg')
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
from matplotlib.patches import Rectangle
import matplotlib.patches as mpatches

# Fluorophore database
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

# Calculate capacity for different thresholds
thresholds = np.linspace(0.1, 0.9, 81)  # 10% to 90%
max_separable = []
selected_panels = {}

names = list(fluorophores.keys())

for threshold in thresholds:
    # Greedy algorithm
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
    
    max_separable.append(len(selected))
    
    # Store panels for key thresholds
    if abs(threshold - 0.3) < 0.01 or abs(threshold - 0.5) < 0.01 or abs(threshold - 0.7) < 0.01:
        selected_panels[threshold] = selected.copy()

max_separable = np.array(max_separable)

# Create comprehensive figure
fig = plt.figure(figsize=(18, 12))

# ============================================================================
# Plot 1: Main curve - Separability vs Capacity
# ============================================================================
ax1 = plt.subplot(2, 3, (1, 2))

# Plot main curve
ax1.plot(thresholds * 100, max_separable, linewidth=4, color='#2E86AB', 
         marker='o', markersize=4, label='Capacity curve')
ax1.fill_between(thresholds * 100, 0, max_separable, alpha=0.2, color='#2E86AB')

# Mark key thresholds
key_thresholds = [0.3, 0.5, 0.6, 0.7]
colors_key = ['#FF6B6B', '#FFA500', '#4ECDC4', '#45B7D1']
labels_key = ['Aggressive\n(High crosstalk)', 'Moderate\n(Balanced)', 
              'Recommended\n(Good separation)', 'Conservative\n(Excellent separation)']

for i, (thresh, color, label) in enumerate(zip(key_thresholds, colors_key, labels_key)):
    idx = np.argmin(np.abs(thresholds - thresh))
    capacity = max_separable[idx]
    
    # Vertical line
    ax1.axvline(thresh * 100, color=color, linestyle='--', linewidth=2, alpha=0.7)
    
    # Point
    ax1.scatter(thresh * 100, capacity, s=300, c=color, edgecolors='black', 
                linewidth=2, zorder=10, marker='o')
    
    # Label
    y_offset = [3, 2, 1, 0][i]
    ax1.annotate(f'{int(capacity)} fluorophores\n{label}', 
                xy=(thresh * 100, capacity),
                xytext=(thresh * 100 + 5, capacity + y_offset),
                fontsize=10, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.5', facecolor=color, alpha=0.7, edgecolor='black'),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))

ax1.set_xlabel('Separability Confidence Threshold (%)', fontsize=14, fontweight='bold')
ax1.set_ylabel('Number of Separable Fluorophores', fontsize=14, fontweight='bold')
ax1.set_title('Fluorophore Separation Capacity vs Confidence Threshold\n' + 
              'Trade-off: More fluorophores ↔ Better separation', 
              fontsize=16, fontweight='bold')
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.set_xlim(10, 90)
ax1.set_ylim(0, max(max_separable) + 2)

# Add shaded regions
ax1.axvspan(10, 40, alpha=0.1, color='red', label='High crosstalk zone')
ax1.axvspan(40, 70, alpha=0.1, color='yellow', label='Balanced zone')
ax1.axvspan(70, 90, alpha=0.1, color='green', label='High confidence zone')

ax1.legend(loc='upper right', fontsize=10)

# ============================================================================
# Plot 2: Derivative - Rate of change
# ============================================================================
ax2 = plt.subplot(2, 3, 3)

# Calculate derivative (rate of change)
d_capacity = np.diff(max_separable)
d_threshold = np.diff(thresholds * 100)
rate_of_change = d_capacity / d_threshold

# Plot
threshold_mid = (thresholds[:-1] + thresholds[1:]) / 2 * 100
ax2.plot(threshold_mid, rate_of_change, linewidth=3, color='#E63946', marker='o', markersize=3)
ax2.fill_between(threshold_mid, 0, rate_of_change, alpha=0.3, color='#E63946', 
                 where=(rate_of_change < 0))
ax2.axhline(0, color='black', linestyle='-', linewidth=1)

ax2.set_xlabel('Separability Threshold (%)', fontsize=12, fontweight='bold')
ax2.set_ylabel('Rate of Change\n(Δ fluorophores / Δ threshold %)', fontsize=11, fontweight='bold')
ax2.set_title('Sensitivity Analysis\n(Where does capacity drop fastest?)', 
              fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(10, 90)

# Mark steepest drop
steepest_idx = np.argmin(rate_of_change)
steepest_thresh = threshold_mid[steepest_idx]
ax2.scatter(steepest_thresh, rate_of_change[steepest_idx], s=200, c='red', 
           edgecolors='black', linewidth=2, zorder=10, marker='v')
ax2.annotate(f'Steepest drop\nat {steepest_thresh:.0f}%', 
            xy=(steepest_thresh, rate_of_change[steepest_idx]),
            xytext=(steepest_thresh + 10, rate_of_change[steepest_idx] - 0.05),
            fontsize=9, fontweight='bold',
            bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.8),
            arrowprops=dict(arrowstyle='->', lw=1.5))

# ============================================================================
# Plots 3-5: Fluorophore panels at different thresholds
# ============================================================================
panel_thresholds = [0.3, 0.5, 0.7]
panel_titles = ['Aggressive (30%)', 'Moderate (50%)', 'Conservative (70%)']
panel_colors = ['#FF6B6B', '#FFA500', '#45B7D1']

for plot_idx, (thresh, title, panel_color) in enumerate(zip(panel_thresholds, panel_titles, panel_colors)):
    ax = plt.subplot(2, 3, 4 + plot_idx)
    
    if thresh in selected_panels:
        selected = selected_panels[thresh]
        n_selected = len(selected)
        
        # Create 2D scatter plot
        wavelength_range = (400, 700)
        lifetime_range = (0, 5.0)
        
        # Draw light grid
        for wl in range(wavelength_range[0], wavelength_range[1] + 1, 50):
            ax.axvline(wl, color='gray', alpha=0.1, linewidth=0.5)
        for lt in np.arange(lifetime_range[0], lifetime_range[1] + 0.1, 1.0):
            ax.axhline(lt, color='gray', alpha=0.1, linewidth=0.5)
        
        # Plot selected fluorophores
        for name in selected:
            props = fluorophores[name]
            wl = props['wavelength_peak']
            lt = props['lifetime_mean']
            wl_width = props['wavelength_fwhm']
            lt_width = props['lifetime_std'] * 2.355
            color = props['color']
            
            # Ellipse
            from matplotlib.patches import Ellipse
            ellipse = Ellipse((wl, lt), wl_width, lt_width,
                            facecolor=color, alpha=0.3, edgecolor=color, linewidth=2)
            ax.add_patch(ellipse)
            
            # Point
            ax.scatter(wl, lt, s=150, c=color, edgecolors='black', linewidth=2, 
                      zorder=10, alpha=0.9)
            
            # Label
            ax.annotate(name, (wl, lt), xytext=(3, 3), textcoords='offset points',
                       fontsize=7, fontweight='bold')
        
        ax.set_xlabel('Wavelength (nm)', fontsize=10, fontweight='bold')
        ax.set_ylabel('Lifetime (ns)', fontsize=10, fontweight='bold')
        ax.set_title(f'{title}\n{n_selected} fluorophores', 
                    fontsize=12, fontweight='bold', color=panel_color)
        ax.set_xlim(wavelength_range)
        ax.set_ylim(lifetime_range)
        ax.grid(True, alpha=0.2)

# ============================================================================
# Plot 6: Summary table
# ============================================================================
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')

# Create summary table
summary_data = []
for thresh in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    idx = np.argmin(np.abs(thresholds - thresh))
    capacity = max_separable[idx]
    
    if thresh <= 0.3:
        quality = "High crosstalk"
        color = '#FFE5E5'
    elif thresh <= 0.6:
        quality = "Balanced"
        color = '#FFFFCC'
    else:
        quality = "Excellent sep."
        color = '#E5FFE5'
    
    summary_data.append([f'{int(thresh*100)}%', capacity, quality, color])

# Draw table
table_y = 0.95
row_height = 0.08

# Header
ax6.text(0.15, table_y, 'Threshold', fontsize=11, fontweight='bold', ha='center')
ax6.text(0.45, table_y, '# Fluorophores', fontsize=11, fontweight='bold', ha='center')
ax6.text(0.75, table_y, 'Quality', fontsize=11, fontweight='bold', ha='center')
table_y -= row_height

# Rows
for thresh_str, capacity, quality, color in summary_data:
    rect = Rectangle((0.05, table_y - row_height * 0.8), 0.9, row_height * 0.8,
                     facecolor=color, edgecolor='black', linewidth=1)
    ax6.add_patch(rect)
    
    ax6.text(0.15, table_y - row_height * 0.4, thresh_str, fontsize=10, ha='center', va='center')
    ax6.text(0.45, table_y - row_height * 0.4, str(capacity), fontsize=10, ha='center', va='center', fontweight='bold')
    ax6.text(0.75, table_y - row_height * 0.4, quality, fontsize=9, ha='center', va='center')
    
    table_y -= row_height

# Add recommendations
ax6.text(0.5, 0.12, 'Recommendations:', fontsize=11, fontweight='bold', ha='center')
ax6.text(0.5, 0.06, '• 30-40%: Use only with advanced unmixing', fontsize=8, ha='center')
ax6.text(0.5, 0.02, '• 50-60%: Good balance for most applications', fontsize=8, ha='center')
ax6.text(0.5, -0.02, '• 70-80%: Best for quantitative imaging', fontsize=8, ha='center')

ax6.set_title('Summary Table', fontsize=13, fontweight='bold', pad=20)

# ============================================================================
# Overall layout
# ============================================================================
fig.suptitle('Separability Confidence vs Fluorophore Capacity Analysis\n' + 
             'Instrument: λ=10nm, τ=0.5ns, Pol=2 states',
             fontsize=18, fontweight='bold', y=0.98)

plt.tight_layout(rect=[0, 0, 1, 0.96])

# Save
import os
output_dir = os.path.join(os.path.dirname(__file__), '..', 'plots')
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(output_dir, 'separability_vs_capacity.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"[OK] Saved to: {output_path}")

output_path_pdf = os.path.join(output_dir, 'separability_vs_capacity.pdf')
plt.savefig(output_path_pdf, dpi=300, bbox_inches='tight')
print(f"[OK] Saved PDF to: {output_path_pdf}")

print("\n" + "="*70)
print("SEPARABILITY VS CAPACITY ANALYSIS COMPLETE")
print("="*70)
print("\nKey findings:")
print(f"  • At 10% threshold: {max_separable[0]} fluorophores (high crosstalk)")
print(f"  • At 30% threshold: {max_separable[np.argmin(np.abs(thresholds - 0.3))]} fluorophores (aggressive)")
print(f"  • At 50% threshold: {max_separable[np.argmin(np.abs(thresholds - 0.5))]} fluorophores (balanced)")
print(f"  • At 60% threshold: {max_separable[np.argmin(np.abs(thresholds - 0.6))]} fluorophores (recommended)")
print(f"  • At 70% threshold: {max_separable[np.argmin(np.abs(thresholds - 0.7))]} fluorophores (conservative)")
print(f"  • At 90% threshold: {max_separable[-1]} fluorophores (excellent separation)")
print()
print("Trade-off:")
print("  Lower threshold → More fluorophores, but higher crosstalk")
print("  Higher threshold → Fewer fluorophores, but better separation")
print()
print("Recommended: 50-60% threshold for most applications")
