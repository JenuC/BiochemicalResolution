"""
Visualization of fluorophore separation capacity in multi-dimensional space.

Creates plots showing:
1. 2D scatter plot (wavelength vs lifetime)
2. 3D visualization (wavelength vs lifetime vs polarization)
3. Resolution grid overlay
4. Common fluorophore examples
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, Ellipse
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.patches as mpatches

# Instrument specifications
wavelength_resolution = 10  # nm
lifetime_resolution = 0.5   # ns
polarization_states = 2

# Common fluorophores database
fluorophores = {
    'DAPI': {'wavelength': 461, 'lifetime': 2.5, 'polarization': 0, 'color': '#0000FF'},
    'CFP': {'wavelength': 475, 'lifetime': 2.7, 'polarization': 0, 'color': '#00FFFF'},
    'GFP': {'wavelength': 509, 'lifetime': 2.5, 'polarization': 0.5, 'color': '#00FF00'},
    'YFP': {'wavelength': 527, 'lifetime': 3.0, 'polarization': 0.5, 'color': '#FFFF00'},
    'mOrange': {'wavelength': 562, 'lifetime': 2.0, 'polarization': 0.3, 'color': '#FF8800'},
    'mCherry': {'wavelength': 610, 'lifetime': 1.5, 'polarization': 0.2, 'color': '#FF0000'},
    'Cy5': {'wavelength': 670, 'lifetime': 1.0, 'polarization': 1.0, 'color': '#8B0000'},
    'NADH (free)': {'wavelength': 460, 'lifetime': 0.4, 'polarization': 0.1, 'color': '#4169E1'},
    'NADH (bound)': {'wavelength': 460, 'lifetime': 2.5, 'polarization': 0.5, 'color': '#1E90FF'},
    'FAD (free)': {'wavelength': 535, 'lifetime': 2.8, 'polarization': 0.2, 'color': '#32CD32'},
    'FAD (bound)': {'wavelength': 535, 'lifetime': 0.4, 'polarization': 0.4, 'color': '#228B22'},
}

# Create figure with multiple subplots
fig = plt.figure(figsize=(18, 12))

# ============================================================================
# Plot 1: 2D Wavelength vs Lifetime with Resolution Grid
# ============================================================================
ax1 = plt.subplot(2, 3, 1)

# Draw resolution grid
wavelength_range = (400, 700)
lifetime_range = (0, 5.0)

# Grid lines
for wl in range(wavelength_range[0], wavelength_range[1] + 1, wavelength_resolution):
    ax1.axvline(wl, color='gray', alpha=0.2, linewidth=0.5)
for lt in np.arange(lifetime_range[0], lifetime_range[1] + 0.1, lifetime_resolution):
    ax1.axhline(lt, color='gray', alpha=0.2, linewidth=0.5)

# Plot fluorophores
for name, props in fluorophores.items():
    wl = props['wavelength']
    lt = props['lifetime']
    color = props['color']
    
    # Plot point
    ax1.scatter(wl, lt, s=200, c=color, edgecolors='black', linewidth=2, 
                zorder=10, alpha=0.8)
    
    # Add resolution uncertainty ellipse
    ellipse = Ellipse((wl, lt), wavelength_resolution, lifetime_resolution,
                     facecolor=color, alpha=0.2, edgecolor=color, linewidth=1)
    ax1.add_patch(ellipse)
    
    # Label
    ax1.annotate(name, (wl, lt), xytext=(5, 5), textcoords='offset points',
                fontsize=8, fontweight='bold')

ax1.set_xlabel('Emission Wavelength (nm)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Fluorescence Lifetime (ns)', fontsize=12, fontweight='bold')
ax1.set_title('2D Fluorophore Separation Space\n(Wavelength × Lifetime)', 
              fontsize=14, fontweight='bold')
ax1.set_xlim(wavelength_range)
ax1.set_ylim(lifetime_range)
ax1.grid(True, alpha=0.3)

# Add resolution annotation
ax1.text(0.02, 0.98, f'Resolution:\nΔλ = {wavelength_resolution} nm\nΔτ = {lifetime_resolution} ns',
         transform=ax1.transAxes, fontsize=10, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# ============================================================================
# Plot 2: 3D Visualization (Wavelength × Lifetime × Polarization)
# ============================================================================
ax2 = plt.subplot(2, 3, 2, projection='3d')

for name, props in fluorophores.items():
    wl = props['wavelength']
    lt = props['lifetime']
    pol = props['polarization']
    color = props['color']
    
    # Plot point
    ax2.scatter(wl, lt, pol, s=200, c=color, edgecolors='black', 
                linewidth=2, alpha=0.8, depthshade=False)
    
    # Add label
    ax2.text(wl, lt, pol, name, fontsize=7)

ax2.set_xlabel('Wavelength (nm)', fontsize=10, fontweight='bold')
ax2.set_ylabel('Lifetime (ns)', fontsize=10, fontweight='bold')
ax2.set_zlabel('Polarization', fontsize=10, fontweight='bold')
ax2.set_title('3D Fluorophore Separation Space\n(Wavelength × Lifetime × Polarization)', 
              fontsize=12, fontweight='bold')
ax2.set_xlim(wavelength_range)
ax2.set_ylim(lifetime_range)
ax2.set_zlim(0, 1)

# ============================================================================
# Plot 3: Resolution Bins Heatmap
# ============================================================================
ax3 = plt.subplot(2, 3, 3)

# Create occupancy grid
n_wl_bins = int((wavelength_range[1] - wavelength_range[0]) / wavelength_resolution)
n_lt_bins = int((lifetime_range[1] - lifetime_range[0]) / lifetime_resolution)
occupancy = np.zeros((n_lt_bins, n_wl_bins))

# Fill occupancy grid
for name, props in fluorophores.items():
    wl = props['wavelength']
    lt = props['lifetime']
    
    wl_bin = int((wl - wavelength_range[0]) / wavelength_resolution)
    lt_bin = int((lt - lifetime_range[0]) / lifetime_resolution)
    
    if 0 <= wl_bin < n_wl_bins and 0 <= lt_bin < n_lt_bins:
        occupancy[lt_bin, wl_bin] += 1

# Plot heatmap
im = ax3.imshow(occupancy, aspect='auto', cmap='YlOrRd', origin='lower',
                extent=[wavelength_range[0], wavelength_range[1], 
                       lifetime_range[0], lifetime_range[1]])
plt.colorbar(im, ax=ax3, label='Fluorophores per bin')

ax3.set_xlabel('Emission Wavelength (nm)', fontsize=12, fontweight='bold')
ax3.set_ylabel('Fluorescence Lifetime (ns)', fontsize=12, fontweight='bold')
ax3.set_title('Resolution Bin Occupancy\n(Darker = More Crowded)', 
              fontsize=14, fontweight='bold')

# Add grid
for wl in range(wavelength_range[0], wavelength_range[1] + 1, wavelength_resolution * 5):
    ax3.axvline(wl, color='white', alpha=0.3, linewidth=0.5)
for lt in np.arange(lifetime_range[0], lifetime_range[1] + 0.1, lifetime_resolution * 2):
    ax3.axhline(lt, color='white', alpha=0.3, linewidth=0.5)

# ============================================================================
# Plot 4: Separation Distance Matrix
# ============================================================================
ax4 = plt.subplot(2, 3, 4)

# Calculate separation distances
names = list(fluorophores.keys())
n_fluor = len(names)
separation_matrix = np.zeros((n_fluor, n_fluor))

for i, name1 in enumerate(names):
    for j, name2 in enumerate(names):
        if i != j:
            wl1, lt1 = fluorophores[name1]['wavelength'], fluorophores[name1]['lifetime']
            wl2, lt2 = fluorophores[name2]['wavelength'], fluorophores[name2]['lifetime']
            
            # Normalized separation distance
            wl_sep = abs(wl2 - wl1) / wavelength_resolution
            lt_sep = abs(lt2 - lt1) / lifetime_resolution
            
            # Euclidean distance in resolution units
            separation_matrix[i, j] = np.sqrt(wl_sep**2 + lt_sep**2)

# Plot heatmap
im = ax4.imshow(separation_matrix, cmap='RdYlGn', vmin=0, vmax=10)
plt.colorbar(im, ax=ax4, label='Separation (resolution units)')

ax4.set_xticks(range(n_fluor))
ax4.set_yticks(range(n_fluor))
ax4.set_xticklabels(names, rotation=45, ha='right', fontsize=8)
ax4.set_yticklabels(names, fontsize=8)
ax4.set_title('Pairwise Fluorophore Separation\n(Green = Well Separated)', 
              fontsize=12, fontweight='bold')

# Add threshold line annotation
ax4.text(0.02, 0.98, 'Threshold:\n>2 = Separable\n>3 = Well separated',
         transform=ax4.transAxes, fontsize=9, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))

# ============================================================================
# Plot 5: Capacity Analysis Bar Chart
# ============================================================================
ax5 = plt.subplot(2, 3, 5)

strategies = [
    'Wavelength\nOnly',
    'Lifetime\nOnly', 
    'Polarization\nOnly',
    'Wavelength +\nLifetime',
    'Wavelength +\nPolarization',
    'Lifetime +\nPolarization',
    'All Three\nDimensions'
]

n_wl_bins = 30
n_lt_bins = 9
n_pol_bins = 2

capacities = [
    n_wl_bins,
    n_lt_bins,
    n_pol_bins,
    n_wl_bins * n_lt_bins,
    n_wl_bins * n_pol_bins,
    n_lt_bins * n_pol_bins,
    n_wl_bins * n_lt_bins * n_pol_bins
]

colors_bar = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#FFA07A', '#98D8C8', '#F7DC6F', '#BB8FCE']
bars = ax5.barh(strategies, capacities, color=colors_bar, edgecolor='black', linewidth=1.5)

# Add value labels
for i, (bar, cap) in enumerate(zip(bars, capacities)):
    ax5.text(cap + 10, i, f'{cap}', va='center', fontweight='bold', fontsize=10)

ax5.set_xlabel('Theoretical Capacity (# of fluorophores)', fontsize=12, fontweight='bold')
ax5.set_title('Separation Capacity by Strategy', fontsize=14, fontweight='bold')
ax5.set_xscale('log')
ax5.grid(True, alpha=0.3, axis='x')

# Add practical estimate line
practical_estimate = 40
ax5.axvline(practical_estimate, color='red', linestyle='--', linewidth=2, 
            label=f'Practical Estimate ({practical_estimate})')
ax5.legend(loc='lower right')

# ============================================================================
# Plot 6: Recommended Fluorophore Panels
# ============================================================================
ax6 = plt.subplot(2, 3, 6)
ax6.axis('off')

# Define recommended panels
panels = {
    '3-Color Panel\n(Conservative)': ['DAPI', 'GFP', 'mCherry'],
    '5-Color Panel\n(Moderate)': ['DAPI', 'CFP', 'YFP', 'mOrange', 'Cy5'],
    'Label-Free\n(Metabolic)': ['NADH (free)', 'NADH (bound)', 'FAD (free)', 'FAD (bound)'],
}

y_pos = 0.95
for panel_name, fluor_list in panels.items():
    # Panel title
    ax6.text(0.05, y_pos, panel_name, fontsize=12, fontweight='bold',
             transform=ax6.transAxes)
    y_pos -= 0.08
    
    # Fluorophores in panel
    for fluor in fluor_list:
        if fluor in fluorophores:
            props = fluorophores[fluor]
            color = props['color']
            wl = props['wavelength']
            lt = props['lifetime']
            
            # Color box
            rect = Rectangle((0.05, y_pos - 0.02), 0.03, 0.04, 
                           facecolor=color, edgecolor='black', linewidth=1,
                           transform=ax6.transAxes)
            ax6.add_patch(rect)
            
            # Text
            ax6.text(0.10, y_pos, f'{fluor}: {wl}nm, {lt}ns',
                    fontsize=9, transform=ax6.transAxes, va='center')
            y_pos -= 0.05
    
    y_pos -= 0.05

# Add summary box
summary_text = f"""
INSTRUMENT SPECS:
• Wavelength: {wavelength_resolution} nm resolution
• Lifetime: {lifetime_resolution} ns resolution  
• Polarization: {polarization_states} states

CAPACITY:
• Theoretical: 540 fluorophores
• Practical: ~40 fluorophores
• Recommended: 3-10 fluorophores
"""

ax6.text(0.05, 0.25, summary_text, fontsize=10, family='monospace',
         transform=ax6.transAxes, verticalalignment='top',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

ax6.set_title('Recommended Fluorophore Panels', fontsize=14, fontweight='bold',
              pad=20)

# ============================================================================
# Overall title and layout
# ============================================================================
fig.suptitle('Fluorophore Separation Capacity Analysis\n' + 
             f'Resolution: λ={wavelength_resolution}nm, τ={lifetime_resolution}ns, Pol={polarization_states} states',
             fontsize=16, fontweight='bold', y=0.98)

plt.tight_layout(rect=[0, 0, 1, 0.96])

# Save figure
import os
output_dir = os.path.join(os.path.dirname(__file__), '..', 'plots')
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(output_dir, 'fluorophore_separation_visualization.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"✓ Saved visualization to: {output_path}")

output_path_pdf = os.path.join(output_dir, 'fluorophore_separation_visualization.pdf')
plt.savefig(output_path_pdf, dpi=300, bbox_inches='tight')
print(f"✓ Saved PDF to: {output_path_pdf}")

# plt.show()  # Comment out to avoid blocking

print("\n" + "="*70)
print("VISUALIZATION COMPLETE")
print("="*70)
print("\nThe plot shows:")
print("1. 2D scatter plot with resolution grid")
print("2. 3D visualization including polarization")
print("3. Resolution bin occupancy heatmap")
print("4. Pairwise separation distance matrix")
print("5. Capacity comparison by strategy")
print("6. Recommended fluorophore panels")
print("\nKey insights:")
print("• Wavelength + Lifetime provides the most separation power")
print("• Some fluorophores overlap (e.g., NADH/FAD at same wavelength)")
print("• Polarization adds 2× multiplier but limited discrimination")
print("• Practical capacity: 3-10 well-separated fluorophores")
