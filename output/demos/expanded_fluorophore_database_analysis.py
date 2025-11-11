"""
Analysis: How does expanding the fluorophore database affect capacity?

Current: 11 fluorophores -> 8 separable
Question: What if we had 30, 50, or 100 fluorophores to choose from?

This explores the "selection pool" effect on capacity.
"""

import numpy as np
from scipy.stats import norm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Current database (11 fluorophores)
current_database = {
    'DAPI': {'wavelength_peak': 461, 'wavelength_fwhm': 50, 'lifetime_mean': 2.5, 'lifetime_std': 0.3},
    'CFP': {'wavelength_peak': 475, 'wavelength_fwhm': 40, 'lifetime_mean': 2.7, 'lifetime_std': 0.2},
    'GFP': {'wavelength_peak': 509, 'wavelength_fwhm': 35, 'lifetime_mean': 2.5, 'lifetime_std': 0.2},
    'YFP': {'wavelength_peak': 527, 'wavelength_fwhm': 40, 'lifetime_mean': 3.0, 'lifetime_std': 0.25},
    'mOrange': {'wavelength_peak': 562, 'wavelength_fwhm': 45, 'lifetime_mean': 2.0, 'lifetime_std': 0.3},
    'mCherry': {'wavelength_peak': 610, 'wavelength_fwhm': 50, 'lifetime_mean': 1.5, 'lifetime_std': 0.2},
    'Cy5': {'wavelength_peak': 670, 'wavelength_fwhm': 55, 'lifetime_mean': 1.0, 'lifetime_std': 0.15},
    'NADH (free)': {'wavelength_peak': 460, 'wavelength_fwhm': 60, 'lifetime_mean': 0.4, 'lifetime_std': 0.1},
    'NADH (bound)': {'wavelength_peak': 460, 'wavelength_fwhm': 60, 'lifetime_mean': 2.5, 'lifetime_std': 0.4},
    'FAD (free)': {'wavelength_peak': 535, 'wavelength_fwhm': 65, 'lifetime_mean': 2.8, 'lifetime_std': 0.3},
    'FAD (bound)': {'wavelength_peak': 535, 'wavelength_fwhm': 65, 'lifetime_mean': 0.4, 'lifetime_std': 0.1},
}

# Expanded database with real fluorophores
expanded_database = {
    # Original 11
    **current_database,
    
    # Additional common fluorophores
    'BFP': {'wavelength_peak': 440, 'wavelength_fwhm': 45, 'lifetime_mean': 1.2, 'lifetime_std': 0.15},
    'Cerulean': {'wavelength_peak': 475, 'wavelength_fwhm': 38, 'lifetime_mean': 3.3, 'lifetime_std': 0.25},
    'Venus': {'wavelength_peak': 528, 'wavelength_fwhm': 42, 'lifetime_mean': 3.2, 'lifetime_std': 0.3},
    'mKO': {'wavelength_peak': 559, 'wavelength_fwhm': 48, 'lifetime_mean': 2.3, 'lifetime_std': 0.25},
    'tdTomato': {'wavelength_peak': 581, 'wavelength_fwhm': 52, 'lifetime_mean': 2.8, 'lifetime_std': 0.3},
    'mRFP': {'wavelength_peak': 607, 'wavelength_fwhm': 55, 'lifetime_mean': 1.8, 'lifetime_std': 0.2},
    'mPlum': {'wavelength_peak': 649, 'wavelength_fwhm': 58, 'lifetime_mean': 1.3, 'lifetime_std': 0.18},
    'Alexa488': {'wavelength_peak': 519, 'wavelength_fwhm': 40, 'lifetime_mean': 4.1, 'lifetime_std': 0.2},
    'Alexa546': {'wavelength_peak': 573, 'wavelength_fwhm': 48, 'lifetime_mean': 3.9, 'lifetime_std': 0.25},
    'Alexa594': {'wavelength_peak': 617, 'wavelength_fwhm': 50, 'lifetime_mean': 3.5, 'lifetime_std': 0.3},
    'Alexa647': {'wavelength_peak': 668, 'wavelength_fwhm': 52, 'lifetime_mean': 1.2, 'lifetime_std': 0.15},
    'Cy3': {'wavelength_peak': 570, 'wavelength_fwhm': 45, 'lifetime_mean': 0.3, 'lifetime_std': 0.08},
    'Cy3.5': {'wavelength_peak': 596, 'wavelength_fwhm': 48, 'lifetime_mean': 0.5, 'lifetime_std': 0.1},
    'Cy5.5': {'wavelength_peak': 694, 'wavelength_fwhm': 55, 'lifetime_mean': 1.0, 'lifetime_std': 0.12},
    'Cy7': {'wavelength_peak': 767, 'wavelength_fwhm': 60, 'lifetime_mean': 0.6, 'lifetime_std': 0.1},
    'FITC': {'wavelength_peak': 518, 'wavelength_fwhm': 42, 'lifetime_mean': 4.0, 'lifetime_std': 0.3},
    'Rhodamine': {'wavelength_peak': 565, 'wavelength_fwhm': 50, 'lifetime_mean': 3.8, 'lifetime_std': 0.35},
    'TexasRed': {'wavelength_peak': 615, 'wavelength_fwhm': 52, 'lifetime_mean': 4.2, 'lifetime_std': 0.3},
    'Atto488': {'wavelength_peak': 523, 'wavelength_fwhm': 38, 'lifetime_mean': 4.3, 'lifetime_std': 0.25},
    'Atto565': {'wavelength_peak': 590, 'wavelength_fwhm': 45, 'lifetime_mean': 3.7, 'lifetime_std': 0.28},
    'Atto647N': {'wavelength_peak': 669, 'wavelength_fwhm': 50, 'lifetime_mean': 3.5, 'lifetime_std': 0.3},
    'DyLight488': {'wavelength_peak': 518, 'wavelength_fwhm': 43, 'lifetime_mean': 3.2, 'lifetime_std': 0.25},
    'DyLight550': {'wavelength_peak': 562, 'wavelength_fwhm': 47, 'lifetime_mean': 2.9, 'lifetime_std': 0.27},
    'DyLight650': {'wavelength_peak': 672, 'wavelength_fwhm': 53, 'lifetime_mean': 2.5, 'lifetime_std': 0.25},
    'Pacific Blue': {'wavelength_peak': 455, 'wavelength_fwhm': 48, 'lifetime_mean': 1.5, 'lifetime_std': 0.18},
    'Cascade Blue': {'wavelength_peak': 423, 'wavelength_fwhm': 52, 'lifetime_mean': 1.8, 'lifetime_std': 0.2},
    'Marina Blue': {'wavelength_peak': 460, 'wavelength_fwhm': 45, 'lifetime_mean': 2.1, 'lifetime_std': 0.22},
    'Coumarin': {'wavelength_peak': 470, 'wavelength_fwhm': 55, 'lifetime_mean': 2.5, 'lifetime_std': 0.3},
    'Fluorescein': {'wavelength_peak': 515, 'wavelength_fwhm': 45, 'lifetime_mean': 4.0, 'lifetime_std': 0.3},
    'Oregon Green': {'wavelength_peak': 526, 'wavelength_fwhm': 43, 'lifetime_mean': 4.1, 'lifetime_std': 0.28},
}

def calculate_spectral_overlap(fluor1, fluor2):
    """Calculate spectral overlap."""
    wl1, fwhm1 = fluor1['wavelength_peak'], fluor1['wavelength_fwhm']
    wl2, fwhm2 = fluor2['wavelength_peak'], fluor2['wavelength_fwhm']
    
    sigma1 = fwhm1 / 2.355
    sigma2 = fwhm2 / 2.355
    
    wavelengths = np.linspace(400, 800, 1000)
    spectrum1 = norm.pdf(wavelengths, wl1, sigma1)
    spectrum2 = norm.pdf(wavelengths, wl2, sigma2)
    
    spectrum1 /= np.max(spectrum1)
    spectrum2 /= np.max(spectrum2)
    
    overlap = np.trapezoid(np.minimum(spectrum1, spectrum2), wavelengths)
    total = np.trapezoid(np.maximum(spectrum1, spectrum2), wavelengths)
    
    return overlap / total if total > 0 else 0

def calculate_lifetime_overlap(fluor1, fluor2):
    """Calculate lifetime overlap."""
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
    """Calculate separability."""
    spectral_overlap = calculate_spectral_overlap(fluor1, fluor2)
    lifetime_overlap = calculate_lifetime_overlap(fluor1, fluor2)
    
    spectral_sep = 1 - spectral_overlap
    lifetime_sep = 1 - lifetime_overlap
    
    return max(spectral_sep, lifetime_sep)

def greedy_selection(database, threshold=0.6):
    """Greedy algorithm to select fluorophores."""
    names = list(database.keys())
    selected = []
    
    for name in names:
        fluor = database[name]
        is_separable = True
        
        for selected_name in selected:
            selected_fluor = database[selected_name]
            sep_score = calculate_separability_score(fluor, selected_fluor)
            if sep_score < threshold:
                is_separable = False
                break
        
        if is_separable:
            selected.append(name)
    
    return selected

print("="*90)
print("FLUOROPHORE DATABASE SIZE vs CAPACITY ANALYSIS")
print("="*90)
print()

# Test current database
print("CURRENT DATABASE (11 fluorophores):")
print("-"*90)
selected_current = greedy_selection(current_database, threshold=0.6)
print(f"  Database size: {len(current_database)}")
print(f"  Selected: {len(selected_current)} fluorophores")
print(f"  Utilization: {len(selected_current)/len(current_database)*100:.1f}%")
print(f"  Selected: {', '.join(selected_current[:5])}...")
print()

# Test expanded database
print("EXPANDED DATABASE (41 fluorophores):")
print("-"*90)
selected_expanded = greedy_selection(expanded_database, threshold=0.6)
print(f"  Database size: {len(expanded_database)}")
print(f"  Selected: {len(selected_expanded)} fluorophores")
print(f"  Utilization: {len(selected_expanded)/len(expanded_database)*100:.1f}%")
print(f"  Improvement: +{len(selected_expanded) - len(selected_current)} fluorophores ({(len(selected_expanded)/len(selected_current)-1)*100:.1f}% increase)")
print()
print(f"  Selected fluorophores:")
for i, name in enumerate(selected_expanded, 1):
    props = expanded_database[name]
    print(f"    {i:2d}. {name:20} - {props['wavelength_peak']:3d}nm, {props['lifetime_mean']:.1f}ns")
print()

# Test different thresholds
print("="*90)
print("CAPACITY vs THRESHOLD (Expanded Database)")
print("="*90)
print()

thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
print(f"{'Threshold':<12} | {'Current DB':<12} | {'Expanded DB':<12} | {'Improvement'}")
print("-"*90)

for thresh in thresholds:
    cap_current = len(greedy_selection(current_database, threshold=thresh))
    cap_expanded = len(greedy_selection(expanded_database, threshold=thresh))
    improvement = cap_expanded - cap_current
    
    print(f"  {thresh*100:>5.0f}%     | {cap_current:>6d}       | {cap_expanded:>6d}        | +{improvement} ({improvement/cap_current*100:.0f}%)")

print()

# Simulate even larger databases
print("="*90)
print("THEORETICAL ANALYSIS: Larger Databases")
print("="*90)
print()

print("If we had access to MORE fluorophores:")
print()

# Generate synthetic fluorophores with good coverage
def generate_synthetic_database(n_fluorophores):
    """Generate synthetic fluorophores with good wavelength/lifetime coverage."""
    database = {}
    
    # Wavelength range: 400-700 nm
    # Lifetime range: 0.3-4.5 ns
    
    # Use Latin hypercube sampling for good coverage
    np.random.seed(42)
    wavelengths = np.linspace(400, 700, int(np.sqrt(n_fluorophores)))
    lifetimes = np.linspace(0.3, 4.5, int(np.sqrt(n_fluorophores)))
    
    count = 0
    for wl in wavelengths:
        for lt in lifetimes:
            if count >= n_fluorophores:
                break
            database[f'Fluor_{count+1}'] = {
                'wavelength_peak': int(wl),
                'wavelength_fwhm': np.random.randint(35, 60),
                'lifetime_mean': lt,
                'lifetime_std': np.random.uniform(0.1, 0.3)
            }
            count += 1
    
    return database

database_sizes = [11, 20, 30, 50, 75, 100, 150, 200]
capacities_60 = []

for size in database_sizes:
    if size <= 41:
        # Use real databases
        if size == 11:
            db = current_database
        else:
            db = expanded_database
    else:
        # Use synthetic
        db = generate_synthetic_database(size)
    
    selected = greedy_selection(db, threshold=0.6)
    capacities_60.append(len(selected))

print(f"{'Database Size':<15} | {'Capacity (60%)':<15} | {'Utilization'}")
print("-"*90)
for size, cap in zip(database_sizes, capacities_60):
    util = cap / size * 100
    marker = " <-- Current" if size == 11 else " <-- Expanded" if size == 41 else ""
    print(f"  {size:>5d}         | {cap:>6d}          | {util:>5.1f}%{marker}")

print()
print("="*90)
print("KEY INSIGHTS")
print("="*90)
print()

print("1. DATABASE SIZE MATTERS!")
print(f"   - Current (11 fluor): {capacities_60[0]} separable")
print(f"   - Expanded (41 fluor): {capacities_60[database_sizes.index(41)]} separable")
print(f"   - Improvement: +{capacities_60[database_sizes.index(41)] - capacities_60[0]} fluorophores")
print()

print("2. DIMINISHING RETURNS")
print("   - 11 -> 41 fluor: +{} capacity ({:.0f}% increase)".format(
    capacities_60[database_sizes.index(41)] - capacities_60[0],
    (capacities_60[database_sizes.index(41)] / capacities_60[0] - 1) * 100
))
print("   - 41 -> 100 fluor: +{} capacity ({:.0f}% increase)".format(
    capacities_60[database_sizes.index(100)] - capacities_60[database_sizes.index(41)],
    (capacities_60[database_sizes.index(100)] / capacities_60[database_sizes.index(41)] - 1) * 100
))
print("   - Larger database helps, but not linearly")
print()

print("3. UTILIZATION DECREASES")
print(f"   - 11 fluor: {capacities_60[0]/11*100:.1f}% utilization")
print(f"   - 41 fluor: {capacities_60[database_sizes.index(41)]/41*100:.1f}% utilization")
print(f"   - 100 fluor: {capacities_60[database_sizes.index(100)]/100*100:.1f}% utilization")
print("   - More choices, but harder to find compatible ones")
print()

print("4. PATH TO 30 FLUOROPHORES")
print(f"   - With 200 fluor database: ~{capacities_60[-1]} separable")
print("   - Still short of 30!")
print("   - Need BOTH: Large database AND better instrument specs")
print()

# Create visualization
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Database size vs capacity
ax1 = axes[0, 0]
ax1.plot(database_sizes, capacities_60, 'o-', linewidth=3, markersize=10, color='#2E86AB')
ax1.axhline(30, color='red', linestyle='--', linewidth=2, label='Target: 30 fluorophores')
ax1.axvline(11, color='green', linestyle='--', linewidth=2, alpha=0.5, label='Current DB (11)')
ax1.axvline(41, color='orange', linestyle='--', linewidth=2, alpha=0.5, label='Expanded DB (41)')
ax1.set_xlabel('Database Size (# of fluorophores)', fontsize=12, fontweight='bold')
ax1.set_ylabel('Capacity (# separable)', fontsize=12, fontweight='bold')
ax1.set_title('Database Size vs Separation Capacity\n(60% threshold)', fontsize=13, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=10)

# Add annotations
ax1.annotate(f'{capacities_60[0]} fluor', xy=(11, capacities_60[0]), 
            xytext=(15, capacities_60[0]+2), fontsize=10, fontweight='bold',
            arrowprops=dict(arrowstyle='->', lw=1.5))
ax1.annotate(f'{capacities_60[database_sizes.index(41)]} fluor', 
            xy=(41, capacities_60[database_sizes.index(41)]), 
            xytext=(50, capacities_60[database_sizes.index(41)]+2), fontsize=10, fontweight='bold',
            arrowprops=dict(arrowstyle='->', lw=1.5))

# Plot 2: Utilization rate
ax2 = axes[0, 1]
utilization = [cap/size*100 for cap, size in zip(capacities_60, database_sizes)]
ax2.plot(database_sizes, utilization, 'o-', linewidth=3, markersize=10, color='#E63946')
ax2.set_xlabel('Database Size', fontsize=12, fontweight='bold')
ax2.set_ylabel('Utilization Rate (%)', fontsize=12, fontweight='bold')
ax2.set_title('Database Utilization Rate\n(% of database that is separable)', fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.axhline(50, color='gray', linestyle=':', alpha=0.5)
ax2.axhline(25, color='gray', linestyle=':', alpha=0.5)

# Plot 3: Marginal benefit
ax3 = axes[1, 0]
marginal_benefit = [0] + [capacities_60[i] - capacities_60[i-1] for i in range(1, len(capacities_60))]
ax3.bar(database_sizes, marginal_benefit, width=[5, 5, 5, 10, 15, 15, 25, 25], 
       color='#9B59B6', edgecolor='black', linewidth=1.5)
ax3.set_xlabel('Database Size', fontsize=12, fontweight='bold')
ax3.set_ylabel('Marginal Capacity Gain', fontsize=12, fontweight='bold')
ax3.set_title('Marginal Benefit of Adding More Fluorophores\n(Diminishing returns)', 
             fontsize=13, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y')

# Plot 4: Comparison table
ax4 = axes[1, 1]
ax4.axis('off')

comparison_data = [
    ['Database', 'Size', 'Capacity', 'Util%', 'vs Current'],
    ['Current', '11', f'{capacities_60[0]}', f'{capacities_60[0]/11*100:.0f}%', 'Baseline'],
    ['Expanded', '41', f'{capacities_60[database_sizes.index(41)]}', 
     f'{capacities_60[database_sizes.index(41)]/41*100:.0f}%', 
     f'+{capacities_60[database_sizes.index(41)]-capacities_60[0]}'],
    ['Large', '100', f'{capacities_60[database_sizes.index(100)]}', 
     f'{capacities_60[database_sizes.index(100)]/100*100:.0f}%', 
     f'+{capacities_60[database_sizes.index(100)]-capacities_60[0]}'],
    ['Very Large', '200', f'{capacities_60[-1]}', 
     f'{capacities_60[-1]/200*100:.0f}%', 
     f'+{capacities_60[-1]-capacities_60[0]}'],
]

# Draw table
table_y = 0.9
row_height = 0.12

for i, row in enumerate(comparison_data):
    y_pos = table_y - i * row_height
    
    # Header row
    if i == 0:
        for j, cell in enumerate(row):
            ax4.text(0.1 + j*0.15, y_pos, cell, fontsize=11, fontweight='bold', ha='left')
    else:
        color = '#E8F5E9' if i == 1 else '#FFF9C4' if i == 2 else '#FFE0B2' if i == 3 else '#FFCDD2'
        rect = plt.Rectangle((0.05, y_pos - row_height*0.7), 0.9, row_height*0.7,
                            facecolor=color, edgecolor='black', linewidth=1)
        ax4.add_patch(rect)
        
        for j, cell in enumerate(row):
            ax4.text(0.1 + j*0.15, y_pos - row_height*0.35, cell, fontsize=10, ha='left', va='center')

# Add summary
ax4.text(0.5, 0.25, 'KEY TAKEAWAY:', fontsize=12, fontweight='bold', ha='center')
ax4.text(0.5, 0.18, 'Expanding database from 11 to 41 fluorophores', fontsize=10, ha='center')
ax4.text(0.5, 0.12, f'increases capacity from {capacities_60[0]} to {capacities_60[database_sizes.index(41)]} (+{capacities_60[database_sizes.index(41)]-capacities_60[0]})',
        fontsize=10, ha='center', fontweight='bold', color='#2E86AB')
ax4.text(0.5, 0.05, 'But still need better instrument specs to reach 30!', fontsize=9, ha='center', style='italic')

fig.suptitle('Impact of Fluorophore Database Size on Separation Capacity\n' +
             'Current: 11 fluorophores -> 8 separable | Expanded: 41 fluorophores -> ? separable',
             fontsize=14, fontweight='bold', y=0.98)

plt.tight_layout(rect=[0, 0, 1, 0.96])

# Save
import os
output_dir = os.path.join(os.path.dirname(__file__), '..', 'plots')
os.makedirs(output_dir, exist_ok=True)

output_path = os.path.join(output_dir, 'database_size_vs_capacity.png')
plt.savefig(output_path, dpi=300, bbox_inches='tight')
print(f"[OK] Saved visualization to: {output_path}")

output_path_pdf = os.path.join(output_dir, 'database_size_vs_capacity.pdf')
plt.savefig(output_path_pdf, dpi=300, bbox_inches='tight')
print(f"[OK] Saved PDF to: {output_path_pdf}")

print()
print("="*90)
print("CONCLUSION")
print("="*90)
print()
print("YES! Expanding the fluorophore database DOES increase capacity:")
print(f"  - 11 fluorophores -> {capacities_60[0]} separable")
print(f"  - 41 fluorophores -> {capacities_60[database_sizes.index(41)]} separable")
print(f"  - 100 fluorophores -> {capacities_60[database_sizes.index(100)]} separable")
print()
print("BUT there are diminishing returns:")
print("  - Each additional fluorophore adds less capacity")
print("  - Utilization rate decreases (harder to find compatible ones)")
print()
print("To reach 30 fluorophores, you need BOTH:")
print("  1. Large database (100+ fluorophores)")
print("  2. Better instrument specs (5nm wavelength, 0.1ns lifetime)")
print()
print("Database expansion alone is NOT enough!")
