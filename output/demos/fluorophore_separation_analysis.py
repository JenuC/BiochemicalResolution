"""
Analysis of fluorophore separation capacity given instrument specifications.

Instrument specs:
- Wavelength resolution: 10 nm
- Lifetime resolution: 0.5 ns  
- Polarization states: 2
"""

import numpy as np

# Instrument specifications
wavelength_resolution = 10  # nm
lifetime_resolution = 0.5   # ns
polarization_states = 2     # binary (parallel/perpendicular)

# Typical biological fluorophore ranges
wavelength_range = (400, 700)  # nm (visible spectrum for fluorescence)
lifetime_range = (0.5, 5.0)    # ns (typical for biological fluorophores)

# Calculate number of resolvable bins in each dimension
n_wavelength_bins = int((wavelength_range[1] - wavelength_range[0]) / wavelength_resolution)
n_lifetime_bins = int((lifetime_range[1] - lifetime_range[0]) / lifetime_resolution)
n_polarization_bins = polarization_states

print("=" * 70)
print("FLUOROPHORE SEPARATION CAPACITY ANALYSIS")
print("=" * 70)
print()
print("Instrument Specifications:")
print(f"  Wavelength resolution: {wavelength_resolution} nm")
print(f"  Lifetime resolution: {lifetime_resolution} ns")
print(f"  Polarization states: {polarization_states}")
print()

print("Biological Fluorophore Ranges:")
print(f"  Wavelength range: {wavelength_range[0]}-{wavelength_range[1]} nm")
print(f"  Lifetime range: {lifetime_range[0]}-{lifetime_range[1]} ns")
print()

print("Resolvable Bins per Dimension:")
print(f"  Wavelength bins: {n_wavelength_bins}")
print(f"  Lifetime bins: {n_lifetime_bins}")
print(f"  Polarization bins: {n_polarization_bins}")
print()

# Calculate total theoretical capacity
# This assumes perfect orthogonality between dimensions
total_capacity_theoretical = n_wavelength_bins * n_lifetime_bins * n_polarization_bins

print("Theoretical Capacity:")
print(f"  Maximum separable fluorophores: {total_capacity_theoretical}")
print()

# Practical considerations
print("=" * 70)
print("PRACTICAL CONSIDERATIONS")
print("=" * 70)
print()

# Factor 1: Overlap and crosstalk
overlap_factor = 0.5  # Assume 50% reduction due to spectral overlap
practical_capacity_1 = total_capacity_theoretical * overlap_factor

print(f"1. Spectral Overlap Reduction (×{overlap_factor}):")
print(f"   Reduced capacity: {int(practical_capacity_1)}")
print()

# Factor 2: Signal-to-noise requirements
# Need sufficient separation for reliable discrimination (typically 2-3σ)
snr_factor = 0.3  # Assume need 3σ separation, reducing usable space
practical_capacity_2 = practical_capacity_1 * snr_factor

print(f"2. SNR Requirements (×{snr_factor}):")
print(f"   Reduced capacity: {int(practical_capacity_2)}")
print()

# Factor 3: Biological constraints
# Not all combinations are biologically relevant or available
bio_factor = 0.5  # Many combinations don't exist in nature
practical_capacity_3 = practical_capacity_2 * bio_factor

print(f"3. Biological Availability (×{bio_factor}):")
print(f"   Reduced capacity: {int(practical_capacity_3)}")
print()

print("=" * 70)
print("SUMMARY")
print("=" * 70)
print()
print(f"Theoretical maximum: {total_capacity_theoretical} fluorophores")
print(f"Practical estimate: {int(practical_capacity_3)} fluorophores")
print()

# Breakdown by strategy
print("Separation Strategy Breakdown:")
print()

print("1. Single-Dimension Strategies:")
print(f"   - Wavelength only: {n_wavelength_bins} fluorophores")
print(f"   - Lifetime only: {n_lifetime_bins} fluorophores")
print(f"   - Polarization only: {n_polarization_bins} fluorophores")
print()

print("2. Two-Dimension Strategies:")
print(f"   - Wavelength + Lifetime: {n_wavelength_bins * n_lifetime_bins} fluorophores")
print(f"   - Wavelength + Polarization: {n_wavelength_bins * n_polarization_bins} fluorophores")
print(f"   - Lifetime + Polarization: {n_lifetime_bins * n_polarization_bins} fluorophores")
print()

print("3. Three-Dimension Strategy:")
print(f"   - Wavelength + Lifetime + Polarization: {total_capacity_theoretical} fluorophores")
print()

# Common fluorophore examples
print("=" * 70)
print("COMMON FLUOROPHORE EXAMPLES")
print("=" * 70)
print()

fluorophores = {
    'DAPI': {'wavelength': 461, 'lifetime': 2.5, 'polarization': 'low'},
    'GFP': {'wavelength': 509, 'lifetime': 2.5, 'polarization': 'medium'},
    'YFP': {'wavelength': 527, 'lifetime': 3.0, 'polarization': 'medium'},
    'mCherry': {'wavelength': 610, 'lifetime': 1.5, 'polarization': 'low'},
    'Cy3': {'wavelength': 570, 'lifetime': 0.3, 'polarization': 'high'},
    'Cy5': {'wavelength': 670, 'lifetime': 1.0, 'polarization': 'high'},
    'NADH (free)': {'wavelength': 460, 'lifetime': 0.4, 'polarization': 'low'},
    'NADH (bound)': {'wavelength': 460, 'lifetime': 2.5, 'polarization': 'medium'},
    'FAD (free)': {'wavelength': 535, 'lifetime': 2.8, 'polarization': 'low'},
    'FAD (bound)': {'wavelength': 535, 'lifetime': 0.4, 'polarization': 'medium'},
}

print("Fluorophore | Wavelength | Lifetime | Polarization | Separable?")
print("-" * 70)

for name, props in fluorophores.items():
    wl = props['wavelength']
    lt = props['lifetime']
    pol = props['polarization']
    
    # Check if within range
    wl_ok = wavelength_range[0] <= wl <= wavelength_range[1]
    lt_ok = lifetime_range[0] <= lt <= lifetime_range[1]
    
    separable = "✓" if (wl_ok and lt_ok) else "✗"
    
    print(f"{name:15} | {wl:4d} nm    | {lt:4.1f} ns | {pol:12} | {separable}")

print()
print("=" * 70)
print("RECOMMENDATIONS")
print("=" * 70)
print()
print("For optimal fluorophore separation:")
print()
print("1. Choose fluorophores with:")
print(f"   - Wavelength separation ≥ {wavelength_resolution * 2} nm (2× resolution)")
print(f"   - Lifetime separation ≥ {lifetime_resolution * 2} ns (2× resolution)")
print("   - Different polarization properties when possible")
print()
print("2. Realistic multiplexing capacity:")
print("   - Conservative: 3-5 fluorophores (well-separated)")
print("   - Moderate: 6-10 fluorophores (with careful selection)")
print("   - Aggressive: 10-15 fluorophores (requires optimization)")
print()
print("3. Example 5-fluorophore panel (well-separated):")
print("   - DAPI (461 nm, 2.5 ns)")
print("   - GFP (509 nm, 2.5 ns) - separated by wavelength")
print("   - YFP (527 nm, 3.0 ns) - separated by wavelength + lifetime")
print("   - mCherry (610 nm, 1.5 ns) - separated by wavelength + lifetime")
print("   - Cy5 (670 nm, 1.0 ns) - separated by wavelength + lifetime")
print()
print("4. For label-free imaging (NADH/FAD):")
print("   - NADH free/bound: Same wavelength, different lifetime (0.4 vs 2.5 ns)")
print("   - FAD free/bound: Same wavelength, different lifetime (2.8 vs 0.4 ns)")
print("   - NADH vs FAD: Different wavelength (460 vs 535 nm)")
print("   - Total: 4 metabolic states distinguishable")
print()
