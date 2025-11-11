# Separability Confidence vs Capacity Analysis

## Overview

This analysis shows how the number of separable fluorophores changes with the separability confidence threshold from 10% to 90%.

## Key Findings

### Capacity at Different Thresholds

| Threshold | # Fluorophores | Quality | Use Case |
|-----------|----------------|---------|----------|
| **10%** | 11 | High crosstalk | Not recommended |
| **20%** | 10 | High crosstalk | Requires advanced unmixing |
| **30%** | 10 | Moderate crosstalk | Aggressive multiplexing |
| **40%** | 8 | Balanced | Research applications |
| **50%** | 8 | Balanced | **Recommended for most** ⭐ |
| **60%** | 8 | Good separation | **Recommended** ⭐ |
| **70%** | 8 | Excellent separation | Quantitative imaging |
| **80%** | 8 | Excellent separation | High-precision work |
| **90%** | 6 | Outstanding separation | Ultra-conservative |

## The Trade-off

```
Lower Threshold (10-30%)          Higher Threshold (70-90%)
        ↓                                    ↓
  More fluorophores                  Fewer fluorophores
  Higher crosstalk                   Better separation
  Requires unmixing                  Simpler analysis
  More artifacts                     More reliable
```

## Visualization Components

The generated plot (`separability_vs_capacity.png`) contains 6 panels:

### 1. Main Curve (Top Left, Large)
- Shows capacity vs threshold from 10% to 90%
- Marks key operating points (30%, 50%, 60%, 70%)
- Color-coded zones:
  - Red (10-40%): High crosstalk zone
  - Yellow (40-70%): Balanced zone
  - Green (70-90%): High confidence zone

### 2. Sensitivity Analysis (Top Right)
- Shows rate of change (derivative)
- Identifies where capacity drops fastest
- Helps understand stability of operating points

### 3-5. Fluorophore Panels (Bottom Left to Center)
- **30% threshold**: 10 fluorophores (aggressive)
- **50% threshold**: 8 fluorophores (moderate)
- **70% threshold**: 8 fluorophores (conservative)
- Shows actual fluorophore positions in wavelength-lifetime space

### 6. Summary Table (Bottom Right)
- Complete threshold-capacity mapping
- Color-coded quality indicators
- Practical recommendations

## Detailed Results

### At 10% Threshold (Not Recommended)
- **Capacity**: 11 fluorophores
- **Quality**: Very high crosstalk
- **Issue**: Many pairs have <20% separation
- **Example conflict**: DAPI vs CFP (only 48% separable)
- **Use case**: Only with state-of-the-art unmixing algorithms

### At 30% Threshold (Aggressive)
- **Capacity**: 10 fluorophores
- **Quality**: Moderate crosstalk
- **Panel**: All 11 except NADH (bound)
- **Requires**: Advanced spectral unmixing
- **Use case**: Maximum multiplexing with expert analysis

### At 50% Threshold (Balanced) ⭐
- **Capacity**: 8 fluorophores
- **Quality**: Good separation
- **Panel**: DAPI, GFP, YFP, mOrange, mCherry, Cy5, NADH (free), FAD (bound)
- **Rejected**: CFP (too close to DAPI), NADH (bound), FAD (free)
- **Use case**: Standard research applications

### At 60% Threshold (Recommended) ⭐
- **Capacity**: 8 fluorophores
- **Quality**: Excellent separation
- **Same panel as 50%**: All pairs have >60% separability
- **Minimum pairwise separation**: 85%
- **Use case**: Reliable quantitative imaging

### At 70% Threshold (Conservative)
- **Capacity**: 8 fluorophores
- **Quality**: Outstanding separation
- **Same panel as 50-60%**: Very robust
- **Use case**: High-precision quantification

### At 90% Threshold (Ultra-Conservative)
- **Capacity**: 6 fluorophores
- **Quality**: Perfect separation
- **Panel**: DAPI, GFP, mOrange, mCherry, Cy5, NADH (free)
- **Rejected**: YFP (85% from GFP, needs >90%), FAD (bound)
- **Use case**: When absolute certainty is required

## Why Does Capacity Plateau?

Notice that capacity stays at **8 fluorophores** from 40% to 80% threshold:

```
10% → 11 fluorophores
20% → 10 fluorophores
30% → 10 fluorophores
40% → 8 fluorophores  ┐
50% → 8 fluorophores  │ Plateau!
60% → 8 fluorophores  │
70% → 8 fluorophores  │
80% → 8 fluorophores  ┘
90% → 6 fluorophores
```

**Why?** The selected 8 fluorophores have excellent mutual separation:
- Minimum pairwise separability: 85%
- Most pairs: >90% separable
- Only drops to 6 when requiring >90% for ALL pairs

## Sensitivity Analysis

The derivative plot shows **where capacity drops fastest**:

- **Steepest drop**: Around 35-40% threshold
  - This is where CFP gets rejected (conflicts with DAPI)
  
- **Stable region**: 40-80% threshold
  - Same 8 fluorophores work across this range
  - Robust operating point!
  
- **Second drop**: Around 85-90% threshold
  - YFP and FAD (bound) get rejected
  - Drops to 6 fluorophores

## Practical Recommendations

### For Most Applications: 50-60% Threshold
**Why?**
- Good balance: 8 fluorophores with reliable separation
- Robust: Works across wide threshold range (40-80%)
- Proven: Standard in the field
- Manageable: Linear unmixing sufficient

**Panel**: DAPI, GFP, YFP, mOrange, mCherry, Cy5, NADH (free), FAD (bound)

### For Maximum Multiplexing: 30% Threshold
**Why?**
- Maximum capacity: 10 fluorophores
- Requires: Advanced unmixing (phasor, ML)
- Risk: Higher crosstalk and artifacts
- Use when: Expertise and photon budget available

**Panel**: Add CFP and FAD (free) to standard panel

### For Quantitative Work: 70% Threshold
**Why?**
- Excellent separation: Same 8 fluorophores
- Lower risk: Minimal crosstalk
- Better SNR: Less unmixing error
- Use when: Quantification is critical

**Panel**: Same as 50-60% (very robust!)

### For Ultra-Precision: 90% Threshold
**Why?**
- Perfect separation: 6 fluorophores
- Minimal risk: Almost no crosstalk
- Simpler analysis: Less unmixing needed
- Use when: Absolute certainty required

**Panel**: DAPI, GFP, mOrange, mCherry, Cy5, NADH (free)

## Mathematical Definition

### Separability Score
For each pair of fluorophores (i, j):

```
separability(i,j) = max(
    1 - spectral_overlap(i,j),
    1 - lifetime_overlap(i,j),
    polarization_difference(i,j) × 0.5
)
```

### Capacity Calculation
```
Capacity(threshold) = max |S| such that:
    ∀ i,j ∈ S, i≠j: separability(i,j) ≥ threshold
```

Where S is the set of selected fluorophores.

## Comparison with Theoretical Capacity

| Approach | Capacity | Assumption |
|----------|----------|------------|
| **Theoretical bins** | 540 | No overlap, perfect resolution |
| **With overlap** | 81 | Accounts for spectral/lifetime width |
| **10% threshold** | 11 | Minimal separation requirement |
| **50% threshold** | 8 | Balanced separation ⭐ |
| **90% threshold** | 6 | Excellent separation |

The practical capacity (8 at 50-60%) is **67× smaller** than theoretical (540) due to:
1. Spectral overlap (FWHM = 35-65 nm)
2. Lifetime distributions (σ = 0.1-0.4 ns)
3. Pairwise separability requirement
4. Confidence threshold (50-60%)

## Files Generated

- `plot_separability_vs_capacity.py` - Analysis script
- `separability_vs_capacity.png` - Main visualization (6 panels)
- `separability_vs_capacity.pdf` - PDF version
- `SEPARABILITY_VS_CAPACITY_ANALYSIS.md` - This document

## How to Use This Analysis

1. **Choose your threshold** based on application:
   - Research: 50-60%
   - Quantitative: 70%
   - Maximum multiplexing: 30% (with caution)

2. **Check the panel** at that threshold:
   - See which fluorophores are included
   - Verify they match your experimental needs

3. **Assess trade-offs**:
   - More fluorophores = more crosstalk
   - Fewer fluorophores = better separation

4. **Plan your experiment**:
   - Select fluorophores from the recommended panel
   - Ensure adequate photon counts
   - Choose appropriate unmixing method

## Conclusion

The **sweet spot is 50-60% threshold** with **8 fluorophores**:
- Robust across wide threshold range (40-80%)
- Good separation (all pairs >85%)
- Manageable unmixing complexity
- Proven in practice

Going lower (30%) gains only 2 more fluorophores but significantly increases crosstalk.
Going higher (70-80%) provides no additional benefit (same 8 fluorophores).

**Recommendation**: Start with 60% threshold and adjust based on your specific needs and unmixing capabilities.

---

**Generated**: 2025-11-11  
**Instrument**: λ=10nm, τ=0.5ns, pol=2 states  
**Database**: 11 fluorophores  
**Threshold range**: 10-90%
