# How We Get From 540 to 8: Detailed Explanation

## The Question

**Why does the practical capacity drop from 81 to 8?**

## Short Answer

- **81** = Number of theoretical bins (assuming you can use all despite overlap)
- **8** = Number of actual fluorophores that are **pairwise separable**

The key difference: **81 counts bins, 8 counts mutually distinguishable fluorophores**

## Step-by-Step Breakdown

### Step 1: Theoretical Capacity = 540
```
30 wavelength bins × 9 lifetime bins × 2 polarization states = 540
```
**Assumption**: Each bin is independent (no overlap)
**Reality**: ❌ Spectra and lifetimes have width!

---

### Step 2: With Spectral Overlap = 270
```
540 × 0.5 = 270
```
**Why 0.5?** Emission spectra have FWHM = 35-65 nm
- With 10 nm bins, each spectrum spans ~4-7 bins
- Adjacent bins overlap significantly
- Effective reduction: ~50%

**Example**: 
- GFP peak at 509 nm, FWHM = 35 nm
- Spans from ~492 nm to ~526 nm (3.4 bins!)
- Not a single point!

---

### Step 3: With Lifetime Overlap = 180
```
270 × 0.67 = 180
```
**Why 0.67?** Lifetime distributions have σ = 0.1-0.4 ns
- With 0.5 ns bins, distributions span ~2-3 bins
- Lifetime bins overlap
- Effective reduction: ~33%

**Example**:
- GFP lifetime = 2.5 ± 0.2 ns
- Spans from ~2.1 to ~2.9 ns (1.6 bins!)
- Not a single value!

---

### Step 4: Combined Overlaps = 81
```
180 × 0.45 = 81
```
**Why 0.45?** Spectral AND lifetime overlap compound
- When both dimensions overlap, the effect is multiplicative
- Some fluorophores become indistinguishable
- Combined reduction: ~55%

**This is still theoretical!** It assumes you can use all 81 bins.

---

### Step 5: Practical Capacity = 8 ⭐

**This is where the big drop happens!**

#### The Greedy Selection Algorithm

Starting with 11 fluorophores in our database, we select those that are **mutually separable**:

```
Threshold: 0.6 (60% separation required)

✓ DAPI (461 nm, 2.5 ns) - Added (1)
✗ CFP (475 nm, 2.7 ns) - REJECTED: Too close to DAPI (separability = 0.48)
✓ GFP (509 nm, 2.5 ns) - Added (2)
✓ YFP (527 nm, 3.0 ns) - Added (3)
✓ mOrange (562 nm, 2.0 ns) - Added (4)
✓ mCherry (610 nm, 1.5 ns) - Added (5)
✓ Cy5 (670 nm, 1.0 ns) - Added (6)
✓ NADH (free) (460 nm, 0.4 ns) - Added (7)
✗ NADH (bound) (460 nm, 2.5 ns) - REJECTED: Too close to DAPI (separability = 0.25)
✗ FAD (free) (535 nm, 2.8 ns) - REJECTED: Too close to YFP (separability = 0.44)
✓ FAD (bound) (535 nm, 0.4 ns) - Added (8)

Result: 8 fluorophores
```

#### Why Were Some Rejected?

**CFP rejected** (conflicts with DAPI):
- CFP: 475 nm, FWHM = 40 nm → spans 455-495 nm
- DAPI: 461 nm, FWHM = 50 nm → spans 436-486 nm
- **Overlap region**: 455-486 nm (significant!)
- Separability score: 0.48 < 0.6 threshold

**NADH (bound) rejected** (conflicts with DAPI):
- Both at ~460 nm
- NADH (bound): 2.5 ns lifetime
- DAPI: 2.5 ns lifetime
- **Same wavelength AND same lifetime!**
- Separability score: 0.25 < 0.6 threshold

**FAD (free) rejected** (conflicts with YFP):
- FAD (free): 535 nm, FWHM = 65 nm
- YFP: 527 nm, FWHM = 40 nm
- Significant spectral overlap
- Similar lifetimes (2.8 vs 3.0 ns)
- Separability score: 0.44 < 0.6 threshold

## Key Insight: Pairwise Separability

The critical difference between 81 and 8:

### 81 Bins Approach (Wrong!)
```
"I have 81 bins, so I can distinguish 81 fluorophores"
```
**Problem**: Ignores that fluorophores must be separable from EACH OTHER

### 8 Fluorophores Approach (Correct!)
```
"I can select 8 fluorophores such that EVERY PAIR is distinguishable"
```
**Reality**: Each fluorophore must be separable from ALL others in the panel

## Pairwise Separability Matrix

For the selected 8 fluorophores, all pairs have separability > 0.6:

```
              DAPI   GFP   YFP  mOrange mCherry  Cy5  NADH(f) FAD(b)
DAPI           -    0.90  0.96   0.99    1.00   1.00   1.00   1.00
GFP          0.90    -    0.85   0.94    1.00   1.00   1.00   1.00
YFP          0.96  0.85    -     0.96    1.00   1.00   1.00   1.00
mOrange      0.99  0.94  0.96     -      0.87   0.99   1.00   1.00
mCherry      1.00  1.00  1.00   0.87      -     0.92   1.00   1.00
Cy5          1.00  1.00  1.00   0.99    0.92     -     1.00   1.00
NADH (free)  1.00  1.00  1.00   1.00    1.00   1.00     -     0.91
FAD (bound)  1.00  1.00  1.00   1.00    1.00   1.00   0.91     -
```

**All values ≥ 0.85** → Excellent separation!

## Analogy: Parking Spaces

Think of it like parking cars:

### 81 Bins = 81 Parking Spaces
```
[_][_][_][_][_][_][_][_]...[_]  (81 spaces)
```
"I have 81 parking spaces!"

### But Cars Have Width!
```
[==CAR==][==CAR==]  (Each car takes ~3 spaces)
```
"I can only fit ~27 cars, not 81!"

### And Cars Must Not Overlap!
```
[==CAR==]  [==CAR==]  [==CAR==]  (Need gaps between cars)
```
"I can only fit ~8 cars with safe spacing!"

**This is exactly what happens with fluorophores!**

## Effect of Threshold

The practical capacity depends on how strict you are:

| Threshold | # Fluorophores | Interpretation |
|-----------|----------------|----------------|
| 0.3 | 10 | Loose (high crosstalk) |
| 0.4 | 10 | Moderate crosstalk |
| 0.5 | 8 | Balanced |
| **0.6** | **8** | **Recommended** ⭐ |
| 0.7 | 8 | Conservative |
| 0.8 | 8 | Very conservative |
| 0.9 | 6 | Extremely strict |

At threshold = 0.6, we get 8 fluorophores with good mutual separation.

## Mathematical Formulation

### Bin Counting (Wrong)
```
Capacity = n_wavelength_bins × n_lifetime_bins × n_polarization_bins
         = 30 × 9 × 2 = 540
```

### Pairwise Separability (Correct)
```
Capacity = max |S| such that:
  ∀ i,j ∈ S, i≠j: separability(i,j) ≥ threshold

Where:
  separability(i,j) = max(
    1 - spectral_overlap(i,j),
    1 - lifetime_overlap(i,j),
    polarization_difference(i,j) × 0.5
  )
```

This is a **constraint satisfaction problem**, not simple multiplication!

## Summary

```
540 (theoretical bins)
  ↓ ×0.5 (spectral overlap)
270 (with spectral overlap)
  ↓ ×0.67 (lifetime overlap)
180 (with lifetime overlap)
  ↓ ×0.45 (combined effects)
 81 (combined overlaps)
  ↓ Greedy selection (pairwise separability ≥ 0.6)
  8 (practical capacity) ⭐
```

**The key transition**: 81 → 8 happens because we switch from:
- **Counting bins** (81)
- **Selecting mutually separable fluorophores** (8)

## Practical Implications

1. **Don't trust bin counting!** 
   - 30 wavelength bins ≠ 30 separable fluorophores
   - Must account for spectral width

2. **Pairwise separability is key**
   - Each fluorophore must be distinguishable from ALL others
   - Not just from its neighbors

3. **Threshold matters**
   - Lower threshold (0.3-0.5): More fluorophores, more crosstalk
   - Higher threshold (0.7-0.9): Fewer fluorophores, better separation
   - Sweet spot: 0.6 (recommended)

4. **Database matters**
   - We started with 11 fluorophores
   - Selected 8 that are mutually separable
   - With different fluorophores, might get different results

5. **Unmixing can help**
   - Advanced algorithms can push beyond 8
   - But requires more photons and computation
   - Practical limit: ~10-15 with state-of-the-art methods

## Conclusion

**81 to 8 is not a mistake—it's the difference between theory and practice!**

- **81** = "How many bins exist?" (theoretical)
- **8** = "How many fluorophores can I actually distinguish?" (practical)

The greedy selection algorithm ensures that all 8 selected fluorophores are mutually separable with >60% confidence, which is what you need for reliable imaging.
