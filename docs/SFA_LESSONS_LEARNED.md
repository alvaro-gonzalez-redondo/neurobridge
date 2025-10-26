# SFA Implementation: Lessons Learned

This document summarizes critical insights discovered during the implementation and debugging of Slow Feature Analysis (SFA) in Neurobridge.

---

## 1. The Harmonic Interference Problem

### Problem
When testing SFA with multiple slow signals at different frequencies, using integer multiples **completely breaks learning**.

**Example of failure:**
```python
# ✗ WRONG: Integer multiples
slow_freqs = [0.5, 1.0, 2.0]  # Hz (ratios: ×2, ×4)

Results:
- Correlation(slowness, weights): -0.330 (NEGATIVE!)
- Output follows: 2.0 Hz (fastest, not slowest)
- Weight ordering: FAST > slow (inverted)
```

### Root Cause
- **Harmonic interference**: Frequencies that are integer multiples create constructive/destructive interference patterns
- **Phase alignment**: 2.0 Hz is exactly in phase with 0.5 Hz at regular intervals (every 2 seconds)
- **Ambiguous learning signal**: The eligibility traces cannot distinguish which frequency is actually slow vs fast

### Solution
Use **irrational frequency ratios** to avoid harmonic relationships:

```python
# ✓ CORRECT: Irrational ratios (golden ratio φ²)
phi = (1 + 5**0.5) / 2
slow_freqs = [0.5 * phi**(2*n) for n in range(5)]
# ≈ [0.5, 1.31, 3.43, 8.97, 23.49] Hz

Results:
- Correlation(slowness, weights): +0.914 (STRONG positive!)
- Output follows: 0.5 Hz (slowest, correct)
- Weight ordering: slow > fast (monotonic decrease)
```

**Key insight**: Natural signals rarely have perfect harmonic relationships. Using irrational ratios is more realistic and avoids mathematical artifacts.

---

## 2. DoE Timescale Matching

### Problem
The Difference-of-Exponentials (DoE) eligibility filter acts as a band-pass filter. If its timescales don't match the signal frequencies being learned, **SFA cannot discriminate between slow and fast features**.

### Understanding the DoE Band-Pass

The DoE filter is defined as:
```python
e(t) = x_fast(t) - x_slow(t)
```

Where:
- `x_fast` decays with time constant `tau_fast`
- `x_slow` decays with time constant `tau_slow`

This creates a band-pass filter with cutoff frequencies:
```
f_low  = 1 / (2π · tau_slow)   # High-pass cutoff
f_high = 1 / (2π · tau_fast)   # Low-pass cutoff
```

**The filter responds maximally to frequencies in the range [f_low, f_high].**

### Example of Mismatch

**Scenario 1: Timescales too fast for signals**
```python
# Signals: 0.5-2.0 Hz (periods: 500-2000 ms)
tau_fast = 10e-3    # 10 ms  → f_high ≈ 16 Hz
tau_slow = 100e-3   # 100 ms → f_low  ≈ 1.6 Hz

# Band-pass: [1.6, 16] Hz
# Problem: All test signals (0.5-2.0 Hz) are at or below the band!
# Result: Poor discrimination, all signals look "equally slow"
```

**Scenario 2: Correctly matched timescales**
```python
# Signals: 0.5-3.4 Hz (with irrational ratios)
tau_fast = 100e-3   # 100 ms  → f_high ≈ 1.6 Hz
tau_slow = 2500e-3  # 2500 ms → f_low  ≈ 0.06 Hz

# Band-pass: [0.06, 1.6] Hz
# Result: 0.5 Hz is in center of band (optimal)
#         3.4 Hz is at upper edge (attenuated)
#         Higher frequencies are filtered out
# Excellent discrimination!
```

### Rule of Thumb

**Target the center of your band-pass at the slowest frequency you want to learn:**

```
tau_slow ≈ 4-5 × period_of_slowest_signal
tau_fast ≈ period_of_slowest_signal / 2
```

For signals around 0.5-3 Hz:
- Period of 0.5 Hz = 2000 ms
- `tau_slow = 2500 ms` (slightly longer than slowest period)
- `tau_fast = 100 ms` (captures variations within signals)

---

## 3. Signal Type Comparison

We tested three types of learning signals in SFA:

### A. Surrogate Temporal Gradient
```python
"signal_type": "surrogate"
"gamma": 1.0
"delta": 0.1
"surrogate_type": "tanh"
"v_scale": 1.0
```

**How it works:**
- Computes smooth temporal derivative: `L' = -γ·tanh(Δy/δ)`
- Based on changes in smoothed postsynaptic voltage
- Negative sign for gradient descent (minimize temporal variation)

**Results:**
- Correlation(slowness, weights): +0.871
- Output follows 0.5 Hz: |r|=0.537
- **Needs Oja normalization** (β=1e-3)
- More hyperparameters to tune (γ, δ, surrogate type)

**Pros:**
- Flexible (multiple surrogate functions)
- Voltage-based (membrane potential)
- Fine-grained control

**Cons:**
- Requires normalization
- More parameters to tune
- Slightly lower correlation

### B. High-Pass Filter (HPF) on Spikes
```python
"signal_type": "hpf"
"tau_hpf": 100e-3
```

**How it works:**
- Filters presynaptic spike trains with exponential HPF
- Removes slow baseline, keeps fast fluctuations
- Directly captures temporal structure

**Results:**
- Correlation(slowness, weights): **+0.914** ✓ (best!)
- Output follows 0.5 Hz: |r|=0.547 ✓
- **No Oja needed** (β=0)
- Simpler parameterization

**Pros:**
- Highest correlation
- No normalization required
- Fewer hyperparameters
- More stable learning

**Cons:**
- Less flexible (no surrogate choice)
- Spike-based only (not voltage)

### C. Voltage HPF
```python
"signal_type": "voltage"
"tau_hpf": 100e-3
```

**How it works:**
- Applies HPF to postsynaptic voltage
- Similar to HPF but on membrane potential
- Captures voltage fluctuations

**Results:**
- Correlation: +0.284 (from earlier tests)
- Lower performance than spike-based HPF

### Recommendation

**For most use cases, use HPF (`signal_type="hpf"`):**
- Best performance
- Simplest parameterization
- Most stable
- No normalization needed

**Use surrogate if:**
- You need voltage-based learning
- You want fine-grained control over gradient shape
- You're exploring different surrogate functions

---

## 4. Learning Rate and Oja Regularization

### Learning Rate Scaling

Different signal types require vastly different learning rates:

| Signal Type | Typical η | Typical β | Notes |
|-------------|-----------|-----------|-------|
| Surrogate | 3e-5 | 1e-3 | Moderate values |
| HPF | **1e-8** | **0** | Much smaller η! |
| Voltage | 1e-5 | 1e-4 | Intermediate |

**Why HPF needs smaller η:**
- HPF signal magnitudes are larger (direct spike filtering)
- Surrogate gradient is already scaled by γ and tanh saturation
- Without Oja (β=0), need smaller η to prevent weight explosion

### Oja Regularization

**Oja's rule** adds weight decay proportional to output activity:
```python
Δw = η·e·L' - β·w·(L')²
```

The `-β·w·(L')²` term:
- Prevents weight explosion
- Normalizes weight vector magnitude
- Acts as implicit competition between inputs

**When you need Oja:**
- ✓ Surrogate gradients (prevent unbounded growth)
- ✓ Long training times
- ✓ Random weight initialization

**When Oja is not needed:**
- ✓ HPF signals (self-limiting)
- ✓ Short experiments
- ✓ Fixed weight initialization

---

## 5. Key Success Factors Checklist

✅ **Non-harmonic frequencies**
- Use irrational ratios (golden ratio, sqrt(2), etc.)
- Avoid integer multiples (0.5, 1.0, 2.0)

✅ **Matched DoE timescales**
- `tau_slow ≈ 4× slowest_period`
- `tau_fast ≈ 0.5× slowest_period`
- Check band-pass [1/(2πτ_slow), 1/(2πτ_fast)]

✅ **Appropriate learning rates**
- HPF: η~1e-8, β=0
- Surrogate: η~3e-5, β~1e-3
- Test with different seeds for robustness

✅ **Sufficient output activity**
- Target: 50-200 Hz output rate
- Too low: weak learning signal
- Too high: saturation effects

✅ **Adequate training time**
- Need at least 20-30 cycles of slowest signal
- For 0.5 Hz: minimum 40-60 seconds
- More is better for convergence

---

## 6. Diagnostic Metrics

Use these metrics to verify SFA is working correctly:

### Primary Metrics
1. **Correlation(slowness_score, final_weights)**
   - Expected: **positive** (>0.3 for weak, >0.7 for strong)
   - Indicates: Neurons tuned to slow signals have higher weights

2. **Correlation(output_activity, slowest_signal)**
   - Expected: **highest** for slowest frequency
   - Indicates: Output tracks the slowest feature

3. **Weight ordering by frequency**
   - Expected: **monotonic decrease** (slow > medium > fast)
   - Indicates: SFA correctly discriminates velocities

### Secondary Metrics
4. **Consistency check**
   - Weight-based prediction matches actual output behavior
   - Indicates: No disconnect between structure and function

5. **Output firing rate**
   - Expected: 50-200 Hz (active but not saturated)
   - Too low: insufficient learning signal
   - Too high: possible saturation

---

## 7. Common Pitfalls and Solutions

### Pitfall 1: Negative Correlation
**Symptom:** Correlation(slowness, weights) is negative
**Causes:**
- Harmonic frequencies (use irrational ratios)
- Wrong sign in learning rule (check surrogate has negative sign)
- Inverted slowness score calculation

**Solution:** Use φ² ratios, verify gradient descent sign

### Pitfall 2: No Learning
**Symptom:** Weights barely change (Δw < 1e-6)
**Causes:**
- Learning rate too low
- Output neuron not firing (check rate)
- DoE timescales mismatched

**Solution:** Increase η, check output rate > 20 Hz, adjust timescales

### Pitfall 3: Weight Explosion
**Symptom:** Weights hit w_max quickly, all saturate
**Causes:**
- Learning rate too high
- Missing Oja regularization
- Output neuron firing too fast

**Solution:** Decrease η, add β>0, reduce input weights

### Pitfall 4: Output Follows Wrong Frequency
**Symptom:** Output correlates with fast signal, not slow
**Causes:**
- DoE timescales too fast (all signals pass through)
- Harmonic interference
- Not enough training time

**Solution:** Increase tau_slow, use irrational freqs, train longer

---

## 8. Testing Protocol

When implementing or modifying SFA, follow this protocol:

1. **Unit test**: Single frequency, verify positive weight change
2. **Two-frequency test**: Slow (0.5 Hz) + Fast (2.0 Hz), verify slow > fast
3. **Multi-frequency test**: Use 5 frequencies with φ² ratios
4. **Timescale sweep**: Test different tau_fast/tau_slow combinations
5. **Robustness**: Multiple seeds (at least 3)
6. **Signal type comparison**: Test surrogate vs HPF

**Expected results:**
- Correlation > 0.7
- Output follows slowest signal
- Monotonic weight decrease with frequency

---

## 9. Future Directions

### Voltage-Based Normalization
- Implement z-score normalization: `(V - μ) / σ`
- Online computation of running mean/std
- Should reduce dependence on output firing rate

### Adaptive Timescales
- Automatically adjust tau_fast/tau_slow based on input statistics
- Could use frequency analysis of input spikes
- Make SFA more robust to different signal ranges

### Competitive Learning
- Multiple output neurons competing for different timescales
- Each neuron specializes in different velocity range
- More biologically realistic

---

## 10. References

**Original SFA paper:**
- Sprekeler, H., et al. (2007). "Slowness: An objective for spike-timing-dependent plasticity?" *PLoS Computational Biology*, 3(6), e112.

**Implementation details:**
- See `neurobridge/plasticity/recipes/sfa.py` for recipe
- See `neurobridge/plasticity/signals/hpf.py` for signal types
- See `neurobridge/plasticity/eligibility/sfa.py` for DoE implementation
- See `examples/10_SFA_slow_feature.py` for working example

---

**Last updated:** 2025-10-24
**Authors:** Neurobridge development team
