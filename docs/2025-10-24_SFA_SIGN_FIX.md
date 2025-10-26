# SFA Sign Correction - Summary

## Problem Identified

The SFA (Slow Feature Analysis) learning rule was implementing **gradient ascent** instead of **gradient descent** on the slowness objective.

### Root Cause

**File**: `neurobridge/plasticity/signals/hpf.py`
**Class**: `SignalTemporalSurrogate`
**Line**: 351 (original)

The temporal surrogate gradient was computing:
```python
L_prime = gamma * torch.tanh(delta_y / self.delta)
```

Where `delta_y = y_smooth(t) - y_smooth(t-1)` represents the temporal change in output.

### Why This Was Wrong

SFA's objective is to **MINIMIZE** temporal variations:
```
J = E[(dy/dt)²]  →  we want to minimize this (slowness principle)
```

For gradient **DESCENT**, the weight update should be:
```
Δw ∝ -∂J/∂w ∝ -(dy/dt) · eligibility
```

The implementation was missing the **negative sign**, causing gradient **ASCENT**:
- Neurons with high mod_weight → cause MORE output variation
- Gradient ascent → INCREASES their weights
- Oja normalization → DECREASES strong weights more
- **Net result**: Negative correlation (learning inverts the correct structure)

## Solution

Added negative sign to all surrogate gradient types:

### Changes Made

**File**: `neurobridge/plasticity/signals/hpf.py`

1. **Tanh surrogate** (line 354):
```python
# Before:
L_prime = self.gamma * torch.tanh(delta_y / self.delta)

# After:
L_prime = -self.gamma * torch.tanh(delta_y / self.delta)  # Negative for gradient descent
```

2. **Triangular surrogate** (line 359):
```python
# Before:
L_prime = self.gamma * torch.clamp(1.0 - torch.abs(delta_y) / self.delta, min=0.0)

# After:
L_prime = -self.gamma * torch.sign(delta_y) * torch.clamp(1.0 - torch.abs(delta_y) / self.delta, min=0.0)
```

3. **Sigmoid surrogate** (line 363):
```python
# Before:
L_prime = self.gamma * (torch.sigmoid(delta_y / self.delta) - 0.5)

# After:
L_prime = -self.gamma * (torch.sigmoid(delta_y / self.delta) - 0.5)
```

## Verification Results

### Test 1: 64 Neurons, Fixed Weights, No Oja
- **Before fix**: Correlation = -0.687 ✗
- **After fix**: Correlation = +0.799 ✓

### Test 2: Comprehensive Test (All Configurations)
All configurations now show **strong positive correlation**:

| Configuration | Corr(change) | Status |
|--------------|--------------|--------|
| Fixed weights, no Oja | +0.799 | ✓ |
| Fixed weights, with Oja | +0.799 | ✓ |
| Random weights, no Oja | +0.774 | ✓ |
| Random weights, with Oja | +0.774 | ✓ |

## Impact

- ✓ SFA now correctly learns to extract slow features
- ✓ Weights increase for neurons with strong modulation
- ✓ Works with all configurations (with/without Oja, fixed/random init)
- ✓ Scaling behavior is now correct (2-64 neurons all work)

## Theory

The corrected rule now implements:
```
Δw_ij = -η · e_ij · (dy/dt)_j - β · w_ij · y²_j
```

Where:
- First term: Hebbian-like, but with NEGATIVE sign (gradient descent on slowness)
- Second term: Oja normalization (prevents unbounded growth)
- Result: Synapses are strengthened when they contribute to REDUCING output variations

This aligns with the theoretical SFA objective from:
> Sprekeler, H., et al. (2007). Slowness: An objective for spike-timing-dependent
> plasticity? PLoS Computational Biology, 3(6), e112.

## Signal Type Comparison

After fixing SignalTemporalSurrogate, tested all three SFA signal types:

| Signal Type | Corr(change) | Status | Notes |
|------------|--------------|--------|-------|
| surrogate  | +0.799       | ✓      | Best performance (after fix) |
| hpf        | +0.304       | ✓      | Works correctly, weaker learning |
| voltage    | +0.284       | ✓      | Works correctly, weaker learning |

**Conclusion**: SignalHPFPost and SignalHPFVoltage do NOT need sign correction. They produce weaker learning signals than the surrogate method, but they learn in the correct direction. The cascaded low-pass filtering approach naturally provides the correct sign through the high-pass filter dynamics.

## Files Modified

1. `/home/alvarogr/Desktop/Neurobridge project/neurobridge/plasticity/signals/hpf.py`
   - Lines 349-367: Added negative signs to SignalTemporalSurrogate (tanh, triangular, sigmoid)
   - Lines 118-120 (SignalHPFPost): NO CHANGE - works correctly as-is
   - Lines 222-225 (SignalHPFVoltage): NO CHANGE - works correctly as-is

## Test Files Created

1. `10b_SFA_minimal_2neurons.py` - Minimal test case (2 neurons)
2. `10c_SFA_scaling_test.py` - Scaling test (2, 4, 8, 16, 32, 64 neurons)
3. `10d_SFA_beta_test.py` - Oja term investigation
4. `10e_SFA_diagnostic.py` - Detailed diagnostics
5. `10f_SFA_sign_test.py` - Sign correction test framework
6. `10g_SFA_quick_test.py` - Quick verification (no plots)
7. `10h_SFA_final_test.py` - Comprehensive verification
8. `10i_SFA_test_hpf_signals.py` - HPF signal types comparison

## Date

2025-10-24

## Status

✓ **FIXED AND VERIFIED**
