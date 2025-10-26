"""
Performance benchmark comparing legacy STDP vs modular e-prop STDP.

This benchmark compares:
- STDPSparse (legacy implementation)
- PlasticSparse with STDP rule (modular e-prop framework)

Criteria: Modular implementation should achieve ≥95% performance of legacy.
"""

import torch
import time
from neurobridge.connection import ConnectionSpec
from neurobridge.neurons import IFNeurons
from neurobridge.sparse_connections import STDPSparse
from neurobridge.plastic_connections import PlasticSparse
from neurobridge import globals
from unittest.mock import Mock


def setup_mock_simulator():
    """Setup a mock simulator for benchmarking."""
    mock_simulator = Mock()
    mock_circuit = Mock()
    mock_circuit.t = 0
    mock_circuit.rank = 0
    mock_simulator.local_circuit = mock_circuit
    globals.simulator = mock_simulator
    return mock_simulator


def benchmark_legacy_stdp(n_neurons=1000, n_synapses=10000, n_steps=1000, device='cpu'):
    """Benchmark legacy STDPSparse implementation.

    Parameters
    ----------
    n_neurons : int
        Number of neurons in pre and post populations.
    n_synapses : int
        Number of synapses.
    n_steps : int
        Number of simulation steps.
    device : str
        Device to run on ('cpu' or 'cuda').

    Returns
    -------
    float
        Time in seconds for n_steps.
    """
    setup_mock_simulator()
    device = torch.device(device)

    # Create neuron groups
    pre = IFNeurons(n_neurons=n_neurons, device=device, delay_max=20)
    pos = IFNeurons(n_neurons=n_neurons, device=device, delay_max=20)

    # Create random sparse connectivity
    idx_pre = torch.randint(0, n_neurons, (n_synapses,), dtype=torch.long)
    idx_pos = torch.randint(0, n_neurons, (n_synapses,), dtype=torch.long)
    weights = torch.rand(n_synapses, dtype=torch.float32) * 0.5
    delays = torch.ones(n_synapses, dtype=torch.long)

    spec = ConnectionSpec(
        pre=pre, pos=pos,
        src_idx=idx_pre, tgt_idx=idx_pos,
        weight=weights, delay=delays,
        params={
            "A_plus": 1e-4,
            "A_minus": -1.2e-4,
            "tau_plus": 20e-3,
            "tau_minus": 20e-3,
            "w_min": 0.0,
            "w_max": 1.0,
            "oja_decay": 1e-5
        }
    )
    conn = STDPSparse(spec)

    # Warm-up
    for _ in range(10):
        conn._update()

    if device.type == 'cuda':
        torch.cuda.synchronize()

    # Benchmark
    start_time = time.perf_counter()

    for t in range(n_steps):
        globals.simulator.local_circuit.t = t

        # Inject random spikes
        if t % 10 == 0:
            pre_spikes = torch.rand(n_neurons, device=device) < 0.05
            pos_spikes = torch.rand(n_neurons, device=device) < 0.05
            pre.inject_spikes(pre_spikes)
            pos.inject_spikes(pos_spikes)
            pre._process()
            pos._process()

        conn._update()

    if device.type == 'cuda':
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start_time

    return elapsed


def benchmark_modular_stdp(n_neurons=1000, n_synapses=10000, n_steps=1000, device='cpu'):
    """Benchmark modular PlasticSparse with STDP rule.

    Parameters
    ----------
    n_neurons : int
        Number of neurons in pre and post populations.
    n_synapses : int
        Number of synapses.
    n_steps : int
        Number of simulation steps.
    device : str
        Device to run on ('cpu' or 'cuda').

    Returns
    -------
    float
        Time in seconds for n_steps.
    """
    setup_mock_simulator()
    device = torch.device(device)

    # Create neuron groups
    pre = IFNeurons(n_neurons=n_neurons, device=device, delay_max=20)
    pos = IFNeurons(n_neurons=n_neurons, device=device, delay_max=20)

    # Create random sparse connectivity
    idx_pre = torch.randint(0, n_neurons, (n_synapses,), dtype=torch.long)
    idx_pos = torch.randint(0, n_neurons, (n_synapses,), dtype=torch.long)
    weights = torch.rand(n_synapses, dtype=torch.float32) * 0.5
    delays = torch.ones(n_synapses, dtype=torch.long)

    spec = ConnectionSpec(
        pre=pre, pos=pos,
        src_idx=idx_pre, tgt_idx=idx_pos,
        weight=weights, delay=delays,
        params={
            "plasticity": {
                "name": "stdp",
                "params": {
                    "A_plus": 1e-4,
                    "A_minus": -1.2e-4,
                    "tau_pre": 20e-3,
                    "tau_post": 20e-3,
                    "w_min": 0.0,
                    "w_max": 1.0,
                    "oja_decay": 1e-5
                }
            }
        }
    )
    conn = PlasticSparse(spec)

    # Warm-up
    for _ in range(10):
        conn._update()

    if device.type == 'cuda':
        torch.cuda.synchronize()

    # Benchmark
    start_time = time.perf_counter()

    for t in range(n_steps):
        globals.simulator.local_circuit.t = t

        # Inject random spikes
        if t % 10 == 0:
            pre_spikes = torch.rand(n_neurons, device=device) < 0.05
            pos_spikes = torch.rand(n_neurons, device=device) < 0.05
            pre.inject_spikes(pre_spikes)
            pos.inject_spikes(pos_spikes)
            pre._process()
            pos._process()

        conn._update()

    if device.type == 'cuda':
        torch.cuda.synchronize()

    elapsed = time.perf_counter() - start_time

    return elapsed


def run_benchmark(device='cpu', n_trials=5):
    """Run complete benchmark comparing legacy vs modular STDP.

    Parameters
    ----------
    device : str
        Device to run on ('cpu' or 'cuda').
    n_trials : int
        Number of trials to average over.

    Returns
    -------
    dict
        Benchmark results with timing statistics.
    """
    print(f"\n{'='*70}")
    print(f"STDP Performance Benchmark on {device.upper()}")
    print(f"{'='*70}\n")

    configs = [
        {"n_neurons": 500, "n_synapses": 5000, "n_steps": 1000, "name": "Small"},
        {"n_neurons": 1000, "n_synapses": 10000, "n_steps": 1000, "name": "Medium"},
        {"n_neurons": 2000, "n_synapses": 20000, "n_steps": 1000, "name": "Large"},
    ]

    results = {}

    for config in configs:
        name = config.pop("name")
        print(f"\nConfiguration: {name}")
        print(f"  Neurons: {config['n_neurons']}, Synapses: {config['n_synapses']}, Steps: {config['n_steps']}")
        print(f"  Running {n_trials} trials...")

        # Legacy STDP
        legacy_times = []
        for trial in range(n_trials):
            t = benchmark_legacy_stdp(**config, device=device)
            legacy_times.append(t)
            print(f"    Legacy trial {trial+1}/{n_trials}: {t:.4f}s")

        # Modular STDP
        modular_times = []
        for trial in range(n_trials):
            t = benchmark_modular_stdp(**config, device=device)
            modular_times.append(t)
            print(f"    Modular trial {trial+1}/{n_trials}: {t:.4f}s")

        # Compute statistics
        legacy_mean = sum(legacy_times) / len(legacy_times)
        modular_mean = sum(modular_times) / len(modular_times)
        speedup = legacy_mean / modular_mean
        overhead_pct = ((modular_mean - legacy_mean) / legacy_mean) * 100

        results[name] = {
            "legacy_mean": legacy_mean,
            "modular_mean": modular_mean,
            "speedup": speedup,
            "overhead_pct": overhead_pct,
            "config": config
        }

        print(f"\n  Results:")
        print(f"    Legacy mean:   {legacy_mean:.4f}s")
        print(f"    Modular mean:  {modular_mean:.4f}s")
        print(f"    Speedup:       {speedup:.2f}x")
        print(f"    Overhead:      {overhead_pct:+.1f}%")

        # Check success criterion
        if modular_mean <= legacy_mean / 0.95:  # Modular should be within 5% slower
            print(f"    ✓ PASS: Performance criterion met (≥95% of legacy)")
        else:
            print(f"    ✗ FAIL: Performance criterion NOT met (modular is {overhead_pct:.1f}% slower)")

    print(f"\n{'='*70}")
    print("Summary")
    print(f"{'='*70}\n")

    all_pass = all(r["overhead_pct"] <= 5.0 for r in results.values())

    if all_pass:
        print("✓ All configurations PASS performance criteria (modular ≥95% of legacy)")
    else:
        print("✗ Some configurations FAIL performance criteria")

    print()

    return results


if __name__ == "__main__":
    # Run on CPU
    cpu_results = run_benchmark(device='cpu', n_trials=5)

    # Run on GPU if available
    if torch.cuda.is_available():
        print("\n" + "="*70)
        cuda_results = run_benchmark(device='cuda', n_trials=5)
    else:
        print("\nGPU not available, skipping CUDA benchmark.")
