"""
Unit tests for the modular plasticity framework.

Tests cover:
- Interface contracts (EligibilityBase, LearningSignalBase, UpdatePolicyBase)
- STDP components (eligibility traces, signals, updates)
- PlasticSparse and PlasticDense connections
- Factory functions
- Multi-GPU restriction validation
"""

import pytest
import torch
from unittest.mock import Mock

from neurobridge.connection import ConnectionSpec
from neurobridge.neurons import IFNeurons
from neurobridge import globals
from neurobridge.plasticity.eligibility.stdp import EligibilitySTDPSparse, EligibilitySTDPDense
from neurobridge.plasticity.signals.post_spikes import SignalPostSpikes
from neurobridge.plasticity.updates.stdp import UpdateSTDPSparse, UpdateSTDPDense
from neurobridge.plasticity.rule import PlasticityRule
from neurobridge.plasticity.recipes.factories import build_rule_for_sparse, build_rule_for_dense
from neurobridge.sparse_connections import StaticSparse
from neurobridge.dense_connections import StaticDense


@pytest.fixture(autouse=True)
def setup_simulator():
    """Setup a mock simulator for all tests."""
    # Create mock simulator
    mock_simulator = Mock()
    mock_circuit = Mock()
    mock_circuit.t = 0
    mock_circuit.rank = 0
    mock_simulator.local_circuit = mock_circuit
    globals.simulator = mock_simulator

    yield

    # Cleanup
    globals.simulator = None


class TestEligibilitySTDPSparse:
    """Test STDP eligibility traces for sparse connections."""

    @pytest.fixture
    def setup_sparse(self):
        """Create a simple sparse connection for testing."""
        device = torch.device('cpu')
        pre = IFNeurons(n_neurons=10, device=device, delay_max=5)
        pos = IFNeurons(n_neurons=5, device=device, delay_max=5)

        # Create connection: neuron 0->0, 1->1, ..., 4->4
        idx_pre = torch.arange(5, dtype=torch.long)
        idx_pos = torch.arange(5, dtype=torch.long)
        weights = torch.ones(5, dtype=torch.float32) * 0.5
        delays = torch.ones(5, dtype=torch.long)

        spec = ConnectionSpec(
            pre=pre, pos=pos,
            src_idx=idx_pre, tgt_idx=idx_pos,
            weight=weights, delay=delays,
            params={}
        )
        conn = StaticSparse(spec)

        return conn, pre, pos

    def test_initialization(self, setup_sparse):
        """Test that eligibility traces are properly initialized."""
        conn, pre, pos = setup_sparse
        eligibility = EligibilitySTDPSparse(tau_pre=20e-3, tau_post=20e-3, dt=1e-3)
        eligibility.bind(conn)

        assert eligibility.x_pre is not None
        assert eligibility.x_post is not None
        assert eligibility.x_pre.shape == (conn.size,)
        assert eligibility.x_post.shape == (conn.size,)
        assert torch.all(eligibility.x_pre == 0.0)
        assert torch.all(eligibility.x_post == 0.0)

    def test_trace_decay(self, setup_sparse):
        """Test that traces decay exponentially."""
        conn, pre, pos = setup_sparse
        tau = 20e-3
        dt = 1e-3
        eligibility = EligibilitySTDPSparse(tau_pre=tau, tau_post=tau, dt=dt)
        eligibility.bind(conn)

        # Set initial trace values
        eligibility.x_pre[:] = 1.0
        eligibility.x_post[:] = 1.0

        # Expected decay factor
        expected_alpha = torch.exp(torch.tensor(-dt / tau))

        # Step without spikes
        eligibility.step(conn)

        # Check decay
        assert torch.allclose(eligibility.x_pre, expected_alpha * torch.ones(conn.size))
        assert torch.allclose(eligibility.x_post, expected_alpha * torch.ones(conn.size))

    def test_spike_addition(self, setup_sparse):
        """Test that spikes are added to traces."""
        conn, pre, pos = setup_sparse
        eligibility = EligibilitySTDPSparse(tau_pre=20e-3, tau_post=20e-3, dt=1e-3)
        eligibility.bind(conn)

        # Inject presynaptic spikes
        pre_spike_mask = torch.zeros(10, dtype=torch.bool)
        pre_spike_mask[0] = True  # Neuron 0 spikes
        pre.inject_spikes(pre_spike_mask)
        pre._process()  # Update spike buffer at t=0

        # Advance time so the spike at delay=1 is accessible
        globals.simulator.local_circuit.t = 1

        # Step eligibility
        x_pre, x_post = eligibility.step(conn)

        # First synapse (0->0) should have increased x_pre
        assert x_pre[0] > 0.0, f"Expected x_pre[0] > 0, got {x_pre[0]}"
        # Others should remain zero (no spikes)
        assert torch.all(x_pre[1:] == 0.0)


class TestEligibilitySTDPDense:
    """Test STDP eligibility traces for dense connections."""

    @pytest.fixture
    def setup_dense(self):
        """Create a simple dense connection for testing."""
        device = torch.device('cpu')
        pre = IFNeurons(n_neurons=10, device=device, delay_max=5)
        pos = IFNeurons(n_neurons=5, device=device, delay_max=5)

        # Create all-to-all connection
        idx_pre = torch.arange(10, dtype=torch.long).repeat(5)
        idx_pos = torch.arange(5, dtype=torch.long).repeat_interleave(10)
        weights = torch.ones(50, dtype=torch.float32) * 0.5

        spec = ConnectionSpec(
            pre=pre, pos=pos,
            src_idx=idx_pre, tgt_idx=idx_pos,
            weight=weights, delay=1,  # uniform delay
            params={}
        )
        conn = StaticDense(spec)

        return conn, pre, pos

    def test_initialization(self, setup_dense):
        """Test that eligibility traces are per-neuron for dense."""
        conn, pre, pos = setup_dense
        eligibility = EligibilitySTDPDense(tau_pre=20e-3, tau_post=20e-3, dt=1e-3)
        eligibility.bind(conn)

        assert eligibility.x_pre.shape == (pre.size,)
        assert eligibility.x_post.shape == (pos.size,)
        assert torch.all(eligibility.x_pre == 0.0)
        assert torch.all(eligibility.x_post == 0.0)


class TestSignalPostSpikes:
    """Test postsynaptic spike learning signal."""

    @pytest.fixture
    def setup_conn(self):
        """Create a simple connection for testing."""
        device = torch.device('cpu')
        pre = IFNeurons(n_neurons=10, device=device, delay_max=5)
        pos = IFNeurons(n_neurons=5, device=device, delay_max=5)

        idx_pre = torch.arange(5, dtype=torch.long)
        idx_pos = torch.arange(5, dtype=torch.long)
        weights = torch.ones(5, dtype=torch.float32) * 0.5
        delays = torch.ones(5, dtype=torch.long)

        spec = ConnectionSpec(
            pre=pre, pos=pos,
            src_idx=idx_pre, tgt_idx=idx_pos,
            weight=weights, delay=delays,
            params={}
        )
        conn = StaticSparse(spec)

        return conn, pre, pos

    def test_returns_post_spikes(self, setup_conn):
        """Test that signal returns postsynaptic spikes."""
        conn, pre, pos = setup_conn
        signal = SignalPostSpikes()

        # Inject postsynaptic spikes
        pos_spike_mask = torch.zeros(5, dtype=torch.bool)
        pos_spike_mask[2] = True  # Neuron 2 spikes
        pos.inject_spikes(pos_spike_mask)
        pos._process()  # Update spike buffer at t=0

        # Advance time so spike is in the buffer
        globals.simulator.local_circuit.t = 1

        # Get signal
        L_prime = signal.step(conn)

        assert L_prime.shape == (5,)
        assert L_prime[2] == 1.0
        assert torch.all(L_prime[[0, 1, 3, 4]] == 0.0)


class TestUpdateSTDPSparse:
    """Test STDP weight update policy for sparse connections."""

    @pytest.fixture
    def setup_sparse(self):
        """Create a simple sparse connection for testing."""
        device = torch.device('cpu')
        pre = IFNeurons(n_neurons=10, device=device, delay_max=5)
        pos = IFNeurons(n_neurons=5, device=device, delay_max=5)

        idx_pre = torch.arange(5, dtype=torch.long)
        idx_pos = torch.arange(5, dtype=torch.long)
        weights = torch.ones(5, dtype=torch.float32) * 0.5
        delays = torch.ones(5, dtype=torch.long)

        spec = ConnectionSpec(
            pre=pre, pos=pos,
            src_idx=idx_pre, tgt_idx=idx_pos,
            weight=weights, delay=delays,
            params={}
        )
        conn = StaticSparse(spec)

        return conn, pre, pos

    def test_weight_clamping(self, setup_sparse):
        """Test that weights are clamped to [w_min, w_max]."""
        conn, pre, pos = setup_sparse
        update = UpdateSTDPSparse(w_min=0.0, w_max=1.0)

        # Set weights outside bounds
        conn.weight[:] = 2.0

        # Create dummy eligibility and signal
        eligibility = (torch.zeros(5), torch.zeros(5))
        learning_signal = torch.zeros(5)

        # Apply update (no changes, but should clamp)
        update.apply(conn, eligibility, learning_signal)

        assert torch.all(conn.weight <= 1.0)
        assert torch.all(conn.weight >= 0.0)

    def test_potentiation(self, setup_sparse):
        """Test LTP: pre spike followed by post spike."""
        conn, pre, pos = setup_sparse
        update = UpdateSTDPSparse(A_plus=0.1, A_minus=0.0, oja_decay=0.0)

        # Simulate: x_pre = 1.0 (recent pre spike), post spike now
        x_pre = torch.ones(5)
        x_post = torch.zeros(5)
        learning_signal = torch.ones(5)  # All post neurons spike

        initial_weights = conn.weight.clone()
        update.apply(conn, (x_pre, x_post), learning_signal)

        # Weights should increase (potentiation)
        assert torch.all(conn.weight > initial_weights)

    def test_depression(self, setup_sparse):
        """Test LTD: post spike followed by pre spike."""
        conn, pre, pos = setup_sparse
        update = UpdateSTDPSparse(A_plus=0.0, A_minus=-0.1, oja_decay=0.0)

        # Inject pre spikes
        pre_spike_mask = torch.ones(10, dtype=torch.bool)
        pre.inject_spikes(pre_spike_mask)
        pre._process()  # Update spike buffer at t=0

        # Advance time so spikes are accessible
        globals.simulator.local_circuit.t = 1

        # Simulate: x_post = 1.0 (recent post spike), pre spike now
        x_pre = torch.zeros(5)
        x_post = torch.ones(5)
        learning_signal = torch.zeros(5)  # No post spikes now

        initial_weights = conn.weight.clone()
        update.apply(conn, (x_pre, x_post), learning_signal)

        # Weights should decrease (depression)
        assert torch.all(conn.weight < initial_weights)


class TestPlasticityRuleIntegration:
    """Test full PlasticityRule integration."""

    @pytest.fixture
    def setup_sparse(self):
        """Create a sparse connection for testing."""
        device = torch.device('cpu')
        pre = IFNeurons(n_neurons=10, device=device, delay_max=5)
        pos = IFNeurons(n_neurons=5, device=device, delay_max=5)

        idx_pre = torch.arange(5, dtype=torch.long)
        idx_pos = torch.arange(5, dtype=torch.long)
        weights = torch.ones(5, dtype=torch.float32) * 0.5
        delays = torch.ones(5, dtype=torch.long)

        spec = ConnectionSpec(
            pre=pre, pos=pos,
            src_idx=idx_pre, tgt_idx=idx_pos,
            weight=weights, delay=delays,
            params={}
        )
        conn = StaticSparse(spec)

        return conn, pre, pos

    def test_rule_step(self, setup_sparse):
        """Test that PlasticityRule.step() runs without errors."""
        conn, pre, pos = setup_sparse

        rule = PlasticityRule(
            eligibility=EligibilitySTDPSparse(),
            signal=SignalPostSpikes(),
            update=UpdateSTDPSparse()
        )
        rule.init_state(conn)

        # Should run without errors
        rule.step(conn)


class TestFactories:
    """Test factory functions."""

    def test_build_stdp_sparse(self):
        """Test STDP sparse rule factory."""
        config = {
            "name": "stdp",
            "params": {"A_plus": 1e-4, "tau_pre": 20e-3}
        }
        rule = build_rule_for_sparse(config)

        assert isinstance(rule, PlasticityRule)
        assert isinstance(rule.eligibility, EligibilitySTDPSparse)
        assert isinstance(rule.signal, SignalPostSpikes)
        assert isinstance(rule.update, UpdateSTDPSparse)

    def test_build_stdp_dense(self):
        """Test STDP dense rule factory."""
        config = {
            "name": "stdp",
            "params": {"A_minus": -1.2e-4}
        }
        rule = build_rule_for_dense(config)

        assert isinstance(rule, PlasticityRule)
        assert isinstance(rule.eligibility, EligibilitySTDPDense)
        assert isinstance(rule.signal, SignalPostSpikes)
        assert isinstance(rule.update, UpdateSTDPDense)

    def test_unknown_rule(self):
        """Test that unknown rule names raise errors."""
        config = {"name": "unknown_rule"}

        with pytest.raises(ValueError, match="Unknown plasticity rule"):
            build_rule_for_sparse(config)

        with pytest.raises(ValueError, match="Unknown plasticity rule"):
            build_rule_for_dense(config)


class TestMultiGPURestriction:
    """Test multi-GPU restriction validation."""

    def test_sparse_rejects_different_devices(self):
        """Test that PlasticSparse rejects pre/pos on different devices."""
        # This test only runs if multiple GPUs are available
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            pytest.skip("Multiple GPUs not available")

        from neurobridge.plastic_connections import PlasticSparse

        device0 = torch.device('cuda:0')
        device1 = torch.device('cuda:1')

        pre = IFNeurons(n_neurons=10, device=device0, delay_max=5)
        pos = IFNeurons(n_neurons=5, device=device1, delay_max=5)

        idx_pre = torch.arange(5, dtype=torch.long)
        idx_pos = torch.arange(5, dtype=torch.long)
        weights = torch.ones(5, dtype=torch.float32)
        delays = torch.ones(5, dtype=torch.long)

        spec = ConnectionSpec(
            pre=pre, pos=pos,
            src_idx=idx_pre, tgt_idx=idx_pos,
            weight=weights, delay=delays,
            params={"plasticity": {"name": "stdp"}}
        )

        with pytest.raises(RuntimeError, match="same device"):
            PlasticSparse(spec)

    def test_dense_rejects_different_devices(self):
        """Test that PlasticDense rejects pre/pos on different devices."""
        # This test only runs if multiple GPUs are available
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            pytest.skip("Multiple GPUs not available")

        from neurobridge.plastic_connections import PlasticDense

        device0 = torch.device('cuda:0')
        device1 = torch.device('cuda:1')

        pre = IFNeurons(n_neurons=10, device=device0, delay_max=5)
        pos = IFNeurons(n_neurons=5, device=device1, delay_max=5)

        idx_pre = torch.arange(5, dtype=torch.long).repeat(5)
        idx_pos = torch.arange(5, dtype=torch.long).repeat_interleave(5)
        weights = torch.ones(25, dtype=torch.float32)

        spec = ConnectionSpec(
            pre=pre, pos=pos,
            src_idx=idx_pre, tgt_idx=idx_pos,
            weight=weights, delay=1,
            params={"plasticity": {"name": "stdp"}}
        )

        with pytest.raises(RuntimeError, match="same device"):
            PlasticDense(spec)


class TestSTDPIntegration:
    """Integration tests for STDP with realistic spike timings."""

    @pytest.fixture
    def setup_plastic_sparse(self):
        """Create a PlasticSparse connection with STDP."""
        from neurobridge.plastic_connections import PlasticSparse

        device = torch.device('cpu')
        pre = IFNeurons(n_neurons=10, device=device, delay_max=20)
        pos = IFNeurons(n_neurons=5, device=device, delay_max=20)

        # Single connection: neuron 0 -> 0
        idx_pre = torch.tensor([0], dtype=torch.long)
        idx_pos = torch.tensor([0], dtype=torch.long)
        weights = torch.tensor([0.5], dtype=torch.float32)
        delays = torch.tensor([1], dtype=torch.long)

        spec = ConnectionSpec(
            pre=pre, pos=pos,
            src_idx=idx_pre, tgt_idx=idx_pos,
            weight=weights, delay=delays,
            params={
                "plasticity": {
                    "name": "stdp",
                    "params": {
                        "A_plus": 0.01,      # High learning rate for visible changes
                        "A_minus": -0.012,   # Slightly asymmetric
                        "tau_pre": 20e-3,
                        "tau_post": 20e-3,
                        "w_min": 0.0,
                        "w_max": 1.0,
                        "oja_decay": 0.0     # Disable for this test
                    }
                }
            }
        )
        conn = PlasticSparse(spec)
        return conn, pre, pos

    def test_ltp_pre_before_post(self, setup_plastic_sparse):
        """Test Long-Term Potentiation: pre spike before post spike."""
        conn, pre, pos = setup_plastic_sparse
        initial_weight = conn.weight[0].item()

        # Pre spike at t=5, post spike at t=10
        # This should cause potentiation (weight increase)

        # Simulate 5 steps
        for t in range(5):
            globals.simulator.local_circuit.t = t
            conn.rule.step(conn)

        # Inject pre spike at t=5
        pre_spikes = torch.zeros(10, dtype=torch.bool)
        pre_spikes[0] = True
        pre.inject_spikes(pre_spikes)
        pre._process()

        # Simulate 5 more steps (t=5 to t=9)
        for t in range(5, 10):
            globals.simulator.local_circuit.t = t
            conn.rule.step(conn)

        # Inject post spike at t=10
        pos_spikes = torch.zeros(5, dtype=torch.bool)
        pos_spikes[0] = True
        pos.inject_spikes(pos_spikes)
        pos._process()

        # Step once more to trigger plasticity update
        globals.simulator.local_circuit.t = 10
        conn.rule.step(conn)

        final_weight = conn.weight[0].item()

        # Weight should increase (LTP)
        assert final_weight > initial_weight, \
            f"Expected LTP: weight should increase. Initial={initial_weight:.4f}, Final={final_weight:.4f}"

    def test_ltd_post_before_pre(self, setup_plastic_sparse):
        """Test Long-Term Depression: post spike before pre spike."""
        conn, pre, pos = setup_plastic_sparse
        initial_weight = conn.weight[0].item()

        # Post spike at t=5, pre spike at t=10
        # This should cause depression (weight decrease)

        # Simulate 5 steps
        for t in range(5):
            globals.simulator.local_circuit.t = t
            conn.rule.step(conn)

        # Inject post spike at t=5
        pos_spikes = torch.zeros(5, dtype=torch.bool)
        pos_spikes[0] = True
        pos.inject_spikes(pos_spikes)
        pos._process()

        # Simulate 5 more steps (t=5 to t=9)
        for t in range(5, 10):
            globals.simulator.local_circuit.t = t
            conn.rule.step(conn)

        # Inject pre spike at t=10
        pre_spikes = torch.zeros(10, dtype=torch.bool)
        pre_spikes[0] = True
        pre.inject_spikes(pre_spikes)
        pre._process()

        # Step once more to trigger plasticity update
        globals.simulator.local_circuit.t = 10
        conn.rule.step(conn)

        final_weight = conn.weight[0].item()

        # Weight should decrease (LTD)
        assert final_weight < initial_weight, \
            f"Expected LTD: weight should decrease. Initial={initial_weight:.4f}, Final={final_weight:.4f}"

    def test_weight_clamping_integration(self, setup_plastic_sparse):
        """Test that weights are clamped to [w_min, w_max] during simulation."""
        conn, pre, pos = setup_plastic_sparse

        # Test upper bound clamping
        # Manually set weight above maximum
        conn.weight[0] = 1.5

        # Run one plasticity step (should clamp)
        globals.simulator.local_circuit.t = 0
        conn.rule.step(conn)

        # Weight should be clamped at w_max=1.0
        assert conn.weight[0] <= 1.0, f"Weight exceeds w_max: {conn.weight[0]}"

        # Test lower bound clamping
        # Manually set weight below minimum
        conn.weight[0] = -0.5

        # Run one plasticity step (should clamp)
        globals.simulator.local_circuit.t = 1
        conn.rule.step(conn)

        # Weight should be clamped at w_min=0.0
        assert conn.weight[0] >= 0.0, f"Weight below w_min: {conn.weight[0]}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
