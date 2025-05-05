import unittest
import torch

from neurobridge import *
from neurobridge.core import ParentStack, ConnectionOperator
from neurobridge.group import Group, SpatialGroup
from neurobridge.engine import LocalCircuit


class TestCore(unittest.TestCase):
    """Tests for core functionality of NeuroBridge."""

    def test_node_hierarchy(self):
        """Test node parent-child relationships."""
        parent = Node()
        child1 = Node()
        child2 = Node()

        parent.add_child(child1)
        parent.add_child(child2)

        self.assertEqual(len(parent.children), 2)
        self.assertIn(child1, parent.children)
        self.assertIn(child2, parent.children)
        self.assertEqual(child1.parent, parent)
        self.assertEqual(child2.parent, parent)

        # Test that a node can only have one parent
        new_parent = Node()
        new_parent.add_child(child1)

        self.assertEqual(len(parent.children), 1)
        self.assertNotIn(child1, parent.children)
        self.assertIn(child2, parent.children)
        self.assertEqual(child1.parent, new_parent)

    def test_parent_stack(self):
        """Test the ParentStack context manager."""
        parent = Node()

        with ParentStack(parent):
            child = Node()

        self.assertIn(child, parent.children)
        self.assertEqual(child.parent, parent)

        # Test nested parent stack
        outer_parent = Node()
        with ParentStack(outer_parent):
            middle_child = Node()

            with ParentStack(middle_child):
                inner_child = Node()

        self.assertIn(middle_child, outer_parent.children)
        self.assertIn(inner_child, middle_child.children)

    def test_process_call_hierarchy(self):
        """Test that _process() is called in the correct order."""

        class TestNode(Node):
            def __init__(self):
                super().__init__()
                self.process_calls = []

            def _process(self):
                self.process_calls.append(len(self.process_calls))

        parent = TestNode()
        child1 = TestNode()
        child2 = TestNode()

        parent.add_child(child1)
        parent.add_child(child2)

        parent._call_process()

        # Children should be processed before parents
        self.assertEqual(child1.process_calls, [0])
        self.assertEqual(child2.process_calls, [0])
        self.assertEqual(parent.process_calls, [0])

        parent._call_process()

        # Verify process is called again on next call
        self.assertEqual(child1.process_calls, [0, 1])
        self.assertEqual(child2.process_calls, [0, 1])
        self.assertEqual(parent.process_calls, [0, 1])


class TestGroup(unittest.TestCase):
    """Tests for the Group class and its derivatives."""

    def setUp(self):
        """Set up common test resources."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_group_creation(self):
        """Test Group initialization."""
        size = 100
        group = Group(size=size, device=self.device)

        self.assertEqual(group.size, size)
        self.assertEqual(group.filter.shape, (size,))
        self.assertEqual(group.filter.dtype, torch.bool)
        self.assertTrue(
            torch.all(group.filter)
        )  # All elements should be selected initially

    def test_filter_operations(self):
        """Test filtering operations on Group."""
        size = 100
        group = Group(size=size, device=self.device)

        # Filter even indices
        filtered = group.where_id(lambda ids: ids % 2 == 0)

        # Original group should not be modified
        self.assertTrue(torch.all(group.filter))

        # Filtered group should have half the elements selected
        self.assertEqual(filtered.filter.sum().item(), size // 2)
        self.assertTrue(torch.all(filtered.filter[::2]))  # Even indices should be True
        self.assertFalse(
            torch.any(filtered.filter[1::2])
        )  # Odd indices should be False

        # Test invalid filter function
        with self.assertRaises(ValueError):
            group.where_id(lambda ids: ids)  # Not returning a boolean mask

        # Test reset_filter
        filtered.reset_filter()
        self.assertTrue(torch.all(filtered.filter))

    def test_spatial_group(self):
        """Test SpatialGroup initialization and operations."""
        size = 100
        spatial_dims = 3
        group = SpatialGroup(size=size, spatial_dimensions=spatial_dims, device=self.device)

        self.assertEqual(group.spatial_dimensions.item(), spatial_dims)
        self.assertEqual(group.positions.shape, (size, spatial_dims))

        # Test position-based filtering
        # Select positions with positive z-coordinate
        filtered = group.where_pos(lambda pos: pos[:, 2] > 0)

        # Approximately half should be selected (normal distribution around 0)
        self.assertGreater(filtered.filter.sum().item(), 0)
        self.assertLess(filtered.filter.sum().item(), size)

        # Test combined filtering
        combined = filtered.where_id(lambda ids: ids < size // 2)
        self.assertLessEqual(combined.filter.sum().item(), filtered.filter.sum().item())
        self.assertLessEqual(combined.filter.sum().item(), size // 2)


class TestNeurons(unittest.TestCase):
    """Tests for neuron models."""

    def setUp(self):
        """Set up common test resources."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def test_neuron_group_base(self):
        """Test NeuronGroup initialization and methods."""
        n_neurons = 50
        delay_max = 15
        group = NeuronGroup(n_neurons=n_neurons, delay_max=delay_max, device=self.device)

        self.assertEqual(group.delay_max.item(), delay_max)
        self.assertEqual(group._spike_buffer.shape, (n_neurons, delay_max))
        self.assertEqual(group._input_currents.shape, (n_neurons,1))
        self.assertEqual(group._input_spikes.shape, (n_neurons,))

        # Test get_spike_buffer
        buffer = group.get_spike_buffer()
        self.assertIs(buffer, group._spike_buffer)

        # Test inject_currents
        currents = torch.rand(n_neurons, device=self.device)
        group.inject_currents(currents)
        self.assertTrue(torch.all(group._input_currents.squeeze() == currents))

        # Test inject_spikes
        spikes = torch.zeros(n_neurons, device=self.device, dtype=torch.bool)
        spikes[::5] = True  # Every 5th neuron spikes
        group.inject_spikes(spikes)
        self.assertTrue(torch.all(group._input_spikes[::5]))
        self.assertFalse(torch.any(group._input_spikes[1::5]))

    def test_parrot_neurons(self):
        """Test ParrotGroup behavior."""
        n_neurons = 50
        group = ParrotNeurons(n_neurons=n_neurons, device=self.device)

        # Create dummy simulator engine for time reference
        class DummyEngine(SimulatorEngine):
            def build_user_network(self, rank, world_size, device):
                pass

        # Create an engine instance but don't fully initialize
        # We just need the time variable
        SimulatorEngine.engine = DummyEngine()
        SimulatorEngine.engine.local_circuit = LocalCircuit(self.device)
        SimulatorEngine.engine.local_circuit.t = torch.zeros(
            1, dtype=torch.long, device=self.device
        )

        # Inject spikes and see if they appear in the buffer
        spikes = torch.zeros(n_neurons, device=self.device, dtype=torch.bool)
        spikes[10] = True  # Neuron 10 spikes
        group.inject_spikes(spikes)

        # Process the group
        group._process()

        # Check if neuron 10 has a spike at t=0
        t_idx = SimulatorEngine.engine.local_circuit.t % group.delay_max
        self.assertTrue(group._spike_buffer[10, t_idx])
        self.assertFalse(torch.any(group._spike_buffer[0:10, t_idx]))
        self.assertFalse(torch.any(group._spike_buffer[11:, t_idx]))

        # Test with currents
        group._input_spikes.fill_(False)
        currents = torch.zeros(n_neurons, device=self.device)
        currents[20] = 1.0  # Positive current to neuron 20
        group.inject_currents(currents)

        # Advance time and process
        SimulatorEngine.engine.local_circuit.t += 1
        group._process()

        # Check if neuron 20 has a spike at the new time
        t_idx = SimulatorEngine.engine.local_circuit.t % group.delay_max
        self.assertTrue(group._spike_buffer[20, t_idx])

    def test_if_neurons(self):
        """Test IFNeuronGroup behavior."""
        n_neurons = 50
        threshold = 0.8
        group = SimpleIFNeurons(n_neurons=n_neurons, threshold=threshold, device=self.device)

        # Create dummy simulator engine for time reference
        class DummyEngine(SimulatorEngine):
            def build_user_network(self, rank, world_size, device):
                pass

        # Create an engine instance but don't fully initialize
        # We just need the time variable
        SimulatorEngine.engine = DummyEngine()
        SimulatorEngine.engine.local_circuit = LocalCircuit(self.device)
        SimulatorEngine.engine.local_circuit.t = torch.zeros(
            1, dtype=torch.long, device=self.device
        )

        # Set initial values to verify behavior
        group.V.fill_(0.0)

        # Inject sub-threshold current
        currents = torch.zeros(n_neurons, device=self.device)
        currents[10] = 0.5  # Below threshold
        group.inject_currents(currents)

        # Process the group
        group._process()

        # Check that no spike was generated but potential increased
        t_idx = SimulatorEngine.engine.local_circuit.t % group.delay_max
        self.assertFalse(group._spike_buffer[10, t_idx])
        self.assertGreater(group.V[10].item(), 0.0)

        # Inject above-threshold current
        currents = torch.zeros(n_neurons, device=self.device)
        currents[20] = 1.0  # Above threshold
        group.inject_currents(currents)

        # Advance time and process
        SimulatorEngine.engine.local_circuit.t += 1
        group._process()

        # Check if neuron 20 spiked and was reset
        t_idx = SimulatorEngine.engine.local_circuit.t % group.delay_max
        self.assertTrue(group._spike_buffer[20, t_idx])
        self.assertEqual(group.V[20].item(), 0.0)  # Voltage should be reset


class TestSynapses(unittest.TestCase):
    """Tests for synaptic connections."""

    def setUp(self):
        """Set up common test resources."""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Create dummy simulator engine for time reference
        class DummyEngine(SimulatorEngine):
            def build_user_network(self, rank, world_size, device):
                pass

        # Create an engine instance but don't fully initialize
        # We just need the time variable
        SimulatorEngine.engine = DummyEngine()
        SimulatorEngine.engine.local_circuit = LocalCircuit(self.device)
        SimulatorEngine.engine.local_circuit.t = torch.zeros(
            1, dtype=torch.long, device=self.device
        )

        # Create source and target neuron groups
        self.source = ParrotNeurons(10, device=self.device)
        self.target = SimpleIFNeurons(5, device=self.device)

    def test_connection_operator(self):
        """Test the ConnectionOperator class and >> operator."""
        conn_op = self.source >> self.target

        self.assertIsInstance(conn_op, ConnectionOperator)
        self.assertIs(conn_op.pre, self.source)
        self.assertIs(conn_op.pos, self.target)

        # Test all-to-all connection
        synapses = conn_op(pattern="all-to-all", weight=0.5, delay=1)

        self.assertIsInstance(synapses, StaticConnection)
        self.assertEqual(synapses.size, 10 * 5)  # 10 source neurons * 5 target neurons
        self.assertTrue(torch.all(synapses.weight == 0.5))
        self.assertTrue(torch.all(synapses.delay == 1))

        # Test one-to-one connection
        source2 = ParrotNeurons(5, device=self.device)  # Same size as target
        synapses = (source2 >> self.target)(pattern="one-to-one", weight=0.2)

        self.assertEqual(synapses.size, 5)  # 5 connections for one-to-one
        self.assertTrue(torch.all(synapses.weight == 0.2))

        # Test specific connection
        idx_pre = torch.tensor([0, 2, 4], device=self.device)
        idx_pos = torch.tensor([1, 3, 4], device=self.device)
        synapses = (self.source >> self.target)(
            pattern="specific", idx_pre=idx_pre, idx_pos=idx_pos, weight=0.3
        )

        self.assertEqual(synapses.size, 3)  # 3 specific connections
        self.assertTrue(torch.all(synapses.idx_pre == idx_pre))
        self.assertTrue(torch.all(synapses.idx_pos == idx_pos))

        # Test function-based weight
        synapses = (self.source >> self.target)(
            pattern="all-to-all",
            weight=lambda pre, post: torch.ones_like(pre, dtype=torch.float32) * 0.1,
        )

        self.assertEqual(synapses.size, 10 * 5)
        self.assertTrue(torch.all(synapses.weight == 0.1))

    def test_static_synapse(self):
        """Test StaticSynapse behavior."""
        # Create connection with source neurons 0-4 connected to target neurons 0-4
        idx_pre = torch.arange(5, device=self.device)
        idx_pos = torch.arange(5, device=self.device)
        weight = torch.ones(5, device=self.device) * 0.5
        delay = torch.zeros(5, device=self.device, dtype=torch.long)

        synapses = StaticConnection(self.source, self.target)
        synapses._establish_connection(
            "specific",
            idx_pre=idx_pre,
            idx_pos=idx_pos,
            weight=weight,
            delay=delay
        )
        synapses._init_connection()

        # Make neuron 0 in source spike
        spikes = torch.zeros(10, device=self.device, dtype=torch.bool)
        spikes[0] = True
        self.source.inject_spikes(spikes)

        # Set up the source spike buffer
        self.source._process()

        # Process the synapses
        synapses._process()

        # Check that target neuron 0 received input current
        self.assertEqual(self.target._input_currents[0].item(), 0.5)
        self.assertEqual(self.target._input_currents[1:].sum().item(), 0.0)

    def test_stdp_synapse(self):
        """Test STDPSynapse behavior."""
        # Create connection with source neurons 0-4 connected to target neurons 0-4
        idx_pre = torch.arange(5, device=self.device)
        idx_pos = torch.arange(5, device=self.device)
        weight = torch.ones(5, device=self.device) * 0.5
        delay = torch.zeros(5, device=self.device, dtype=torch.long)

        synapses = STDPConnection(self.source, self.target)
        synapses._establish_connection(
            "specific",
            idx_pre=idx_pre,
            idx_pos=idx_pos,
            weight=weight,
            delay=delay,
        )
        synapses._init_connection(
            A_plus=0.1,
            A_minus=0.12,
            w_max=1.0,
        )

        # Initial weights should be as specified
        self.assertTrue(torch.all(synapses.weight == 0.5))

        # Make source neuron 0 spike
        spikes = torch.zeros(10, device=self.device, dtype=torch.bool)
        spikes[0] = True
        self.source.inject_spikes(spikes)

        # Set up the source spike buffer
        self.source._process()

        # Process the synapses - this should update x_pre
        synapses._process()

        # x_pre for connection 0 should be increased
        self.assertGreater(synapses.x_pre[0].item(), 0.0)
        self.assertEqual(synapses.x_pre[1:].sum().item(), 0.0)

        # Original weight should be unchanged (no post-synaptic spike yet)
        self.assertEqual(synapses.weight[0].item(), 0.5)

        # Now make target neuron 0 spike (should cause potentiation)
        spikes = torch.zeros(5, device=self.device, dtype=torch.bool)
        spikes[0] = True
        self.target.inject_spikes(spikes)

        # Set up the target spike buffer
        self.target._process()

        # Advance time and process synapses again
        SimulatorEngine.engine.local_circuit.t += 1
        synapses._process()

        # Weight should have increased due to STDP potentiation
        self.assertGreater(synapses.weight[0].item(), 0.5)

        # x_pos for connection 0 should be increased
        self.assertGreater(synapses.x_pos[0].item(), 0.0)


if __name__ == "__main__":
    unittest.main()
