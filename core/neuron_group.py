from abc import ABC, abstractmethod
import torch


class NeuronGroup(ABC):

    def __init__(self, size: int, delay_max: int, device: str):
        self.size = size
        self.delay_max = delay_max
        self.device = torch.device(device)
        self.spike_buffer = torch.zeros((delay_max, size), dtype=torch.bool, device=self.device)
        self.t = 0


    def _register_spikes(self, spikes: torch.Tensor):
        self.spike_buffer[self.t % self.delay_max] = spikes.bool()


    def get_spikes_at(self, delay_steps: torch.Tensor, neuron_indices: torch.Tensor) -> torch.Tensor:
        time_indices = (self.t - delay_steps) % self.delay_max
        return self.spike_buffer[time_indices, neuron_indices]


    @abstractmethod
    def step(self): pass


    def reset(self):
        self.spike_buffer.zero_()
        self.t = 0

