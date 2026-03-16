#This file, is only used when cuda --implementation
import torch
import numpy as np
#import ternlogic_cuda  # your compiled CUDA extension

class PackTernaryTensor:
    def __init__(self, t: torch.LongTensor, num_gates: int, device='cuda'):
        """
        t: [neurons, batch] tensor of gate indices (0..num_gates-1)
        num_gates: maximum number of possible gates per neuron
        """
        assert len(t.shape) == 2, t.shape
        self.device = device
        self.num_gates = num_gates

        # Compute how many bits we need per neuron
        self.bits_per_neuron = int(np.ceil(np.log2(num_gates)))
        self.bit_count = 32  # fixed CUDA word size
        self.neurons_per_int = self.bit_count // self.bits_per_neuron

        # Pad neurons so they fit into integers
        total_neurons = t.size(0)
        pad_neurons = (-total_neurons) % self.neurons_per_int
        if pad_neurons > 0:
            t = torch.cat([t, torch.zeros((pad_neurons, t.size(1)), dtype=t.dtype)], dim=0)
        self.pad_neurons = pad_neurons

        if device == 'cuda':
            t = t.to(device).T.contiguous()  # transpose to [batch, neurons]
            # Calls your CUDA extension: returns packed uint32 tensor + pad length
            self.t, self.pad_len = ternlogic_cuda.tensor_packtern_cuda(t)
        else:
            raise NotImplementedError(device)

    def group_sum(self, k):
        """
        Groups neurons in chunks of size k and sums them.
        """
        assert self.device == 'cuda'
        # Pass original neuron count including padding
        neurons = self.t.size(0) * self.neurons_per_int - self.pad_neurons
        batch = self.t.size(0)  # because we transposed to [batch, neurons]
        return ternlogic_cuda.groupternsum(self.t, self.pad_len, k, neurons, batch)

    def flatten(self, start_dim=0, end_dim=-1, **kwargs):
        return self

    # For readable printing
    def _get_member_repr(self, member):
        if len(member) <= 4:
            return ' '.join([np.binary_repr(int(x), width=self.bit_count)[::-1] for x in member])
        first_three = [np.binary_repr(int(x), width=self.bit_count)[::-1] for x in member[:3]]
        sep = "..."
        last = np.binary_repr(int(member[-1]), width=self.bit_count)[::-1]
        return f"{' '.join(first_three)} {sep} {last}"

    def __repr__(self):
        return '\n'.join([self._get_member_repr(item) for item in self.t])