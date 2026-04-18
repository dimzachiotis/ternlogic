import torch
import sys
import os
import itertools

top_level_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if top_level_dir not in sys.path:
    sys.path.insert(0, top_level_dir)

from difflogic.difflogic import LogicLayer, GroupSum
from difflogic.functional import tern_op

number_of_gates = 13

# All 9 possible ternary input combinations
tern_vals = [-1.0, 0.0, 1.0]
all_inputs = list(itertools.product(tern_vals, tern_vals))
# shape: [9, 2] — each row is (a, b)
x_test = torch.tensor(all_inputs, dtype=torch.float32)  # [9, 2]

print("Input combinations (a, b):")
print(x_test)
print()

# ── Build one layer per gate ───────────────────────────────────
# Each neuron is hardwired to one gate by setting its weight
# very high for that gate index and low for all others

def make_single_layer(device, implementation):
    """
    Creates a LogicLayer with number_of_gates neurons.
    Neuron i is hardwired to gate i by setting weights[i, i] = 10.0
    and all other weights to -10.0.
    Indices: neuron i takes input a=0, b=1 (first two features).
    """
    layer = LogicLayer(
        in_dim=2,
        out_dim=number_of_gates,
        device=device,
        implementation=implementation,
        connections='random'
    )

    # Force each neuron i to select gate i
    with torch.no_grad():
        layer.weights.data = torch.full(
            (number_of_gates, number_of_gates), -10.0, device=device
        )
        for i in range(number_of_gates):
            layer.weights.data[i, i] = 10.0

    # Force all neurons to use inputs 0 and 1
    layer.indices = (
        torch.zeros(number_of_gates, dtype=torch.int64, device=device),  # a index
        torch.ones(number_of_gates,  dtype=torch.int64, device=device),  # b index
    )

    # Rebuild CUDA backward indices since we changed layer.indices
    if implementation == 'cuda':
        import numpy as np
        given_x_indices_of_y = [[] for _ in range(2)]
        for y in range(number_of_gates):
            given_x_indices_of_y[0].append(y)  # all neurons use input 0
            given_x_indices_of_y[1].append(y)  # all neurons use input 1
        layer.given_x_indices_of_y_start = torch.tensor(
            np.array([0] + [len(g) for g in given_x_indices_of_y]).cumsum(),
            device=device, dtype=torch.int64
        )
        layer.given_x_indices_of_y = torch.tensor(
            [item for sublist in given_x_indices_of_y for item in sublist],
            dtype=torch.int64, device=device
        )

    return layer

# ── Forward pass test ─────────────────────────────────────────
print("=" * 60)
print("FORWARD PASS TEST")
print("=" * 60)

layer_cuda   = make_single_layer('cuda', 'cuda')
layer_python = make_single_layer('cpu',  'python')

# Copy weights to ensure identical initialization
layer_python.weights.data = layer_cuda.weights.data.cpu()

layer_cuda.eval()
layer_python.eval()

x_cuda   = x_test.cuda()
x_python = x_test.clone()

with torch.no_grad():
    out_cuda   = layer_cuda(x_cuda).cpu()    # [9, 13]
    out_python = layer_python(x_python)      # [9, 13]

print(f"Max forward diff: {(out_cuda - out_python).abs().max().item():.2e}")
print()

# Check each gate output against ground truth
print(f"{'Gate':<6} {'Inputs':<20} {'CUDA':>8} {'Python':>8} {'Expected':>10} {'Match':>6}")
print("-" * 65)
all_match = True
for gate_i in range(number_of_gates):
    for row_i, (a_val, b_val) in enumerate(all_inputs):
        a = torch.tensor([a_val])
        b = torch.tensor([b_val])
        expected = tern_op(a, b, gate_i).item()
        cuda_val   = out_cuda[row_i, gate_i].item()
        python_val = out_python[row_i, gate_i].item()
        match = abs(cuda_val - expected) < 1e-4 and abs(python_val - expected) < 1e-4
        if not match:
            all_match = False
            print(f"  {gate_i:<4} a={a_val:+.0f} b={b_val:+.0f}          {cuda_val:>8.4f} {python_val:>8.4f} {expected:>10.4f}  ✗ MISMATCH")

if all_match:
    print("  All gates match expected values ✓")

# ── Gradient test ─────────────────────────────────────────────
print()
print("=" * 60)
print("GRADIENT TEST")
print("=" * 60)

layer_cuda2   = make_single_layer('cuda', 'cuda')
layer_python2 = make_single_layer('cpu',  'python')
layer_python2.weights.data = layer_cuda2.weights.data.cpu()

layer_cuda2.train()
layer_python2.train()

x_cuda2   = x_test.cuda().requires_grad_(True)
x_python2 = x_test.clone().requires_grad_(True)

out_cuda2   = layer_cuda2(x_cuda2)
out_python2 = layer_python2(x_python2)

# Use sum as scalar loss
loss_cuda   = out_cuda2.sum()
loss_python = out_python2.sum()

loss_cuda.backward()
loss_python.backward()

weight_grad_diff = (layer_cuda2.weights.grad.cpu() - layer_python2.weights.grad).abs().max().item()
input_grad_diff  = (x_cuda2.grad.cpu() - x_python2.grad).abs().max().item()

print(f"Weight gradient max diff: {weight_grad_diff:.2e}")
print(f"Input  gradient max diff: {input_grad_diff:.2e}")

print()
print("=" * 60)
if all_match and weight_grad_diff < 1e-4 and input_grad_diff < 1e-4:
    print("ALL TESTS PASSED ✓")
else:
    print("SOME TESTS FAILED ✗ — check mismatches above")
print("=" * 60)