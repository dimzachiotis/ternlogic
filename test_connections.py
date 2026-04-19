import torch
import sys
import os

top_level_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if top_level_dir not in sys.path:
    sys.path.insert(0, top_level_dir)

from difflogic.difflogic import LogicLayer, GroupSum

in_dim      = 784
out_dim     = 800
num_layers  = 4
num_classes = 10
tau         = 30.0
seed        = 0

def make_model(device, implementation):
    layers = [torch.nn.Flatten()]
    layers.append(LogicLayer(in_dim=in_dim, out_dim=out_dim, device=device, implementation=implementation))
    for _ in range(num_layers - 1):
        layers.append(LogicLayer(in_dim=out_dim, out_dim=out_dim, device=device, implementation=implementation))
    layers.append(GroupSum(k=num_classes, tau=tau, device=device))
    return torch.nn.Sequential(*layers).to(device)

torch.manual_seed(seed)
model_cuda = make_model('cuda', 'cuda')

torch.manual_seed(seed)
model_python = make_model('cpu', 'python')

cuda_layers   = [l for l in model_cuda   if isinstance(l, LogicLayer)]
python_layers = [l for l in model_python if isinstance(l, LogicLayer)]

print("=" * 60)
print("CONNECTION IDENTITY TEST")
print("=" * 60)

all_same = True
for i, (lc, lp) in enumerate(zip(cuda_layers, python_layers)):
    idx_diff_0  = (lc.indices[0].cpu() - lp.indices[0]).abs().max().item()
    idx_diff_1  = (lc.indices[1].cpu() - lp.indices[1]).abs().max().item()
    weight_diff = (lc.weights.data.cpu() - lp.weights.data).abs().max().item()
    match = idx_diff_0 == 0 and idx_diff_1 == 0
    if not match:
        all_same = False
    print(f"Layer {i}: index_a diff={idx_diff_0:.2e}  index_b diff={idx_diff_1:.2e}  weights diff={weight_diff:.2e}  {'✓' if match else '✗ MISMATCH'}")

print()
if all_same:
    print("All connections identical between CUDA and CPU ✓")
else:
    print("Connections differ between CUDA and CPU ✗")
    print("→ Fix: re-seed between model builds or generate connections on CPU always.")
print("=" * 60)