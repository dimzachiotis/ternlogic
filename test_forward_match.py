import torch
import sys
import os

top_level_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if top_level_dir not in sys.path:
    sys.path.insert(0, top_level_dir)

from difflogic.difflogic import LogicLayer, GroupSum

# ── params (match your usual run) ──────────────────────────────
in_dim      = 784
out_dim     = 800
num_layers  = 4
num_classes = 10
tau         = 30.0
seed        = 0
# ───────────────────────────────────────────────────────────────

torch.manual_seed(seed)

# Build CUDA model
def make_model(device, implementation):
    layers = [torch.nn.Flatten()]
    layers.append(LogicLayer(in_dim=in_dim, out_dim=out_dim, device=device, implementation=implementation))
    for _ in range(num_layers - 1):
        layers.append(LogicLayer(in_dim=out_dim, out_dim=out_dim, device=device, implementation=implementation))
    layers.append(GroupSum(k=num_classes, tau=tau, device=device))
    model = torch.nn.Sequential(*layers)
    return model.to(device)

# Build both models with the SAME seed so weights are identical
torch.manual_seed(seed)
model_cuda = make_model('cuda', 'cuda')

torch.manual_seed(seed)
model_python = make_model('cpu', 'python')

# Now copy weights from CUDA model → Python model (move to CPU)
cuda_layers   = [l for l in model_cuda   if isinstance(l, LogicLayer)]
python_layers = [l for l in model_python if isinstance(l, LogicLayer)]

for lc, lp in zip(cuda_layers, python_layers):
    lp.weights.data = lc.weights.data.cpu()
    lp.indices = (lc.indices[0].cpu(), lc.indices[1].cpu())

# Small test batch
torch.manual_seed(42)
x_test = torch.randn(8, in_dim)

# Ternary quantize exactly as your eval() function does
x_tern = torch.where(x_test < -0.5, -1.0, torch.where(x_test > 0.5, 1.0, 0.0)).float()

# ── Intermediate layer output test ────────────────────────────
model_cuda.eval()
model_python.eval()

x_cuda   = x_tern.cuda()
x_python = x_tern.clone()

print("\n--- Intermediate layer comparison ---")
for i, (lc, lp) in enumerate(zip(
    [l for l in model_cuda   if isinstance(l, LogicLayer)],
    [l for l in model_python if isinstance(l, LogicLayer)]
)):
    with torch.no_grad():
        x_cuda   = lc(x_cuda)
        x_python = lp(x_python)
        diff = (x_cuda.cpu() - x_python).abs().max().item()
        print(f"Layer {i} output max diff: {diff}")

# # ──────────────────────────────
# model_cuda.eval()
# model_python.eval()

# with torch.no_grad():
#     out_cuda   = model_cuda(x_tern.cuda()).cpu()
#     out_python = model_python(x_tern)

# print("CUDA   output:", out_cuda[0])
# print("Python output:", out_python[0])
# print("Max diff:", (out_cuda - out_python).abs().max().item())
# print("Outputs match:", torch.allclose(out_cuda, out_python.float(), atol=1e-4))

# ── Gradient test ──────────────────────────────────────────────
torch.manual_seed(seed)
model_cuda2 = make_model('cuda', 'cuda')
torch.manual_seed(seed)
model_python2 = make_model('cpu', 'python')

# Copy weights
cuda_layers2   = [l for l in model_cuda2   if isinstance(l, LogicLayer)]
python_layers2 = [l for l in model_python2 if isinstance(l, LogicLayer)]
for lc, lp in zip(cuda_layers2, python_layers2):
    lp.weights.data = lc.weights.data.cpu()
    lp.indices = (lc.indices[0].cpu(), lc.indices[1].cpu())

model_cuda2.train()
model_python2.train()

torch.manual_seed(42)
x_test2 = torch.randn(8, in_dim)
x_tern2 = torch.where(x_test2 < -0.5, -1.0, torch.where(x_test2 > 0.5, 1.0, 0.0)).float()
y_test  = torch.randint(0, num_classes, (8,))

loss_fn = torch.nn.CrossEntropyLoss()

# CUDA backward
out_cuda2 = model_cuda2(x_tern2.cuda())
loss_cuda = loss_fn(out_cuda2, y_test.cuda())
loss_cuda.backward()

# Python backward
out_python2 = model_python2(x_tern2)
loss_python = loss_fn(out_python2.float(), y_test)
loss_python.backward()

print("\n--- Gradient comparison ---")
for i, (lc, lp) in enumerate(zip(cuda_layers2, python_layers2)):
    diff = (lc.weights.grad.cpu() - lp.weights.grad).abs().max().item()
    print(f"Layer {i} weight grad max diff: {diff}")