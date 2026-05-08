import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
import matplotlib.pyplot as plt
import os

# --- Configuration ---
output_dir = "/home/dzachiotis/thesis/graphs/"
filename = "best_architecture_lr_001.png"
target_lr = 0.01

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

full_path = os.path.join(output_dir, filename)

mlflow.set_tracking_uri("sqlite:////home/dzachiotis/thesis/mlflow_shared/mlflow.db")
client = MlflowClient()

# --- Data Extraction ---
experiment = mlflow.get_experiment_by_name("ternary search lr")
runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="attributes.status = 'FINISHED'"
)

records = []
for run_id in runs['run_id']:
    run = client.get_run(run_id)
    params = run.data.params
    
    records.append({
        'k':         int(params.get('num_neurons')),
        'l':         int(params.get('num_layers')),
        'seed':      int(params.get('seed')),
        'lr':        float(params.get('learning_rate')),
        'kl':        int(params.get('total_neurons')),
        'test_acc':  run.data.metrics.get('64_tern_testing_acc_3')
    })

df = pd.DataFrame(records)

# --- Logic Processing ---

# 1. Filter for the specific learning rate
df_filtered = df[df['lr'] == target_lr].copy()

# 2. Average over seeds for each specific architecture (k, l)
summary = df_filtered.groupby(['k', 'l', 'kl'])['test_acc'].agg(['mean', 'std']).reset_index()
summary.columns = ['k', 'l', 'kl', 'mean_acc', 'std_acc']

# 3. Find the BEST architecture (k, l) for each unique total neuron budget (kl)
# This handles cases where different (k, l) pairs result in the same total kl
idx = summary.groupby('kl')['mean_acc'].idxmax()
best_points = summary.loc[idx].sort_values('kl')

# --- Visualization ---
fig, ax = plt.subplots(figsize=(12, 8))

# Plot the "Frontier" line
ax.plot(
    best_points['kl'], 
    best_points['mean_acc'], 
    marker='o', 
    linestyle='-', 
    color='#2c3e50', 
    linewidth=2, 
    markersize=10, 
    label=f'Best Arch (LR={target_lr})'
)

# Add error bars (standard deviation across seeds)
ax.errorbar(
    best_points['kl'], 
    best_points['mean_acc'], 
    yerr=best_points['std_acc'], 
    fmt='none', 
    ecolor='gray', 
    alpha=0.5, 
    capsize=4
)

# Annotate each point with its specific (k, l) configuration
for _, row in best_points.iterrows():
    label = f"k={int(row['k'])}, l={int(row['l'])}"
    ax.annotate(
        label, 
        (row['kl'], row['mean_acc']),
        textcoords="offset points", 
        xytext=(0, 15), 
        ha='center', 
        fontsize=10,
        fontweight='bold',
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", alpha=0.8)
    )

# Formatting
ax.set_xscale('log')
ax.set_xlim(best_points['kl'].min() * 0.8, best_points['kl'].max() * 1.2)
ax.set_xlabel('Total Neurons Budget ($k \\times l$)', fontsize=12)
ax.set_ylabel('Mean Test Accuracy (Average over Seeds)', fontsize=12)
ax.set_title(f'Performance Frontier: Best Architecture per Budget\n(Learning Rate = {target_lr})', fontsize=15, pad=20)
ax.grid(True, linestyle='--', alpha=0.6, which='both')

plt.tight_layout()
plt.savefig(full_path, dpi=200)
plt.show()

print("Optimized Architectures for LR 0.01:")
print(best_points[['kl', 'k', 'l', 'mean_acc']].to_string(index=False))