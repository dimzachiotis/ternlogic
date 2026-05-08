import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
import matplotlib.pyplot as plt
import os

# --- Configuration ---
output_dir = "/home/dzachiotis/thesis/graphs/"
filename = "ternary_vs_binary_comparison.png"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

mlflow.set_tracking_uri("sqlite:////home/dzachiotis/thesis/mlflow_shared/mlflow.db")
client = MlflowClient()

def get_best_frontier(exp_name, metric_name, target_lr=None):
    """Fetches experiment data, averages seeds, and picks best arch per budget."""
    experiment = mlflow.get_experiment_by_name(exp_name)
    runs = mlflow.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="attributes.status = 'FINISHED'"
    )
    
    records = []
    for _, run_row in runs.iterrows():
        run = client.get_run(run_row['run_id'])
        p = run.data.params
        m = run.data.metrics
        
        # Calculate total neurons (kl)
        k = int(p.get('num_neurons', 0))
        l = int(p.get('num_layers', 0))
        
        records.append({
            'k': k, 'l': l, 'kl': k * l,
            'lr': float(p.get('learning_rate', 0)),
            'test_acc': m.get(metric_name)
        })
    
    df = pd.DataFrame(records)
    
    # Filter by LR if specified (for the ternary experiment)
    if target_lr is not None:
        df = df[df['lr'] == target_lr]
    
    # Average over seeds
    summary = df.groupby(['k', 'l', 'kl'])['test_acc'].agg(['mean', 'std']).reset_index()
    
    # Pick best (k, l) pair for each total neuron budget (kl)
    idx = summary.groupby('kl')['mean'].idxmax()
    return summary.loc[idx].sort_values('kl')

# --- 1. Process Data ---
# Ternary: Only LR 0.01
frontier_tern = get_best_frontier("ternary search lr", "64_tern_testing_acc_3", target_lr=0.01)

# Binary: Baseline (no LR filter needed based on your snippet)
frontier_bin = get_best_frontier("binary baseline mnist", "64_bin_testing_acc")

# --- 2. Visualization (Subplots) ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8), sharey=True)

def plot_frontier(ax, data, title, color):
    ax.plot(data['kl'], data['mean'], marker='o', linestyle='-', color=color, linewidth=2, markersize=8)
    ax.errorbar(data['kl'], data['mean'], yerr=data['std'], fmt='none', ecolor='gray', alpha=0.4)
    
    # Annotate architectures
    for _, row in data.iterrows():
        ax.annotate(f"({int(row['k'])}, {int(row['l'])})", 
                    (row['kl'], row['mean']), 
                    textcoords="offset points", xytext=(0,10), 
                    ha='center', fontsize=9, fontweight='bold')
    
    ax.set_xscale('log')
    ax.set_title(title, fontsize=14, pad=15)
    ax.set_xlabel('Total Neurons ($k \\times l$)', fontsize=12)
    ax.grid(True, linestyle='--', alpha=0.6, which='both')

# Plot left: Ternary
plot_frontier(ax1, frontier_tern, "Ternary Search Frontier (LR=0.01)", "#2980b9")
ax1.set_ylabel('Mean Test Accuracy', fontsize=12)

# Plot right: Binary
plot_frontier(ax2, frontier_bin, "Binary Baseline Frontier", "#c0392b")

plt.suptitle("Architecture Efficiency: Ternary vs. Binary Baselines", fontsize=18, y=1.02)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, filename), dpi=200, bbox_inches='tight')
plt.show()

print("Processing complete. Comparison graph saved.")