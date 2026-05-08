import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
import matplotlib.pyplot as plt
import os

# --- Configuration ---
output_dir = "/home/dzachiotis/thesis/graphs/"
filename = "group1_vs_binary.png"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

full_path = os.path.join(output_dir, filename)

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
        
        k = int(p.get('num_neurons', 0))
        l = int(p.get('num_layers', 0))
        
        records.append({
            'k': k, 'l': l, 'kl': k * l,
            'lr': float(p.get('learning_rate', 0)),
            'test_acc': m.get(metric_name)
        })
    
    df = pd.DataFrame(records)
    if target_lr is not None:
        df = df[df['lr'] == target_lr]
    
    # Average over seeds
    summary = df.groupby(['k', 'l', 'kl'])['test_acc'].agg(['mean', 'std']).reset_index()
    
    # Pick best (k, l) pair for each total neuron budget (kl)
    idx = summary.groupby('kl')['mean'].idxmax()
    return summary.loc[idx].sort_values('kl')

# --- 1. Process Data ---
frontier_tern = get_best_frontier("ternary search lr", "64_tern_testing_acc_3", target_lr=0.01)
frontier_bin = get_best_frontier("binary baseline mnist", "64_bin_testing_acc")

# --- 2. Create the Combined Plot ---
fig, ax = plt.subplots(figsize=(14, 9))

# Scatter Ternary (Circles)
ax.scatter(
    frontier_tern['kl'], 
    frontier_tern['mean'], 
    color='#1f77b4', 
    marker='o', 
    s=120, 
    label='Ternary Search (LR=0.01)', 
    edgecolors='black', 
    zorder=3
)

# Scatter Binary (Squares)
ax.scatter(
    frontier_bin['kl'], 
    frontier_bin['mean'], 
    color='#d62728', 
    marker='s', 
    s=100, 
    label='Binary Baseline', 
    edgecolors='black', 
    alpha=0.8,
    zorder=3
)

# --- 3. Annotations ---
# Offsetting Ternary labels UP and Binary labels DOWN to prevent overlap
for _, row in frontier_tern.iterrows():
    ax.annotate(f"({int(row['k'])}, {int(row['l'])})", 
                (row['kl'], row['mean']), 
                textcoords="offset points", xytext=(0, 12), 
                ha='center', fontsize=9, color='#1f77b4', fontweight='bold')

for _, row in frontier_bin.iterrows():
    ax.annotate(f"({int(row['k'])}, {int(row['l'])})", 
                (row['kl'], row['mean']), 
                textcoords="offset points", xytext=(0, -18), 
                ha='center', fontsize=9, color='#d62728', fontweight='bold')

# --- 4. Formatting ---
ax.set_xscale('log')
ax.set_xlabel('Total Neurons ($k \\times l$)', fontsize=13)
ax.set_ylabel('Mean Test Accuracy', fontsize=13)
ax.set_title('Efficiency Frontier: Ternary vs. Binary (Best Architecture Points)', fontsize=16, pad=20)
ax.grid(True, which='both', linestyle='--', alpha=0.4)
ax.legend(loc='lower right', fontsize=12)

# Adjust limits to ensure all annotations fit
ax.set_ylim(min(frontier_bin['mean'].min(), frontier_tern['mean'].min()) - 0.05, 1.0)

plt.tight_layout()
plt.savefig(full_path, dpi=200)
plt.show()

print(f"Combined scatter plot saved to: {full_path}")