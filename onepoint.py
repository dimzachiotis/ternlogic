import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
import matplotlib.pyplot as plt
import os

# --- Configuration ---
output_dir = "/home/dzachiotis/thesis/graphs/"
filename = "best_of_each.png"
target_lr = 0.01

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

full_path = os.path.join(output_dir, filename)

mlflow.set_tracking_uri("sqlite:////home/dzachiotis/thesis/mlflow_shared/mlflow.db")
client = MlflowClient()

def get_all_architectures(exp_name, metric_name, target_lr=None):
    """Extracts runs and averages seeds for ALL (k, l) pairs."""
    experiment = mlflow.get_experiment_by_name(exp_name)
    if experiment is None:
        print(f"Warning: Experiment '{exp_name}' not found.")
        return pd.DataFrame()

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
    if target_lr is not None and not df.empty:
        df = df[df['lr'] == target_lr]
    
    if df.empty:
        return pd.DataFrame(columns=['k', 'l', 'kl', 'mean_acc', 'std_acc'])

    summary = df.groupby(['k', 'l', 'kl'])['test_acc'].agg(['mean', 'std']).reset_index()
    summary.columns = ['k', 'l', 'kl', 'mean_acc', 'std_acc'] 
    return summary

def get_best_per_kl(exp_name, metric_name, target_lr=None):
    """Filters the summary to keep only the best performing (k, l) for each kl value."""
    summary = get_all_architectures(exp_name, metric_name, target_lr)
    if summary.empty:
        return summary
    
    # Identify the index of the maximum mean_acc for each unique 'kl'
    best_indices = summary.groupby('kl')['mean_acc'].idxmax()
    return summary.loc[best_indices].sort_values('kl')

# --- 1. Get Data (Best points only) ---
best_group1 = get_best_per_kl("ternary search lr", "64_tern_testing_acc_3", target_lr=target_lr)
best_group1_1 = get_best_per_kl("group1_1", "64_tern_testing_acc_3", target_lr=target_lr)
best_bin = get_best_per_kl("binary baseline mnist", "64_bin_testing_acc")

# --- 2. Create Plot ---
fig, ax = plt.subplots(figsize=(14, 9))

# Plotting Best Ternary Group 1
if not best_group1.empty:
    ax.plot(best_group1['kl'], best_group1['mean_acc'], color='#1f77b4', alpha=0.3, linestyle='--')
    ax.scatter(best_group1['kl'], best_group1['mean_acc'], 
               label=f'Best Group 1 (LR={target_lr})', 
               color='#1f77b4', marker='o', s=120, edgecolors='black', zorder=3)
    for _, row in best_group1.iterrows():
        ax.annotate(f"({int(row['k'])}, {int(row['l'])})", (row['kl'], row['mean_acc']),
                    textcoords="offset points", xytext=(0, 12), ha='center', fontsize=9, color='#1f77b4')

# Plotting Best Ternary Group 1.1
if not best_group1_1.empty:
    ax.plot(best_group1_1['kl'], best_group1_1['mean_acc'], color='#30b41f', alpha=0.3, linestyle='--')
    ax.scatter(best_group1_1['kl'], best_group1_1['mean_acc'], 
               label=f'Best Group 1.1 (LR={target_lr})', 
               color="#30b41f", marker='<', s=120, edgecolors='black', zorder=3)
    for _, row in best_group1_1.iterrows():
        ax.annotate(f"({int(row['k'])}, {int(row['l'])})", (row['kl'], row['mean_acc']),
                    textcoords="offset points", xytext=(10, -15), ha='left', fontsize=9, color="#30b41f")

# Plotting Best Binary
if not best_bin.empty:
    ax.plot(best_bin['kl'], best_bin['mean_acc'], color='#d62728', alpha=0.3, linestyle='--')
    ax.scatter(best_bin['kl'], best_bin['mean_acc'], 
               label='Best Binary Baseline', 
               color='#d62728', marker='s', s=100, edgecolors='black', zorder=3)
    for _, row in best_bin.iterrows():
        ax.annotate(f"({int(row['k'])}, {int(row['l'])})", (row['kl'], row['mean_acc']),
                    textcoords="offset points", xytext=(0, -18), ha='center', fontsize=9, color='#d62728')

# --- 3. Formatting ---
ax.set_xscale('log')
ax.set_xlabel('Total Neurons ($k \\times l$)', fontsize=14, fontweight='bold')
ax.set_ylabel('Mean Test Accuracy', fontsize=14, fontweight='bold')
ax.set_title('Performance Frontier: Best Architecture per Total Neuron Count', fontsize=16, pad=25)
ax.grid(True, which='both', linestyle=':', alpha=0.6)
ax.legend(loc='lower right', fontsize=12, frameon=True, shadow=True)

# Dynamic Y-axis limits
all_means = pd.concat([best_group1['mean_acc'], best_group1_1['mean_acc'], best_bin['mean_acc']])
if not all_means.empty:
    ax.set_ylim(all_means.min() - 0.02, min(all_means.max() + 0.02, 1.0))

plt.tight_layout()
plt.savefig(full_path, dpi=300)
plt.show()

print(f"Success: Best architectures plot saved to {full_path}")