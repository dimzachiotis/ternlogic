import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
import matplotlib.pyplot as plt
import os

# --- Configuration ---
output_dir = "/home/dzachiotis/thesis/graphs/"
filename = "binary_baseline_vs_group1_8_residual_vs_group1_8_baseline_only_common.png"
target_lr = 0.01

# --- SPECIFIC POINTS TO PLOT ---
# Add the exact (num_neurons, num_layers) configurations you want to keep.
# For example: [(64, 2), (128, 4), (256, 2)]
specific_points = [
    (64000, 6), 
    (32000, 6),
    (16000, 6),
    (8000, 6),
    (4000, 6),
    (2000, 6),
    (64000, 4), 
    (32000, 4),
] 

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

full_path = os.path.join(output_dir, filename)

mlflow.set_tracking_uri("sqlite:////home/dzachiotis/thesis/mlflow_shared/mlflow.db")
client = MlflowClient()

def get_all_architectures(exp_name, metric_name, target_lr=None):
    """Extracts runs and averages seeds for ALL (k, l) pairs."""
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
    
    # Average over seeds for every unique (k, l) architecture
    summary = df.groupby(['k', 'l', 'kl'])['test_acc'].agg(['mean', 'std']).reset_index()
    summary.columns = ['k', 'l', 'kl', 'mean_acc', 'std_acc'] 
    return summary.sort_values('kl')

def filter_to_specific_points(df, allowed_pairs):
    """Filters a dataframe to contain only rows where (k, l) is in allowed_pairs."""
    if not allowed_pairs:
        return df  # If the list is empty, don't filter anything
    # Create a boolean mask checking if the (k, l) tuple is in our list
    mask = df.apply(lambda row: (int(row['k']), int(row['l'])) in allowed_pairs, axis=1)
    return df[mask]

# --- 1. Get Data ---
all_group1_8 = get_all_architectures("group1_8_baseline", "64_tern_testing_acc_3", target_lr=target_lr)
all_group1_8_residual = get_all_architectures("group1_8_baseline_residual", "64_tern_testing_acc_3", target_lr=target_lr)
all_bin = get_all_architectures("binary_baseline_mnist", "64_bin_testing_acc")

# --- 1.5 Filter Data to Specific Points ---
all_group1_8 = filter_to_specific_points(all_group1_8, specific_points)
all_group1_8_residual = filter_to_specific_points(all_group1_8_residual, specific_points)
all_bin = filter_to_specific_points(all_bin, specific_points)

print("Filtered Ternary Group 1.8:\n", all_group1_8)
print("Filtered Ternary Group 1.8 Residual:\n", all_group1_8_residual)
print("Filtered Binary:\n", all_bin)

# --- 2. Create Scatter Plot ---
fig, ax = plt.subplots(figsize=(14, 12))

# Scatter Ternary Group 1.8
if not all_group1_8.empty:
    ax.scatter(all_group1_8['kl'], all_group1_8['mean_acc'], 
               label='Ternary Group 1.8',  
               color='#1f77b4', marker='h', s=100, alpha=0.7, edgecolors='black')

# Scatter Ternary Group 1.8 Residual
if not all_group1_8_residual.empty:
    ax.scatter(all_group1_8_residual['kl'], all_group1_8_residual['mean_acc'], 
               label='Ternary Group 1.8 residual',  
               color="#b41f94", marker='H', s=100, alpha=0.7, edgecolors='black')

# Scatter Binary
if not all_bin.empty:
    ax.scatter(all_bin['kl'], all_bin['mean_acc'], 
               label='Binary Architectures', 
               color='#d62728', marker='s', s=80, alpha=0.6, edgecolors='black')

# --- 3. Annotations ---
for _, row in all_group1_8.iterrows():
    ax.annotate(f"({int(row['k'])}, {int(row['l'])})", 
                (row['kl'], row['mean_acc']), 
                textcoords="offset points", xytext=(0, 10), 
                ha='center', fontsize=8, color='#1f77b4')
    
for _, row in all_group1_8_residual.iterrows():
    ax.annotate(f"({int(row['k'])}, {int(row['l'])})", 
                (row['kl'], row['mean_acc']), 
                textcoords="offset points", xytext=(0, 10), 
                ha='center', fontsize=8, color="#b41f94")

for _, row in all_bin.iterrows():
    ax.annotate(f"({int(row['k'])}, {int(row['l'])})", 
                (row['kl'], row['mean_acc']), 
                textcoords="offset points", xytext=(0, -15), 
                ha='center', fontsize=8, color='#d62728')

# --- 4. Formatting ---
ax.set_xscale('log')
ax.set_xlabel('Total Neurons ($k \\times l$)', fontsize=13)
ax.set_ylabel('Mean Test Accuracy', fontsize=13)
ax.set_title('Architecture Comparison: Specific Configurations (Ternary vs. Binary)', fontsize=16, pad=20)
ax.grid(True, which='both', linestyle='--', alpha=0.4)
ax.legend(loc='lower right', fontsize=12)

# Dynamic Y-axis layout based on available data
all_means = pd.concat([all_bin['mean_acc'], all_group1_8['mean_acc'], all_group1_8_residual['mean_acc']])
if not all_means.empty:
    ax.set_ylim(all_means.min() - 0.05, 1.0)
else:
    ax.set_ylim(0.0, 1.0)

plt.tight_layout()
plt.savefig(full_path, dpi=200)
plt.show()

print("Scatter plot generated showing specific architecture variations.")