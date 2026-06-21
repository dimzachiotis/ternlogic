import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
import matplotlib.pyplot as plt
import os

# --- Configuration ---
output_dir = "/home/dzachiotis/thesis/graphs/"
filename = "binary_baseline_vs_group1_6_vs_group1_7_vs_group1_8_baseline.png"
target_lr = 0.01

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

# # --- 1. Get Data (All points, no filtering for max) ---
all_group1_7 = get_all_architectures("group1_7_baseline", "64_tern_testing_acc_3", target_lr=target_lr)
print(all_group1_7)
all_group1_8 = get_all_architectures("group1_8_baseline", "64_tern_testing_acc_3", target_lr=target_lr)
print(all_group1_8)
all_group1_6 = get_all_architectures("group1_6_baseline", "64_tern_testing_acc_3", target_lr=target_lr)
print(all_group1_6)
all_bin = get_all_architectures("binary_baseline_mnist", "64_bin_testing_acc")
print(all_bin)
# print(all_bin)

# # --- 2. Create Scatter Plot ---
fig, ax = plt.subplots(figsize=(14, 12))

# ax.scatter(all_group1_11['kl'], all_group1_11['mean_acc'], 
#            label='Ternary Group 1.11',  # Unique Label
#            color="#f5a1ee", marker='H', s=100, alpha=0.7, edgecolors='black')

# ax.scatter(all_group1_17['kl'], all_group1_17['mean_acc'], 
#            label='Ternary Group 1.17',  # Unique Label
#            color="#faec2a", marker='h', s=100, alpha=0.7, edgecolors='black')

ax.scatter(all_group1_7['kl'], all_group1_7['mean_acc'], 
           label='Ternary Group 1.7',  # Unique Label
           color="#2e1fb4", marker='>', s=100, alpha=0.7, edgecolors='black')

# Scatter Ternary Group 1
ax.scatter(all_group1_8['kl'], all_group1_8['mean_acc'], 
           label='Ternary Group 1.8',  # Unique Label
           color='#1f77b4', marker='h', s=100, alpha=0.7, edgecolors='black')

# Scatter Ternary Group 1
ax.scatter(all_group1_6['kl'], all_group1_6['mean_acc'], 
           label='Ternary Group 1.6',  # Unique Label
           color="#1fb458", marker='o', s=100, alpha=0.7, edgecolors='black')

# Scatter Binary
ax.scatter(all_bin['kl'], all_bin['mean_acc'], 
           label='Binary Architectures', 
           color='#d62728', marker='s', s=80, alpha=0.6, edgecolors='black')

# # --- 3. Annotations ---
# # Labeling every point with its (k, l)
# for _, row in all_group1_11.iterrows():
#     ax.annotate(f"({int(row['k'])}, {int(row['l'])})", 
#                 (row['kl'], row['mean_acc']), 
#                 textcoords="offset points", xytext=(10, 5), 
#                 ha='left', fontsize=8, color="#f5a1ee")
    
# for _, row in all_group1_17.iterrows():
#     ax.annotate(f"({int(row['k'])}, {int(row['l'])})", 
#                 (row['kl'], row['mean_acc']), 
#                 textcoords="offset points", xytext=(10, 5), 
#                 ha='left', fontsize=8, color="#faec2a")
    
for _, row in all_group1_7.iterrows():
    ax.annotate(f"({int(row['k'])}, {int(row['l'])})", 
                (row['kl'], row['mean_acc']), 
                textcoords="offset points", xytext=(10, 5), 
                ha='left', fontsize=8, color="#2e1fb4")

for _, row in all_group1_8.iterrows():
    ax.annotate(f"({int(row['k'])}, {int(row['l'])})", 
                (row['kl'], row['mean_acc']), 
                textcoords="offset points", xytext=(0, 10), 
                ha='center', fontsize=8, color='#1f77b4')
    
for _, row in all_group1_6.iterrows():
    ax.annotate(f"({int(row['k'])}, {int(row['l'])})", 
                (row['kl'], row['mean_acc']), 
                textcoords="offset points", xytext=(0, 10), 
                ha='center', fontsize=8, color="#1fb458")

for _, row in all_bin.iterrows():
    ax.annotate(f"({int(row['k'])}, {int(row['l'])})", 
                (row['kl'], row['mean_acc']), 
                textcoords="offset points", xytext=(0, -15), 
                ha='center', fontsize=8, color='#d62728')

# # --- 4. Formatting ---
ax.set_xscale('log')
ax.set_xlabel('Total Neurons ($k \\times l$)', fontsize=13)
ax.set_ylabel('Mean Test Accuracy', fontsize=13)
ax.set_title('Architecture Comparison: All Configurations (Ternary vs. Binary)', fontsize=16, pad=20)
ax.grid(True, which='both', linestyle='--', alpha=0.4)
ax.legend(loc='lower right', fontsize=12)

# Adjust Y-axis to see the spread clearly
ax.set_ylim(min(all_bin['mean_acc'].min(), all_group1_8['mean_acc'].min()) - 0.05, 1.0)

plt.tight_layout()
plt.savefig(full_path, dpi=200)
plt.show()

print("Scatter plot generated showing all architecture variations.")