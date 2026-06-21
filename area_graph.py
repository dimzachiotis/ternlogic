import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
import matplotlib.pyplot as plt
import os

# --- Configuration ---
output_dir = "/home/dzachiotis/thesis/graphs/"
target_lr = 0.01

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

mlflow.set_tracking_uri("sqlite:////home/dzachiotis/thesis/mlflow_shared/mlflow.db")
client = MlflowClient()

def get_all_architectures(exp_name, metric_name, target_lr=None):
    """Extracts runs and averages both accuracy and area metrics over seeds for each (k, l) pair."""
    experiment = mlflow.get_experiment_by_name(exp_name)
    if experiment is None:
        raise ValueError(f"Experiment '{exp_name}' not found.")
        
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
        
        # Helper to safely convert MLflow string parameters to float
        def safe_float(val):
            try:
                return float(val) if val is not None else None
            except ValueError:
                return None

        # Extract the area parameters per run/seed
        area = safe_float(p.get('area'))
        area_2inputs = safe_float(p.get('area_2inputs'))
        area_mul = safe_float(p.get('area_mul_inputs'))
        
        records.append({
            'k': k, 'l': l, 'kl': k * l,
            'area': area, 
            'area_2inputs': area_2inputs, 
            'area_mul': area_mul,
            'lr': safe_float(p.get('learning_rate', 0)),
            'test_acc': m.get(metric_name)
        })
    
    df = pd.DataFrame(records)
    if target_lr is not None:
        df = df[df['lr'] == target_lr]
    
    # Group ONLY by the architecture identifiers
    # and aggregate accuracy (mean + std) AND area metrics (mean) over the seeds
    summary = df.groupby(['k', 'l', 'kl']).agg({
        'test_acc': ['mean', 'std'],
        'area': 'mean',
        'area_2inputs': 'mean',
        'area_mul': 'mean'
    }).reset_index()
    
    # Flatten the multi-index columns cleanly
    summary.columns = [
        'k', 'l', 'kl', 
        'mean_acc', 'std_acc', 
        'mean_area', 'mean_area_2inputs', 'mean_area_mul'
    ] 
    return summary

# --- 1. Get Data ---
all_group1_8 = get_all_architectures("group1_8_baseline", "64_tern_testing_acc_3", target_lr=target_lr)
all_group1_7 = get_all_architectures("group1_7_baseline", "64_tern_testing_acc_3", target_lr=target_lr)
all_group1_6 = get_all_architectures("group1_6_baseline", "64_tern_testing_acc_3", target_lr=target_lr)
all_bin = get_all_architectures("binary_baseline_mnist", "64_bin_testing_acc")

# --- 2. Configuration for the two Area Plots ---
plot_modes = [
    {
        "name": "multiple_inputs",
        "ternary_col": "mean_area_mul",
        "filename": "binary_baseline_vs_group1_6_vs_group1_7_vs_group1_8_area_mul_inputs.png",
        "title": "Architecture Comparison: Accuracy vs Mean Area (Multiple Inputs)"
    },
    {
        "name": "2_inputs",
        "ternary_col": "mean_area_2inputs",
        "filename": "binary_baseline_vs_group1_6_vs_group1_7_vs_group1_8_area_2inputs.png",
        "title": "Architecture Comparison: Accuracy vs Mean Area (2-Inputs)"
    }
]

# --- 3. Generate Plots ---
for mode in plot_modes:
    fig, ax = plt.subplots(figsize=(14, 9))
    
    # Extract averaged area x-axis columns
    binary_x = all_bin['mean_area']
    ternary_x = all_group1_8[mode['ternary_col']]

    # Scatter Ternary Group 1.7
    ax.scatter(ternary_x, all_group1_8['mean_acc'], 
               label='Ternary Group 1.8', 
               color='#1f77b4', marker='h', s=100, alpha=0.7, edgecolors='black')
    
    # Scatter Ternary Group 1.7
    ax.scatter(ternary_x, all_group1_7['mean_acc'], 
               label='Ternary Group 1.7', 
               color="#2e1fb4", marker='>', s=100, alpha=0.7, edgecolors='black')
    
    # Scatter Ternary Group 1.6
    ax.scatter(ternary_x, all_group1_6['mean_acc'], 
               label='Ternary Group 1.6', 
               color="#1fb458", marker='o', s=100, alpha=0.7, edgecolors='black')
    
    # Scatter Binary
    ax.scatter(binary_x, all_bin['mean_acc'], 
               label='Binary Architectures', 
               color='#d62728', marker='s', s=80, alpha=0.6, edgecolors='black')
    
    # Annotate Ternary Points
    for _, row in all_group1_8.iterrows():
        x_val = row[mode['ternary_col']]
        if pd.notna(x_val):
            ax.annotate(f"({int(row['k'])}, {int(row['l'])})", 
                        (x_val, row['mean_acc']), 
                        textcoords="offset points", xytext=(0, 10), 
                        ha='center', fontsize=8, color='#1f77b4')

    for _, row in all_group1_7.iterrows():
        x_val = row[mode['ternary_col']]
        if pd.notna(x_val):
            ax.annotate(f"({int(row['k'])}, {int(row['l'])})", 
                        (x_val, row['mean_acc']), 
                        textcoords="offset points", xytext=(0, 10), 
                        ha='center', fontsize=8, color="#2e1fb4")
            
    for _, row in all_group1_6.iterrows():
        x_val = row[mode['ternary_col']]
        if pd.notna(x_val):
            ax.annotate(f"({int(row['k'])}, {int(row['l'])})", 
                        (x_val, row['mean_acc']), 
                        textcoords="offset points", xytext=(0, 10), 
                        ha='center', fontsize=8, color="#1fb458")
            
    # Annotate Binary Points
    for _, row in all_bin.iterrows():
        x_val = row['mean_area']
        if pd.notna(x_val):
            ax.annotate(f"({int(row['k'])}, {int(row['l'])})", 
                        (x_val, row['mean_acc']), 
                        textcoords="offset points", xytext=(0, -15), 
                        ha='center', fontsize=8, color='#d62728')
            
    # --- Formatting ---
    ax.set_xscale('log')  
    ax.set_xlabel('Mean Hardware Area (Averaged over Seeds)', fontsize=13)
    ax.set_ylabel('Mean Test Accuracy', fontsize=13)
    ax.set_title(mode['title'], fontsize=16, pad=20)
    ax.grid(True, which='both', linestyle='--', alpha=0.4)
    ax.legend(loc='lower right', fontsize=12)
    
    # Adjust Y-axis to see the spread clearly
    min_y = min(all_bin['mean_acc'].min(), all_group1_8['mean_acc'].min()) - 0.05
    ax.set_ylim(min_y, 1.0)
    
    plt.tight_layout()
    full_path = os.path.join(output_dir, mode['filename'])
    plt.savefig(full_path, dpi=200)
    plt.close()  
    
    print(f"Generated plot: {full_path}")