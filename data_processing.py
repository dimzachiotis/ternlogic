import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
import matplotlib.pyplot as plt
import os

output_dir = "/home/dzachiotis/thesis/graphs/"
filename = "binary_mnist_scatter.png"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

full_path = os.path.join(output_dir, filename)

mlflow.set_tracking_uri("sqlite:////home/dzachiotis/thesis/mlflow_shared/mlflow.db")
client = MlflowClient()

experiment = mlflow.get_experiment_by_name("binary baseline mnist")
runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    filter_string="attributes.status = 'FINISHED'"
)

records = []
for run_id in runs['run_id']:
    run = client.get_run(run_id)
    params = run.data.params
    
    records.append({
        'k':        int(params.get('num_neurons')),
        'l':        int(params.get('num_layers')),
        'seed':     int(params.get('seed')),
        'dataset':  params.get('dataset'),
        'test_acc': run.data.metrics.get('64_bin_testing_acc')
    })

df = pd.DataFrame(records)

summary = df.groupby(['k', 'l'])['test_acc'].agg(['mean', 'std', 'count']).reset_index()
summary.columns = ['k', 'l', 'mean_acc', 'std_acc', 'num_seeds']
summary['kl'] = summary['k'] * summary['l']
summary = summary.sort_values(['kl']).reset_index(drop=True)

# print(summary.to_string(index=False))

fig, ax = plt.subplots(figsize=(20, 18))

ax.set_xscale('log')

ax.set_xlim(1000, 1000000)

# Use ax.errorbar with fmt='o' to create scatter points without lines.
ax.errorbar(
    summary['kl'],
    summary['mean_acc'],
    yerr=summary['std_acc'],
    fmt='o',                    # Scatter points only (no lines)
    markersize=7,               # Size of the mean markers
    markerfacecolor='royalblue',# Blue color for markers
    markeredgecolor='black',    # Black outline
    ecolor='gray',              # Gray error bars
    capsize=4,                  # Error bar caps
    alpha=0.8,                  # Transparency
    label='Mean Accuracy ($\pm$1 SD)'
)

# 2. Add labels for each point (k and l)
for i, row in summary.iterrows():
    # label text showing (neurons per layer, num layers)
    label = f"({int(row['k'])}, {int(row['l'])})"
    
    ax.annotate(
        label, 
        (row['kl'], row['mean_acc']),
        textcoords="offset points", # how to position the text
        xytext=(0, 10),             # distance from point (x,y)
        ha='center',                # horizontal alignment
        fontsize=8,
        color='black'
    )

# Formatting
ax.set_xlabel('Total Neurons ($k \\times l$)', fontsize=13)
ax.set_ylabel('Mean Test Accuracy', fontsize=13)
ax.set_title('Binary Network Accuracy on MNIST vs. Total Neuron Budget', fontsize=15, pad=18)
ax.grid(True, linestyle='--', alpha=0.5, which='both')

# Adding a simple legend
ax.legend(frameon=True, fontsize=11)

plt.tight_layout()
plt.savefig(full_path, dpi=200, bbox_inches='tight')
plt.show()