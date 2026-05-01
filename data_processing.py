import pandas as pd
import mlflow
from mlflow.tracking import MlflowClient
import matplotlib.pyplot as plt
import os

output_dir = "/home/dzachiotis/thesis/graphs/"
filename = "explore_lr.png"

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

full_path = os.path.join(output_dir, filename)

mlflow.set_tracking_uri("sqlite:////home/dzachiotis/thesis/mlflow_shared/mlflow.db")
client = MlflowClient()

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
        'k':        int(params.get('num_neurons')),
        'l':        int(params.get('num_layers')),
        'seed':     int(params.get('seed')),
        'lr':       float(params.get('learning_rate')),
        'kl':       int(params.get('total_neurons')),
        'dataset':  params.get('dataset'),
        'test_acc': run.data.metrics.get('64_tern_testing_acc_3')
    })

df = pd.DataFrame(records)

summary = df.groupby(['k', 'l', 'lr', 'kl'])['test_acc'].agg(['mean', 'std']).reset_index()
summary.columns = ['k', 'l', 'lr', 'kl', 'mean_acc', 'std_acc']
summary = summary.sort_values(['lr', 'kl'])

lrs = summary['lr'].unique()
cmap = plt.get_cmap('tab10')

fig, ax = plt.subplots(figsize=(20, 18))

# 1. Iterate through each learning rate group
for i, lr in enumerate(lrs):
    # Filter data for the current learning rate
    group = summary[summary['lr'] == lr]
    color = cmap(i % 10) 

    # 2. Plot lines and points WITHOUT yerr (std)
    ax.errorbar(
        group['kl'],        
        group['mean_acc'],  
        fmt='-o',           # '-' connects points, 'o' adds markers
        label=f'LR: {lr}',
        color=color,
        markersize=8,
        alpha=0.9,
        linewidth=2
    )

    # 3. Add labels specifically for this group's points
    for _, row in group.iterrows():
        label = f"({int(row['k'])}, {int(row['l'])})"
        ax.annotate(
            label, 
            (row['kl'], row['mean_acc']),
            textcoords="offset points", 
            xytext=(0, 10), 
            ha='center', 
            fontsize=9,
            color=color, 
            weight='bold',
            alpha=0.8
        )

# Formatting remains the same...
ax.set_xscale('log')
ax.set_xlim(1000, 1000000)
ax.set_xlabel('Total Neurons ($k \\times l$)', fontsize=14)
ax.set_ylabel('Mean Test Accuracy', fontsize=14)
ax.set_title('Binary Network Accuracy on MNIST vs. Total Neuron Budget', fontsize=18, pad=20)
ax.grid(True, linestyle='--', alpha=0.5, which='both')
ax.legend(title="Learning Rate", frameon=True, fontsize=12, title_fontsize=13)

plt.tight_layout()
plt.savefig(full_path, dpi=200, bbox_inches='tight')
plt.show()
# 1. Ensure we have the average over seeds (which you have in 'summary')
# 2. Find the index of the best learning rate for each (k, l) pair
idx = summary.groupby(['k', 'l'])['mean_acc'].idxmax()

# 3. Create a new dataframe with only the best performing LRs
best_lr_summary = summary.loc[idx].copy()

# 4. Sort by k and then by l
best_lr_summary = best_lr_summary.sort_values(['k', 'l']).reset_index(drop=True)

# Display the results
print("Best Learning Rate per Architecture (Sorted by k and l):")
print(best_lr_summary[['k', 'l', 'lr', 'mean_acc', 'std_acc']])