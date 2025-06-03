import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import scikit_posthocs as sp
from scipy.stats import friedmanchisquare
from statsmodels.stats.multitest import multipletests
from sklearn.preprocessing import MinMaxScaler

# ----------------------------
# 1. Load and Prepare Data
# ----------------------------
def load_and_label_data(file_path, dataset_name):
    df = pd.read_csv(file_path)
    df['Dataset'] = dataset_name
    return df

datasets = {
    'OCTA500_3M': 'stats_OCTA500_3M.csv',
    'OCTA500_6M': 'stats_OCTA500_6M.csv', 
    'ROSSA': 'stats_ROSSA.csv'
}

df_list = [load_and_label_data(file, name) for name, file in datasets.items()]
full_df = pd.concat(df_list).reset_index(drop=True)

# Clean model names
full_df['Model'] = full_df.iloc[:, 0].str.replace('_', '-')
full_df = full_df.drop(columns=full_df.columns[0])

# ----------------------------
# 2. Statistical Analysis
# ----------------------------
def friedman_nemenyi_analysis(df, metric):
    data = []
    model_names = df['Model'].unique()

    for model in model_names:
        scores = df[df['Model'] == model][metric].values
        data.append(scores)

    data = np.array(data).T
    stat, p = friedmanchisquare(*data.T)
    print(f"\nFriedman test for {metric}: χ²={stat:.3f}, p={p:.5f}")

    if p < 0.05:
        nemenyi = sp.posthoc_nemenyi_friedman(data)
        ranks = np.mean(np.argsort(np.argsort(-data, axis=1)), axis=0) + 1  # Lower is better
        plot_cd_diagram(ranks, model_names, metric)

def plot_cd_diagram(avg_ranks, labels, metric):
    sorted_idx = np.argsort(avg_ranks)
    sorted_ranks = avg_ranks[sorted_idx]
    sorted_labels = np.array(labels)[sorted_idx]

    plt.figure(figsize=(10, 2))
    for i, (label, rank) in enumerate(zip(sorted_labels, sorted_ranks)):
        plt.plot([rank, rank], [0.2, 0.8], color='black')
        plt.text(rank, 0.9, label, ha='center', va='bottom', fontsize=10, rotation=45)

    plt.title(f'Critical Difference Diagram - {metric}')
    plt.yticks([])
    plt.xlabel('Average Rank (Lower is Better)')
    plt.tight_layout()
    plt.savefig(f'nemenyi_{metric}.png', dpi=300)
    plt.close()

# Analyze key metrics
for metric in ['Dice_mean', 'Jaccard_mean', 'Accuracy_mean']:
    friedman_nemenyi_analysis(full_df, metric)

# ----------------------------
# 3. Visualization
# ----------------------------
def plot_metric_comparison(df, metric, ylabel):
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Model', y=metric, hue='Dataset', data=df)
    plt.xticks(rotation=45)
    plt.ylabel(ylabel)
    plt.title(f'{metric.split("_")[0]} Score Comparison')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(f'{metric}_comparison.png', dpi=300)
    plt.close()

plot_metric_comparison(full_df, 'Dice_mean', 'Dice Coefficient')
plot_metric_comparison(full_df, 'Accuracy_mean', 'Accuracy')

# ----------------------------
# 4. Radar Plot
# ----------------------------
def plot_radar_chart(df):
    metrics = ['Dice_mean', 'Jaccard_mean', 'Accuracy_mean', 'Sensitivity_mean', 'Specificity_mean', 'SSIM_mean']
    models = ['HV-OCTAMamba', 'AC-Mamba', 'UNet++', 'OCTAMamba']
    
    scaler = MinMaxScaler()
    plot_data = df[df['Model'].isin(models)].groupby('Model')[metrics].mean()
    plot_data = pd.DataFrame(scaler.fit_transform(plot_data), columns=metrics, index=plot_data.index)
    
    angles = np.linspace(0, 2*np.pi, len(metrics), endpoint=False)
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, polar=True)
    
    for model in models:
        values = plot_data.loc[model].values.flatten().tolist()
        values += values[:1]
        ax.plot(np.append(angles, angles[0]), values, 'o-', label=model)
        ax.fill(np.append(angles, angles[0]), values, alpha=0.1)
    
    ax.set_thetagrids(angles * 180/np.pi, [m.split('_')[0] for m in metrics])
    ax.set_title('Multi-Metric Performance Comparison (Normalized)', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
    plt.savefig('radar_chart_all.png', dpi=300, bbox_inches='tight')
    plt.close()

plot_radar_chart(full_df)

# ----------------------------
# 5. Generate LaTeX Table
# ----------------------------

# def create_latex_table(df, dataset_name):
#     subset = df[df['Dataset']==dataset_name]
#     latex = subset[['Model', 'Dice_mean', 'Dice_std', 'Jaccard_mean', 'Accuracy_mean']].copy()
#     latex.columns = ['Model', 'Dice', 'Dice STD', 'Jaccard', 'Accuracy']
    
#     for col in ['Dice', 'Jaccard', 'Accuracy']:
#         latex[col] = latex.apply(lambda x: f"{x[col]:.3f} ± {x[f'{col} STD']:.3f}", axis=1)
    
#     latex = latex.drop(columns=['Dice STD'])
#     return latex.to_latex(index=False, caption=f"Performance on {dataset_name} Dataset")

def create_latex_table(df, dataset_name):
    subset = df[df['Dataset'] == dataset_name].copy()
    
    # Metrics to include
    metrics = ['Dice', 'Jaccard', 'Accuracy']
    
    # Create columns for mean ± std format
    for metric in metrics:
        mean_col = f"{metric}_mean"
        std_col = f"{metric}_std"
        if mean_col in subset.columns and std_col in subset.columns:
            subset[metric] = subset.apply(lambda x: f"{x[mean_col]:.3f} ± {x[std_col]:.3f}", axis=1)
    
    # Select only the Model and formatted metrics
    final_table = subset[['Model'] + metrics]
    return final_table.to_latex(index=False, caption=f"Performance on {dataset_name} Dataset")


for dataset in datasets.keys():
    print(f"\nLaTeX Table for {dataset}:")
    print(create_latex_table(full_df, dataset))

# Save CSV
full_df.to_csv('combined_metrics.csv', index=False)
print("\nAnalysis complete. Results saved to:")
print("- combined_metrics.csv")
print("- Boxplots, radar chart, and CD plots")
print("- LaTeX table printed above")
