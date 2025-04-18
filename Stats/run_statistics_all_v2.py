import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Chemin du dossier contenant les fichiers .npy
dice_folder = "dice_scores"

# Charger tous les fichiers .npy dans un dictionnaire
dice_scores = {}
for file in os.listdir(dice_folder):
    if file.endswith(".npy"):
        name = file.replace("_scores.npy", "")
        scores = np.load(os.path.join(dice_folder, file))
        dice_scores[name] = scores

# --- 1. Statistiques descriptives ---
def describe(scores, name):
    print(f"\n{name} Statistics:")
    print(f"Mean: {np.mean(scores):.4f}")
    print(f"Std: {np.std(scores):.4f}")
    print(f"Median: {np.median(scores):.4f}")
    print(f"Min: {np.min(scores):.4f}, Max: {np.max(scores):.4f}")

print("=== Descriptive Statistics ===")
for model_name, scores in dice_scores.items():
    describe(scores, model_name)

# --- 2. Wilcoxon Test ---
##### version 1:
# reference_model = "HV_OCTAMamba"  # Changer ici si tu veux comparer à un autre modèle
# print(f"\n=== Wilcoxon Tests (compared to {reference_model}) ===")
# ref_scores = dice_scores[reference_model]
# for model_name, scores in dice_scores.items():
#     if model_name == reference_model:
#         continue
#     stat, p_value = stats.wilcoxon(ref_scores, scores)
#     print(f"{model_name} vs {reference_model} --> Stat = {stat:.4f}, p = {p_value:.4f}")
#     if p_value < 0.05:
#         print("  => Statistically significant (p < 0.05)")
#     else:
#         print("  => Not significant (p ≥ 0.05)")
##### version 2:
# Group scores by dataset
from collections import defaultdict

# Regroup by dataset name
grouped_by_dataset = defaultdict(dict)
for model_key, scores in dice_scores.items():
    if model_key.count("_") < 2:
        continue  # Skip malformed names
    *model_parts, dataset = model_key.split("_")
    model_name = "_".join(model_parts)
    grouped_by_dataset[dataset][model_name] = scores

# Run Wilcoxon test per dataset
print("\n=== Wilcoxon Tests (compared to HV_OCTAMamba) ===")
for dataset, models in grouped_by_dataset.items():
    reference_model = "HV_OCTAMamba"
    if reference_model not in models:
        print(f"[WARNING] Reference model missing in {dataset}")
        continue

    print(f"\nDataset: {dataset}")
    ref_scores = models[reference_model]
    for model_name, scores in models.items():
        if model_name == reference_model:
            continue
        if len(ref_scores) != len(scores):
            print(f"  [Error] Cannot compare {model_name} with {reference_model} in {dataset} (unequal length)")
            continue
        try:
            stat, p_value = stats.wilcoxon(ref_scores, scores)
            print(f"{model_name} vs {reference_model} --> Stat = {stat:.4f}, p = {p_value:.4f}")
            if p_value < 0.05:
                print("  => Statistically significant (p < 0.05)")
            else:
                print("  => Not significant (p ≥ 0.05)")
        except Exception as e:
            print(f"  [Error] {model_name} vs {reference_model}: {e}")


# --- 3. Bootstrap Confidence Interval ---
def bootstrap_ci(data, n_bootstrap=1000, alpha=0.05):
    boot_means = [np.mean(np.random.choice(data, size=len(data), replace=True)) for _ in range(n_bootstrap)]
    lower = np.percentile(boot_means, 100 * alpha / 2)
    upper = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return lower, upper

print("\n=== 95% Confidence Intervals ===")
for model_name, scores in dice_scores.items():
    ci = bootstrap_ci(scores)
    print(f"{model_name}: [{ci[0]:.4f}, {ci[1]:.4f}]")

##### # --- 4. Visualisation Boxplot --- ######################## version 1
# plt.figure(figsize=(10, 6))
# sns.boxplot(data=[v for v in dice_scores.values()], palette="Set3")
# plt.xticks(ticks=range(len(dice_scores)), labels=dice_scores.keys(), rotation=45)
# plt.ylabel("Dice Score")
# plt.title("Dice Score Comparison Between Models")
# plt.grid(True, linestyle='--', alpha=0.6)
# plt.tight_layout()
# plt.savefig("dice_score_comparison_all_models.png", dpi=300)
# plt.show()

##### # --- 4. Visualisation Boxplot --- ######################## version 2
# --- 4. Visualisation Boxplot (per dataset) ---
for dataset, model_scores in grouped_by_dataset.items():
    plt.figure(figsize=(10, 5))
    data = list(model_scores.values())
    labels = list(model_scores.keys())
    sns.boxplot(data=data, palette="Set2")
    plt.xticks(ticks=range(len(labels)), labels=labels, rotation=45)
    plt.ylabel("Dice Score")
    plt.title(f"Dice Score Comparison - {dataset}")
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.savefig(f"dice_score_comparison_{dataset}.png", dpi=300)
    plt.show()
