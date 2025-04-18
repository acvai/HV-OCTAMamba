import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import os
from collections import defaultdict

# --- Load Scores ---
dice_folder = "dice_scores"
dice_scores = {}
for file in os.listdir(dice_folder):
    if file.endswith(".npy"):
        name = file.replace("_scores.npy", "")
        scores = np.load(os.path.join(dice_folder, file))
        dice_scores[name] = scores

# --- Group by Dataset ---
grouped_by_dataset = defaultdict(dict)
for model_key, scores in dice_scores.items():
    if model_key.count("_") < 2:
        continue
    *model_parts, dataset = model_key.split("_")
    model_name = "_".join(model_parts)
    grouped_by_dataset[dataset][model_name] = scores

# --- 1. Descriptive Statistics ---
def describe(scores, name):
    print(f"\n{name} Statistics:")
    print(f"Mean: {np.mean(scores):.4f}")
    print(f"Std: {np.std(scores):.4f}")
    print(f"Median: {np.median(scores):.4f}")
    print(f"Min: {np.min(scores):.4f}, Max: {np.max(scores):.4f}")

print("=== Descriptive Statistics ===")
for model_name, scores in dice_scores.items():
    describe(scores, model_name)

# --- 2. Wilcoxon + Paired T-test + MAE ---
print("\n=== Statistical Tests (vs HV_OCTAMamba) ===")
for dataset, models in grouped_by_dataset.items():
    reference_model = "HV_OCTAMamba"
    if reference_model not in models:
        print(f"[WARNING] Reference model missing in {dataset}")
        continue

    print(f"\n--- Dataset: {dataset} ---")
    ref_scores = models[reference_model]
    for model_name, scores in models.items():
        if model_name == reference_model:
            continue
        if len(ref_scores) != len(scores):
            print(f"[!] Skipped {model_name}: unequal lengths")
            continue

        # Wilcoxon
        try:
            w_stat, w_p = stats.wilcoxon(ref_scores, scores)
            print(f"Wilcoxon {model_name} vs {reference_model} → Stat={w_stat:.4f}, p={w_p:.4f}")
        except:
            print(f"Wilcoxon {model_name} failed")

        # T-test
        try:
            t_stat, t_p = stats.ttest_rel(ref_scores, scores)
            print(f"T-test   {model_name} vs {reference_model} → T={t_stat:.4f}, p={t_p:.4f}")
        except:
            print(f"T-test {model_name} failed")

        # MAE
        mae = np.mean(np.abs(ref_scores - scores))
        print(f"MAE      {model_name} vs {reference_model}: {mae:.4f}")

# --- 3. Bootstrap CI ---
def bootstrap_ci(data, n_bootstrap=1000, alpha=0.05):
    boot_means = [np.mean(np.random.choice(data, size=len(data), replace=True)) for _ in range(n_bootstrap)]
    lower = np.percentile(boot_means, 100 * alpha / 2)
    upper = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return lower, upper

print("\n=== 95% Confidence Intervals ===")
for model_name, scores in dice_scores.items():
    ci = bootstrap_ci(scores)
    print(f"{model_name}: [{ci[0]:.4f}, {ci[1]:.4f}]")

# --- 4. Shapiro Normality Test ---
print("\n=== Normality Check (Shapiro-Wilk Test) ===")
for model_name, scores in dice_scores.items():
    stat, p = stats.shapiro(scores)
    print(f"{model_name}: W = {stat:.4f}, p = {p:.4f} → {'Normal' if p > 0.05 else 'Not normal'}")

# --- 5. Boxplot Per Dataset ---
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
