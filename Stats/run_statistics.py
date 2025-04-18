import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import random

# Exemple : Dice scores pour chaque image (à remplacer par tes vraies données)
hv_octamamba_scores = np.array([0.8522, 0.8834, 0.8521, 0.7748, 0.8681, 0.8609, 0.8623, 0.8221, 0.8554, 0.8797, 0.8566, 0.8533, 0.8526, 0.8212, 0.8862, 0.8576, 0.8663, 0.8474, 0.868, 0.8643, 0.8789, 0.8566, 0.8559, 0.8508, 0.8756, 0.8979, 0.8626, 0.8083, 0.8506, 0.8811, 0.8545, 0.864, 0.8363, 0.8706, 0.8539, 0.8517, 0.8615, 0.8699, 0.8318, 0.8862, 0.892, 0.8794, 0.8835, 0.8575, 0.8727, 0.8675, 0.8533, 0.8359, 0.8283, 0.8788])
octamamba_scores    = np.array([0.8524, 0.8714, 0.842, 0.7724, 0.8608, 0.8428, 0.8362, 0.8033, 0.8494, 0.8661, 0.8402, 0.8424, 0.8428, 0.8272, 0.8697, 0.845, 0.8462, 0.8355, 0.8607, 0.8466, 0.8674, 0.8472, 0.8529, 0.8415, 0.8653, 0.8899, 0.8534, 0.8114, 0.8469, 0.8663, 0.8429, 0.8371, 0.8377, 0.8539, 0.8513, 0.8503, 0.8603, 0.8725, 0.8191, 0.8713, 0.8773, 0.8738, 0.8672, 0.8416, 0.8532, 0.8587, 0.8409, 0.8232, 0.8201, 0.8779])

# --- 1. Statistiques descriptives ---
def describe(scores, name):
    print(f"\n{name} Statistics:")
    print(f"Mean: {np.mean(scores):.4f}")
    print(f"Std: {np.std(scores):.4f}")
    print(f"Median: {np.median(scores):.4f}")
    print(f"Min: {np.min(scores):.4f}, Max: {np.max(scores):.4f}")

describe(hv_octamamba_scores, "HV-OCTAMamba")
describe(octamamba_scores, "OCTAMamba")

# --- 2. Test de Wilcoxon (paired, non paramétrique) ---
stat, p_value = stats.wilcoxon(hv_octamamba_scores, octamamba_scores)
print(f"\nWilcoxon signed-rank test:")
print(f"Statistic = {stat:.4f}, p-value = {p_value:.4f}")
if p_value < 0.05:
    print("=> Statistically significant difference (p < 0.05)")
else:
    print("=> No statistically significant difference (p ≥ 0.05)")

# --- 3. Bootstrap pour intervalle de confiance 95% ---
def bootstrap_ci(data, n_bootstrap=1000, alpha=0.05):
    boot_means = []
    for _ in range(n_bootstrap):
        sample = np.random.choice(data, size=len(data), replace=True)
        boot_means.append(np.mean(sample))
    lower = np.percentile(boot_means, 100 * alpha / 2)
    upper = np.percentile(boot_means, 100 * (1 - alpha / 2))
    return lower, upper

ci_hv = bootstrap_ci(hv_octamamba_scores)
ci_octa = bootstrap_ci(octamamba_scores)

print(f"\n95% Confidence Interval (HV-OCTAMamba): [{ci_hv[0]:.4f}, {ci_hv[1]:.4f}]")
print(f"95% Confidence Interval (OCTAMamba): [{ci_octa[0]:.4f}, {ci_octa[1]:.4f}]")

# --- 4. Visualisation (boxplot) ---
plt.figure(figsize=(6, 4))
sns.boxplot(data=[hv_octamamba_scores, octamamba_scores], palette="Set2")
plt.xticks([0, 1], ["HV-OCTAMamba", "OCTAMamba"])
plt.ylabel("Dice Score")
plt.title("Comparison of Segmentation Performance")
plt.grid(True)
plt.tight_layout()
plt.savefig("dice_score_comparison.png")
plt.show()
