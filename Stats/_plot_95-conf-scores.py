import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np

# Use LaTeX-like style
plt.rcParams.update({
    "text.usetex": False,
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 13,
    "legend.fontsize": 11,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11
})

# Raw confidence interval data
raw_data = """
AC_Mamba_OCTA500_3M: [0.8086, 0.8234]
AC_Mamba_OCTA500_6M: [0.7992, 0.8106]
AC_Mamba_ROSSA: [0.8865, 0.8999]
H2Former_OCTA500_3M: [0.8121, 0.8242]
H2Former_OCTA500_6M: [0.7741, 0.7850]
H2Former_ROSSA: [0.8599, 0.8711]
HV_OCTAMamba_OCTA500_3M: [0.8681, 0.8803]
HV_OCTAMamba_OCTA500_6M: [0.8264, 0.8365]
HV_OCTAMamba_ROSSA: [0.8943, 0.9075]
H_vmunet_OCTA500_3M: [0.7157, 0.7338]
H_vmunet_OCTA500_6M: [0.6719, 0.6854]
H_vmunet_ROSSA: [0.7464, 0.7593]
MISSFormer_OCTA500_3M: [0.8264, 0.8389]
MISSFormer_OCTA500_6M: [0.8080, 0.8182]
MISSFormer_ROSSA: [0.8458, 0.8583]
OCTAMamba_OCTA500_3M: [0.8378, 0.8510]
OCTAMamba_OCTA500_6M: [0.8181, 0.8278]
OCTAMamba_ROSSA: [0.8940, 0.9063]
R2UNet_OCTA500_3M: [0.7238, 0.7392]
R2UNet_OCTA500_6M: [0.7260, 0.7404]
R2UNet_ROSSA: [0.7797, 0.7927]
Swin_Unet_OCTA500_3M: [0.7946, 0.8094]
Swin_Unet_OCTA500_6M: [0.7796, 0.7915]
Swin_Unet_ROSSA: [0.7969, 0.8088]
UNet++_OCTA500_3M: [0.8420, 0.8543]
UNet++_OCTA500_6M: [0.8229, 0.8347]
UNet++_ROSSA: [0.8853, 0.8986]
U_Net_OCTA500_3M: [0.7879, 0.8017]
U_Net_OCTA500_6M: [0.7542, 0.7652]
U_Net_ROSSA: [0.8329, 0.8438]
VM_UNetv2_OCTA500_3M: [0.6996, 0.7169]
VM_UNetv2_OCTA500_6M: [0.6058, 0.6189]
VM_UNetv2_ROSSA: [0.6613, 0.6775]
VM_UNet_OCTA500_3M: [0.7859, 0.8012]
VM_UNet_OCTA500_6M: [0.7687, 0.7808]
VM_UNet_ROSSA: [0.8517, 0.8637]
"""

# Parse into DataFrame
data = {
    "Model": [],
    "Dataset": [],
    "Lower": [],
    "Upper": [],
    "Mean": []
}

for line in raw_data.strip().splitlines():
    tag, ci = line.split(":")
    parts = tag.split("_")
    model = "_".join(parts[:-1]).replace("U_Net", "U-Net").replace("UNet++", "UNet++") \
                .replace("VM_UNet", "VM-UNet").replace("VM_UNetv2", "VM-UNetv2").replace("HV_OCTAMamba", "HV-OCTAMamba")
    dataset = parts[-1]
    low, high = eval(ci.strip())
    mean = (low + high) / 2

    data["Model"].append(model)
    data["Dataset"].append(dataset)
    data["Lower"].append(low)
    data["Upper"].append(high)
    data["Mean"].append(mean)

df = pd.DataFrame(data)

# Sort by model and set color palette
model_order = sorted(df["Model"].unique())
palette = {"OCTA500_3M": "#1f77b4", "OCTA500_6M": "#2ca02c", "ROSSA": "#d62728"}

# Plot setup
fig, ax = plt.subplots(figsize=(16, 7))

# Plot each dataset
for i, dataset in enumerate(["OCTA500_3M", "OCTA500_6M", "ROSSA"]):
    subset = df[df["Dataset"] == dataset].sort_values(by="Model")
    x = np.arange(len(subset)) + i * 0.25 - 0.25  # Adjust position by dataset
    y = subset["Mean"].values
    yerr = [y - subset["Lower"].values, subset["Upper"].values - y]
    
    ax.errorbar(x, y, yerr=yerr, fmt='o', label=dataset, capsize=3, color=palette[dataset], markersize=6, lw=1)

# Axes formatting
ax.set_xticks(np.arange(len(model_order)))
ax.set_xticklabels(model_order, rotation=45, ha='right')
ax.set_ylabel("Dice Coefficient")
ax.set_title("95\% Confidence Intervals of Dice Scores Across Models and Datasets")
ax.grid(True, linestyle="--", alpha=0.4)
ax.legend(title="Dataset")
plt.tight_layout()
plt.savefig("dice_ci_final_plot.png", dpi=300)
plt.show()
