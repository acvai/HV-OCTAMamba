import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from skimage.metrics import structural_similarity as ssim
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.stats import friedmanchisquare, wilcoxon
from statsmodels.stats.multitest import multipletests
import cv2

# Config
metrics = ['Dice', 'Jaccard', 'Accuracy', 'Sensitivity', 'Specificity', 'SSIM']
datasets = ["OCTA500_3M", "OCTA500_6M", "ROSSA"]
models_dirs = {
    "HV-OCTAMamba": "HV_OCTAMamba_last",
    "OCTAMamba": "OCTAMamba",
    "AC-Mamba": "AC-Mamba",
    "H-vmunet": "H_vmunet",
    "H2Former": "H2Former",
    "MISSFormer": "MISSFormer",
    "VM-UNet": "VM-UNet",
    "VM-UNetv2": "VM_Unetv2",
    "Swin-Unet": "Swin_Unet",
    "R2UNet": "R2U_Net",
    "UNet++": "UNetpp",
    "U-Net": "UNet"
}

base_dataset_path = "/mnt/c/Users/Amine/Projects/HV_OCTAMamba/dataset"
base_output_path = "/mnt/c/Users/Amine/Projects/HV_OCTAMamba/output"
target_size = (224, 224)


def load_masks(folder, filenames):
    masks = []
    for fname in filenames:
        path = os.path.join(folder, fname)
        if not os.path.exists(path):
            alt_ext = '.png' if fname.endswith('.bmp') else '.bmp'
            path = os.path.join(folder, os.path.splitext(fname)[0] + alt_ext)
            if not os.path.exists(path):
                continue
        mask = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
        masks.append((mask > 127).astype(np.uint8))
    return np.array(masks)


def calculate_metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true.flatten(), y_pred.flatten(), labels=[0, 1]).ravel()
    return {
        'Dice': 2 * tp / (2 * tp + fp + fn + 1e-8),
        'Jaccard': tp / (tp + fp + fn + 1e-8),
        'Accuracy': (tp + tn) / (tp + tn + fp + fn),
        'Sensitivity': tp / (tp + fn + 1e-8),
        'Specificity': tn / (tn + fp + 1e-8),
        'SSIM': ssim(y_true, y_pred, data_range=1)
    }


def compute_CD(avranks, n, alpha="0.05"):
    import math
    q_alpha = {
        "0.05": {2: 1.960, 3: 2.569, 4: 2.913, 5: 3.138, 6: 3.314, 7: 3.460, 8: 3.583, 9: 3.693, 10: 3.792, 11: 3.882, 12: 3.960, 13: 4.030}
    }
    k = len(avranks)
    q = q_alpha[str(alpha)][k]
    return q * math.sqrt(k * (k + 1) / (6.0 * n))


def plot_nemenyi(avranks, labels, cd, title="Nemenyi Diagram", filename="nemenyi.png"):
    sorted_indices = np.argsort(avranks)
    sorted_ranks = np.array(avranks)[sorted_indices]
    sorted_labels = np.array(labels)[sorted_indices]
    plt.figure(figsize=(12, 2))
    plt.xlim(min(sorted_ranks) - 0.5, max(sorted_ranks) + 0.5)
    plt.yticks([])
    plt.title(title)
    for i, (rank, label) in enumerate(zip(sorted_ranks, sorted_labels)):
        plt.plot([rank, rank], [0.2, 0.8], 'k-')
        plt.text(rank, 0.05, label, rotation=90, ha='center')
    x_start = max(sorted_ranks) - cd
    x_end = max(sorted_ranks)
    plt.plot([x_start, x_end], [0.9, 0.9], 'r-', lw=2)
    plt.text((x_start + x_end)/2, 0.95, f"CD={cd:.2f}", ha='center', color='r')
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


for dataset in datasets:
    print(f"\n=== Processing Dataset: {dataset} ===")
    gt_dir = os.path.join(base_dataset_path, dataset, "test", "label")
    filenames = sorted([f for f in os.listdir(gt_dir) if f.endswith(('.png', '.bmp'))])
    true_masks = load_masks(gt_dir, filenames)

    pred_masks = {}
    for model_name, model_folder in models_dirs.items():
        pred_dir = os.path.join(base_output_path, model_folder, dataset, f"{model_folder}_v1_output_masks")
        pred_masks[model_name] = load_masks(pred_dir, filenames)

    results = {m: [] for m in metrics}
    model_names = list(models_dirs.keys())

    for model in model_names:
        if len(pred_masks[model]) != len(true_masks):
            print(f"[WARNING] Skipped {model} due to size mismatch")
            for m in metrics:
                results[m].append([np.nan] * len(true_masks))
            continue
        model_result = {m: [] for m in metrics}
        for i in range(len(true_masks)):
            scores = calculate_metrics(true_masks[i], pred_masks[model][i])
            for m in metrics:
                model_result[m].append(scores[m])
        for m in metrics:
            results[m].append(model_result[m])

    # Convert results
    df_stats = pd.DataFrame(index=model_names)
    nemenyi_data = {}
    for m in metrics:
        scores = results[m]
        df_stats[f'{m}_mean'] = np.nanmean(scores, axis=1)
        df_stats[f'{m}_std'] = np.nanstd(scores, axis=1)

        try:
            stat, p = friedmanchisquare(*scores)
            print(f"Friedman test for {m}: χ²={stat:.4f}, p={p:.4f}")
            if p < 0.05:
                avranks = np.nanmean([rankdata for rankdata in np.argsort(scores, axis=0)], axis=1)
                cd = compute_CD(avranks, n=len(true_masks))
                plot_nemenyi(avranks, model_names, cd, title=f"Nemenyi - {dataset} - {m}", filename=f"nemenyi_{dataset}_{m}.png")
        except Exception as e:
            print(f"Error in Friedman test for {m}: {e}")

    # PCA plot
    matrix = np.array([np.nanmean(results[m], axis=1) for m in metrics]).T
    if np.isnan(matrix).any():
        matrix = np.nan_to_num(matrix)
    scaler = StandardScaler()
    X = scaler.fit_transform(matrix)
    pca = PCA(n_components=2)
    components = pca.fit_transform(X)
    plt.figure(figsize=(8, 6))
    plt.scatter(components[:, 0], components[:, 1])
    for i, label in enumerate(model_names):
        plt.annotate(label, (components[i, 0], components[i, 1]))
    plt.title(f"PCA - {dataset}")
    plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
    plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
    plt.grid()
    plt.tight_layout()
    plt.savefig(f"pca_{dataset}.png")
    plt.close()

    # Save stats
    df_stats.to_csv(f"stats_{dataset}.csv")

################################  version 1 #######################################
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# from skimage.metrics import structural_similarity as ssim
# from sklearn.metrics import confusion_matrix
# from scipy.stats import friedmanchisquare, wilcoxon
# from sklearn.decomposition import PCA
# from sklearn.preprocessing import StandardScaler
# from Orange.evaluation.scoring import compute_CD, graph_ranks
# from statsmodels.stats.multitest import multipletests
# import seaborn as sns
# import os
# from PIL import Image
# import cv2



# # ----------------------------
# # 1. Configuration initiale
# # ----------------------------
# datasets = ["OCTA500_3M", "OCTA500_6M", "ROSSA"]
# models_dirs = {
#     "HV-OCTAMamba": "HV_OCTAMamba_last",
#     "OCTAMamba": "OCTAMamba",
#     "AC-Mamba": "AC-Mamba",
#     "H-vmunet": "H_vmunet",
#     "H2Former": "H2Former",
#     "MISSFormer": "MISSFormer",
#     "VM-UNet": "VM-UNet",
#     "VM-UNetv2": "VM_Unetv2",
#     "Swin-Unet": "Swin_Unet",
#     "R2UNet": "R2U_Net",
#     "UNet++": "UNetpp",
#     "U-Net": "UNet"
# }
# MODELS = list(models_dirs.keys())
# METRICS = ['Dice', 'Jaccard', 'Accuracy', 'Sensitivity', 'Specificity', 'SSIM']
# target_size = (224, 224)

# base_dataset_path = "/mnt/c/Users/Amine/Projects/HV_OCTAMamba/dataset"
# base_output_path = "/mnt/c/Users/Amine/Projects/HV_OCTAMamba/output"

# # ----------------------------
# # 2. Chargement des masques
# # ----------------------------
# def load_masks(mask_dir, filenames):
#     masks = []
#     for fname in filenames:
#         path = os.path.join(mask_dir, fname)
#         if not os.path.exists(path):
#             alt_ext = ".png" if fname.endswith(".bmp") else ".bmp"
#             fname_alt = os.path.splitext(fname)[0] + alt_ext
#             path = os.path.join(mask_dir, fname_alt)
#         if not os.path.exists(path):
#             masks.append(None)
#             continue
#         mask = np.array(Image.open(path).convert("L"))
#         mask = cv2.resize(mask, target_size, interpolation=cv2.INTER_NEAREST)
#         mask = (mask > 127).astype(np.uint8)
#         masks.append(mask)
#     return masks

# # ----------------------------
# # 3. Calcul des métriques
# # ----------------------------
# def calculate_metrics(y_true, y_pred):
#     tn, fp, fn, tp = confusion_matrix(y_true.flatten(), y_pred.flatten()).ravel()
#     return {
#         'Dice': 2 * tp / (2 * tp + fp + fn + 1e-8),
#         'Jaccard': tp / (tp + fp + fn + 1e-8),
#         'Accuracy': (tp + tn) / (tp + tn + fp + fn),
#         'Sensitivity': tp / (tp + fn + 1e-8),
#         'Specificity': tn / (tn + fp + 1e-8),
#         'SSIM': ssim(y_true, y_pred, data_range=1)
#     }

# def calculate_all_metrics(true_masks, pred_masks_by_model):
#     results = {metric: [] for metric in METRICS}
#     for model in MODELS:
#         model_metrics = {metric: [] for metric in METRICS}
#         for i in range(len(true_masks)):
#             if true_masks[i] is None or pred_masks_by_model[model][i] is None:
#                 continue
#             metrics = calculate_metrics(true_masks[i], pred_masks_by_model[model][i])
#             for metric in METRICS:
#                 model_metrics[metric].append(metrics[metric])
#         for metric in METRICS:
#             results[metric].append(model_metrics[metric])
#     return results

# # ----------------------------
# # 4. Traitement par dataset
# # ----------------------------
# for dataset in datasets:
#     print(f"\n=== Processing Dataset: {dataset} ===")
#     gt_dir = os.path.join(base_dataset_path, dataset, "test", "label")
#     filenames = sorted([f for f in os.listdir(gt_dir) if f.endswith(('.png', '.bmp'))])
#     true_masks = load_masks(gt_dir, filenames)

#     pred_masks = {}
#     for model_name, model_folder in models_dirs.items():
#         pred_dir = os.path.join(base_output_path, model_folder, dataset, f"{model_folder}_v1_output_masks")
#         pred_masks[model_name] = load_masks(pred_dir, filenames)

#     results = calculate_all_metrics(true_masks, pred_masks)

#     # ----------------------------
#     # 5. Analyse Statistique
#     # ----------------------------
#     def perform_statistical_analysis(results):
#         df = pd.DataFrame(index=MODELS)
#         for metric in METRICS:
#             df[f'{metric}_mean'] = np.mean(results[metric], axis=1)
#             df[f'{metric}_std'] = np.std(results[metric], axis=1)
        
#         friedman_results = {}
#         nemenyi_diagrams = {}
#         wilcoxon_results = {}
#         for metric in METRICS:
#             stat, p = friedmanchisquare(*[np.array(results[metric][i]) for i in range(len(MODELS))])
#             friedman_results[metric] = (stat, p)
#             if p < 0.05:
#                 ranks = np.mean(results[metric], axis=1)
                
#                 # cd = Orange.evaluation.compute_CD(ranks, len(results[metric][0]))
#                 cd = compute_CD(ranks, len(true_masks))
#                 graph_ranks(ranks, MODELS, cd=cd, width=6, textspace=1.5)

#                 nemenyi_diagrams[metric] = (ranks, cd)
#                 pvals = []
#                 for i in range(1, len(MODELS)):
#                     _, p = wilcoxon(results[metric][0], results[metric][i])
#                     pvals.append(p)
#                 rejected, corrected_p, _, _ = multipletests(pvals, method='fdr_bh')
#                 wilcoxon_results[metric] = corrected_p
#         return df, friedman_results, nemenyi_diagrams, wilcoxon_results

#     df_stats, friedman_res, nemenyi, wilcoxon_res = perform_statistical_analysis(results)

#     # ----------------------------
#     # 6. Visualisations
#     # ----------------------------
#     def generate_visualizations(results, nemenyi, wilcoxon_res, dataset):
#         for metric in METRICS:
#             plt.figure(figsize=(12, 6))
#             sns.boxplot(data=pd.DataFrame(np.array(results[metric]).T, columns=MODELS))
#             plt.xticks(rotation=45)
#             plt.title(f'{dataset} - Distribution {metric}')
#             plt.tight_layout()
#             plt.savefig(f'stats_output/{dataset}_boxplot_{metric}.png', dpi=300)
#             plt.close()
        
#         for metric in nemenyi:
#             ranks, cd = nemenyi[metric]
#             Orange.evaluation.graph_ranks(ranks, MODELS, cd=cd, width=6, textspace=1.5)
#             plt.title(f'{dataset} - Nemenyi Diagram - {metric}')
#             plt.savefig(f'stats_output/{dataset}_nemenyi_{metric}.png', dpi=300)
#             plt.close()

#         X = StandardScaler().fit_transform(np.array([np.mean(results[m], axis=1) for m in METRICS]).T)
#         pca = PCA(n_components=2)
#         components = pca.fit_transform(X)
#         plt.figure(figsize=(10, 8))
#         plt.scatter(components[:, 0], components[:, 1])
#         for i, model in enumerate(MODELS):
#             plt.annotate(model, (components[i, 0], components[i, 1]))
#         plt.xlabel(f'PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)')
#         plt.ylabel(f'PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)')
#         plt.title(f'{dataset} - PCA Analysis')
#         plt.grid()
#         plt.savefig(f'stats_output/{dataset}_pca.png', dpi=300)
#         plt.close()

#     os.makedirs("stats_output", exist_ok=True)
#     df_stats.to_csv(f'stats_output/stats_{dataset}.csv')
#     generate_visualizations(results, nemenyi, wilcoxon_res, dataset)

# print("✅ Analyse complète pour tous les datasets.")
