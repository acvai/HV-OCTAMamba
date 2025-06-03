import os
import cv2
import numpy as np
from sklearn.metrics import f1_score

def compute_dice(pred, gt):
    pred = pred.flatten()
    gt = gt.flatten()
    return f1_score(gt, pred)

# === CONFIGURATION ===
base_dataset_path = "/mnt/c/Users/Amine/Projects/HV_OCTAMamba/dataset"
base_output_path = "/mnt/c/Users/Amine/Projects/HV_OCTAMamba/output"
target_size = (224, 224)

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

# === CALCUL DES SCORES DICE ===
for dataset_name in datasets:
    gt_dir = os.path.join(base_dataset_path, dataset_name, "test", "label")
    filenames = sorted([f for f in os.listdir(gt_dir) if f.endswith(('.png', '.bmp'))])

    print(f"\n=== Dataset: {dataset_name} ===")
    
    for model_name, model_folder in models_dirs.items():
        pred_dir = os.path.join(base_output_path, model_folder, dataset_name, f"{model_folder}_v1_output_masks")
        dice_scores = []

        for fname in filenames:
            gt_path = os.path.join(gt_dir, fname)

            # Essaye .bmp d'abord, sinon .png
            pred_path = os.path.join(pred_dir, fname)
            if not os.path.exists(pred_path):
                alt_ext = ".png" if fname.endswith(".bmp") else ".bmp"
                fname_alt = os.path.splitext(fname)[0] + alt_ext
                pred_path = os.path.join(pred_dir, fname_alt)
            if not os.path.exists(pred_path):
                print(f"[WARNING] Missing: {model_name} - {dataset_name} - {fname}")
                continue

            gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
            pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)

            gt = cv2.resize(gt, target_size, interpolation=cv2.INTER_NEAREST)
            pred = cv2.resize(pred, target_size, interpolation=cv2.INTER_NEAREST)

            gt_bin = (gt > 127).astype(np.uint8)
            pred_bin = (pred > 127).astype(np.uint8)

            dice = compute_dice(pred_bin, gt_bin)
            dice_scores.append(dice)

        dice_scores = np.array(dice_scores)
        print(f"{model_name}: Mean = {dice_scores.mean():.4f}, Std = {dice_scores.std():.4f}, N = {len(dice_scores)}, Array: {dice_scores}")
        np.save(f"dice_scores/dice_scores_{model_name.replace('-', '_')}_{dataset_name}.npy", dice_scores)
