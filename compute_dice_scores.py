import os
import cv2
import numpy as np
from sklearn.metrics import f1_score

# Fonction pour calculer le Dice Score entre deux masques binaires
def compute_dice(pred, gt):
    pred = pred.flatten()
    gt = gt.flatten()
    return f1_score(gt, pred)

# Répertoires des masques ground truth et des masques prédits
gt_dir = "/mnt/c/Users/Amine/Projects/HV_OCTAMamba/dataset/OCTA500_3M/test/label"  # ex: .../test/label/
hv_octamamba_dir = "/mnt/c/Users/Amine/Projects/HV_OCTAMamba/output/z_backup_12-04-2025/HV_OCTAMamba_last/OCTA500_3M/HV_OCTAMamba_last_v1_output_masks"
octamamba_dir = "/mnt/c/Users/Amine/Projects/HV_OCTAMamba/output/z_backup_12-04-2025/OCTAMamba/OCTA500_3M/OCTAMamba_v1_output_masks"

# Liste des fichiers (suppose que tous les noms de fichiers sont identiques entre gt et prédictions)
filenames = sorted([f for f in os.listdir(gt_dir) if f.endswith(('.png', '.bmp'))])

# Vecteurs pour stocker les scores
hv_scores = []
octa_scores = []

# for fname in filenames:
    # gt_path = os.path.join(gt_dir, fname)
    # hv_path = os.path.join(hv_octamamba_dir, fname)
    # octa_path = os.path.join(octamamba_dir, fname)

    # # Charger les masques en niveaux de gris (0–255) puis binariser
    # gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    # hv = cv2.imread(hv_path, cv2.IMREAD_GRAYSCALE)
    # octa = cv2.imread(octa_path, cv2.IMREAD_GRAYSCALE)

    # gt_bin = (gt > 127).astype(np.uint8)
    # hv_bin = (hv > 127).astype(np.uint8)
    # octa_bin = (octa > 127).astype(np.uint8)

    # # Calcul du Dice
    # hv_dice = compute_dice(hv_bin, gt_bin)
    # octa_dice = compute_dice(octa_bin, gt_bin)

    # hv_scores.append(hv_dice)
    # octa_scores.append(octa_dice)

    # print(f"{fname}: HV-OCTAMamba Dice = {hv_dice:.4f}, OCTAMamba Dice = {octa_dice:.4f}")

# Spécifie la taille cible
target_size = (224, 224)

for fname in filenames:
    gt_path = os.path.join(gt_dir, fname)
    hv_path = os.path.join(hv_octamamba_dir, fname)
    octa_path = os.path.join(octamamba_dir, fname)

    # Charger les masques
    gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
    hv = cv2.imread(hv_path, cv2.IMREAD_GRAYSCALE)
    octa = cv2.imread(octa_path, cv2.IMREAD_GRAYSCALE)

    # Resize vers une taille standard
    gt = cv2.resize(gt, target_size, interpolation=cv2.INTER_NEAREST)
    hv = cv2.resize(hv, target_size, interpolation=cv2.INTER_NEAREST)
    octa = cv2.resize(octa, target_size, interpolation=cv2.INTER_NEAREST)

    # Binarisation
    gt_bin = (gt > 127).astype(np.uint8)
    hv_bin = (hv > 127).astype(np.uint8)
    octa_bin = (octa > 127).astype(np.uint8)

    # Dice
    hv_dice = compute_dice(hv_bin, gt_bin)
    octa_dice = compute_dice(octa_bin, gt_bin)

    hv_scores.append(hv_dice)
    octa_scores.append(octa_dice)

    print(f"{fname}: HV-OCTAMamba Dice = {hv_dice:.4f}, OCTAMamba Dice = {octa_dice:.4f}")


# Convertir en tableau numpy si tu veux continuer l'analyse
hv_scores = np.array(hv_scores)
octa_scores = np.array(octa_scores)

# Sauvegarder ou imprimer les résultats
print("\n=== Résumé ===")
print("HV-OCTAMamba scores:", np.round(hv_scores, 4).tolist())
print("OCTAMamba scores:", np.round(octa_scores, 4).tolist())

# Optionnel : sauvegarder les résultats dans un fichier
np.save("hv_octamamba_scores.npy", hv_scores)
np.save("octamamba_scores.npy", octa_scores)
