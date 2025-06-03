import os
import cv2
import numpy as np
from tqdm import tqdm
from sklearn.metrics import f1_score

# === CONFIGURATION ===
base_dataset_path = "/mnt/c/Users/Amine/Projects/HV_OCTAMamba/dataset"
base_output_path = "/mnt/c/Users/Amine/Projects/HV_OCTAMamba/output"
overlay_output_path = "/mnt/c/Users/Amine/Projects/HV_OCTAMamba/overlays"
target_size = (224, 224)

# Couleurs en BGR
COLORS = {
    'fp': (0, 0, 255),    # Rouge pour les faux positifs
    'fn': (255, 0, 0),    # Bleu pour les faux négatifs
    'gt': (0, 255, 0),    # Vert pour la vérité terrain
    'tp': (0, 128, 0),    # Vert foncé pour les vrais positifs
    'bg': (255, 255, 255) # Blanc pour le fond
}

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

# === FONCTIONS UTILITAIRES ===
def compute_dice(pred, gt):
    pred = pred.flatten()
    gt = gt.flatten()
    return f1_score(gt, pred)

def add_legend(img):
    """Ajoute une légende colorée à l'image"""
    h, w = img.shape[:2]
    legend = np.zeros((100, w, 3), dtype=np.uint8) + 255
    
    # Texte de légende
    cv2.putText(legend, 'FP (False Positive): Red', (10, 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLORS['fp'], 1)
    cv2.putText(legend, 'FN (False Negative): Blue', (10, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLORS['fn'], 1)
    cv2.putText(legend, 'TP (True Positive): Dark Green', (10, 60), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLORS['tp'], 1)
    cv2.putText(legend, 'GT (Ground Truth): Light Green', (10, 80), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLORS['gt'], 1)
    
    return np.vstack([img, legend])

def generate_full_overlay(gt, pred):
    """Overlay avec zones pleines colorées"""
    gt_bin = (gt > 127).astype(np.uint8)
    pred_bin = (pred > 127).astype(np.uint8)
    
    fp = np.logical_and(pred_bin == 1, gt_bin == 0)
    fn = np.logical_and(pred_bin == 0, gt_bin == 1)
    tp = np.logical_and(pred_bin == 1, gt_bin == 1)
    
    overlay = np.zeros((*gt.shape, 3), dtype=np.uint8)
    overlay[fp] = COLORS['fp']
    overlay[fn] = COLORS['fn']
    overlay[tp] = COLORS['tp']
    
    # Fusion avec la vérité terrain
    gt_rgb = cv2.cvtColor(gt, cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(overlay, 0.7, gt_rgb, 0.3, 0)

def generate_outline_overlay(gt, pred):
    """Version avec contours seulement"""
    gt_bin = (gt > 127).astype(np.uint8)
    pred_bin = (pred > 127).astype(np.uint8)
    
    # Détection des contours
    fp = cv2.Canny(np.uint8(np.logical_and(pred_bin == 1, gt_bin == 0)) * 255, 50, 150)
    fn = cv2.Canny(np.uint8(np.logical_and(pred_bin == 0, gt_bin == 1)) * 255, 50, 150)
    tp = cv2.Canny(np.uint8(np.logical_and(pred_bin == 1, gt_bin == 1)) * 255, 50, 150)
    
    overlay = cv2.cvtColor(gt, cv2.COLOR_GRAY2BGR)
    overlay[fp > 0] = COLORS['fp']
    overlay[fn > 0] = COLORS['fn']
    overlay[tp > 0] = COLORS['tp']
    
    return overlay

# === EXECUTION PRINCIPALE ===
for dataset_name in datasets:
    gt_dir = os.path.join(base_dataset_path, dataset_name, "test", "label")
    filenames = sorted([f for f in os.listdir(gt_dir) if f.endswith(('.png', '.bmp'))])
    
    print(f"\n=== Dataset: {dataset_name} ===")
    
    for model_name, model_folder in models_dirs.items():
        pred_dir = os.path.join(base_output_path, model_folder, dataset_name, f"{model_folder}_v1_output_masks")
        dice_scores = []
        
        # Dossiers de sortie
        full_overlay_dir = os.path.join(overlay_output_path, "full", model_folder, dataset_name)
        outline_overlay_dir = os.path.join(overlay_output_path, "outline", model_folder, dataset_name)
        os.makedirs(full_overlay_dir, exist_ok=True)
        os.makedirs(outline_overlay_dir, exist_ok=True)
        
        print(f"\nModel: {model_name}")

        # On ne traite que la première image du dataset
        if filenames:  # Vérifie qu'il y a bien des fichiers
            fname = filenames[0]  # Prend la première image
            gt_path = os.path.join(gt_dir, fname)
            
            # Gestion des extensions
            pred_path = os.path.join(pred_dir, fname)
            if not os.path.exists(pred_path):
                alt_ext = ".png" if fname.endswith(".bmp") else ".bmp"
                pred_path = os.path.join(pred_dir, os.path.splitext(fname)[0] + alt_ext)
            if os.path.exists(pred_path):  # Si le fichier de prédiction existe
                # Lecture et redimensionnement
                gt = cv2.imread(gt_path, cv2.IMREAD_GRAYSCALE)
                pred = cv2.imread(pred_path, cv2.IMREAD_GRAYSCALE)
                gt = cv2.resize(gt, target_size, interpolation=cv2.INTER_NEAREST)
                pred = cv2.resize(pred, target_size, interpolation=cv2.INTER_NEAREST)
                
                # Calcul du score Dice
                gt_bin = (gt > 127).astype(np.uint8)
                pred_bin = (pred > 127).astype(np.uint8)
                dice = compute_dice(pred_bin, gt_bin)
                dice_scores.append(dice)
                
                # Génération des overlays
                full_overlay = generate_full_overlay(gt, pred)
                outline_overlay = generate_outline_overlay(gt, pred)
                
                # Sauvegarde en .png (même si le nom original était en .bmp)
                base_name = os.path.splitext(fname)[0]  # Supprime l'extension originale
                cv2.imwrite(os.path.join(full_overlay_dir, f"full_{base_name}.png"), full_overlay)
                cv2.imwrite(os.path.join(outline_overlay_dir, f"outline_{base_name}.png"), outline_overlay)

print("\n=== Traitement terminé ===")