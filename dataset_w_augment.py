import cv2
import os
import numpy as np
import random
import json
from PIL import Image
from torch.utils.data.dataset import Dataset
from torch.utils.data import Dataset
from torch.utils.data import  DataLoader
from tqdm import tqdm
from typing import *
import shutil
from torchvision import transforms
from utils import *
from loss import *
import datetime
##### for data augmentation:
import albumentations as A
from albumentations.pytorch import ToTensorV2


# log
t = str(datetime.datetime.now()).split(".")[0]
t = t.replace(":",".")
log_root = f"log/{t}"
writer = SummaryWriter(log_dir=log_root)

# dataset path
path_ROSSA = "dataset/ROSSA"
path_OCTA500_6M = "dataset/OCTA500_6M"
path_OCTA500_3M = "dataset/OCTA500_3M"


def prepareDatasets():
    all_datasets = {}

    all_datasets['OCTA500_3M'] = {
        "train": SegmentationDataset(os.path.join(path_OCTA500_3M, "train"), augment=True),
        "val": SegmentationDataset(os.path.join(path_OCTA500_3M, "val")),
        "test": SegmentationDataset(os.path.join(path_OCTA500_3M, "test"))
    }
    all_datasets['OCTA500_6M'] = {
        "train": SegmentationDataset(os.path.join(path_OCTA500_6M, "train"), augment=True),
        "val": SegmentationDataset(os.path.join(path_OCTA500_6M, "val")),
        "test": SegmentationDataset(os.path.join(path_OCTA500_6M, "test"))
    }
    all_datasets['ROSSA'] = {
        "train": SegmentationDataset([os.path.join(path_ROSSA, x) for x in ["train_manual", "train_sam"]], augment=True),
        "val": SegmentationDataset(os.path.join(path_ROSSA, "val")),
        "test": SegmentationDataset(os.path.join(path_ROSSA, "test"))
    }

    return all_datasets






class SegmentationDataset(Dataset):
    def __init__(self, ls_path_dataset, start=0, end=1, augment=False):
        super().__init__()

        self.augment = augment

        if not isinstance(ls_path_dataset, list):
            ls_path_dataset = [ls_path_dataset]

        self.ls_item = []
        for path_dataset in ls_path_dataset:
            path_dir_image = os.path.join(path_dataset, "image")
            path_dir_label = os.path.join(path_dataset, "label")
            ls_file = os.listdir(path_dir_image)

            for name in ls_file:
                path_image = os.path.join(path_dir_image, name)
                path_label = os.path.join(path_dir_label, name)
                assert os.path.exists(path_image)
                assert os.path.exists(path_label)
                self.ls_item.append({
                    "name": name,
                    "path_image": path_image,
                    "path_label": path_label,
                })

        random.seed(0)
        random.shuffle(self.ls_item)
        start = int(start * len(self.ls_item))
        end = int(end * len(self.ls_item))
        self.ls_item = self.ls_item[start:end]

        self.transform_train = A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.GaussianBlur(p=0.3),
            A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=0.2),
            A.Resize(224, 224),
        ])

        self.transform_val = A.Compose([
            A.Resize(224, 224),
        ])

    def __len__(self):
        return len(self.ls_item)

    def __getitem__(self, index):
        index = index % len(self)
        item = self.ls_item[index]

        name = item['name']
        path_image = item['path_image']
        path_label = item['path_label']

        image = cv2.imread(path_image, cv2.IMREAD_GRAYSCALE).astype("float32") / 255.0
        label = cv2.imread(path_label, cv2.IMREAD_GRAYSCALE).astype("float32") / 255.0

        # Apply transforms
        if self.augment:
            augmented = self.transform_train(image=image, mask=label)
        else:
            augmented = self.transform_val(image=image, mask=label)

        image = augmented['image']
        label = augmented['mask']

        # Add channel dim
        image = image[np.newaxis, :, :]
        label = label[np.newaxis, :, :]

        return name, image, label




