import cv2
import torch
import os
import numpy as np

from modelszoo.AC_Mamba import *
from modelszoo.VM_UNet import *
from modelszoo.Swin_UNet import *
from modelszoo.MISSFormer import *
from modelszoo.H2Former import *
from modelszoo.VM_UNet2 import *
from modelszoo.R2UNet import *
from modelszoo.H_vmunet import *
from model.HV_OCTAMamba import *
from modelszoo.unetpp import *
from model.OCTAMamba import *
# from modelszoo.VM_UNetpp import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class ObjectCreator:
    def __init__(self, args, cls) -> None:
        self.args = args
        self.cls_net = cls
    def __call__(self):
        return self.cls_net(**self.args)
    

models = {
    "HV_OCTAMamba_last": ObjectCreator(cls=HV_OCTAMamba, args=dict()),
    "OCTAMamba": ObjectCreator(cls=OCTAMamba, args=dict()),
    "AC-Mamba":ObjectCreator(cls=AC_MambaSeg,args=dict()),
    "VM_Unetv2": ObjectCreator(cls=VMUNetV2,args=dict(input_channels=1,num_classes=1)),
    "Swin_Unet": ObjectCreator(cls=SwinUnet,args=dict(num_classes=1,img_size=224)),
    "MISSFormer": ObjectCreator(cls=MISSFormer,args=dict(num_classes=1)),
    "H2Former": ObjectCreator(cls=res34_swin_MS, args=dict(image_size=224, num_class=1)),
    "R2U_Net": ObjectCreator(cls=R2U_Net,args=dict(img_ch=1, output_ch=1)),
    "H_vmunet":ObjectCreator(cls=H_vmunet,args=dict(num_classes=1, input_channels=1)),
    "UNetpp": ObjectCreator(cls=ResNet34UnetPlus, args=dict()),
    "VM-UNet": ObjectCreator(cls=VMUNet,args=dict(input_channels=1)),
    # "VM-UNetpp": VM_UNetpp,
}



datasets = ["OCTA500_3M", "OCTA500_6M", "ROSSA"]

base_path = '/mnt/c/Users/Amine/Projects/HV_OCTAMamba'

for model_name, model_class in models.items():
    for dataset in datasets:
        model_path = f"{base_path}/result/{model_name}/{dataset}/model_best.pth"
        image_folder = f"{base_path}/dataset/{dataset}/test/image/"
        output_folder = f"{base_path}/output/{model_name}/{dataset}/{model_name}_v1_output_masks"
        os.makedirs(output_folder, exist_ok=True)

        # Load model
        model = model_class().to(device)
        model.load_state_dict(torch.load(model_path, map_location=device))
        model.eval()

        print(f"Processing: Model = {model_name}, Dataset = {dataset}")

        # List images
        image_files = [f for f in os.listdir(image_folder) if f.endswith(('.png', '.bmp', '.jpg', '.jpeg'))]


        for image_file in image_files:
            image_path = os.path.join(image_folder, image_file)
            image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE).astype("float32") / 255.0
            image = cv2.resize(image, (224, 224), interpolation=cv2.INTER_LINEAR)
            image = image.reshape((1, 1, image.shape[0], image.shape[1]))
            image_tensor = torch.from_numpy(image).to(device)

            with torch.no_grad():
                output = model(image_tensor)
                if isinstance(output, list):
                    output = output[0]
                output = output.squeeze().cpu().numpy()

            output = (output > 0.5).astype(np.uint8) * 255
            mask_save_path = os.path.join(output_folder, image_file)
            cv2.imwrite(mask_save_path, output)
            print(f"Saved: {mask_save_path}")
