
import re
import json
import cv2
import torch
import numpy as np

import kornia as K
import albumentations as A

from PIL import Image
from pathlib import Path
from datasets import register
from datasets.preprocess import parse_background_color, resize_with_aspect_and_gray_padding
from torchvision import transforms
from torch.utils.data import Dataset
try:
    from skimage.feature import local_binary_pattern
except ModuleNotFoundError:
    local_binary_pattern = None

def resize_fn(img, size):
    return transforms.ToTensor()(
        transforms.Resize(size, transforms.InterpolationMode.BICUBIC)(transforms.ToPILImage()(img))
    )

@register('parallel_images_lp')
class SR_paired_images_wrapper_lp(Dataset):
    def __init__(
            self,
            imgW,
            imgH,
            aug,
            image_aspect_ratio,
            background,
            lbp = False,
            EIR = False,
            test = False,
            preprocessed = False,
            dataset=None,
            ):
        
        self.imgW = imgW
        self.imgH = imgH
        self.background = parse_background_color(background)
        self.dataset = dataset
        self.aug = aug
        self.ar = image_aspect_ratio
        self.lbp = lbp
        self.EIR = EIR
        self.test = test
        self.preprocessed = preprocessed
        
        self.transform = np.array([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, brightness_by_max=True, always_apply=True, p=1.0),
                A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0), always_apply=True, p=1.0),
                A.InvertImg(always_apply=True),
                
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, always_apply=True, p=1.0),
                A.RandomGamma(gamma_limit=(80, 120), eps=None, always_apply=True, p=1.0),
                None
            ])
    
    def Open_image(self, img, cvt=True):
        img = cv2.imread(img)
        if cvt is True:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    
    def extract_plate_numbers(self, file_path, pattern):
        if Path(file_path).exists():
            plate_numbers = []
            with open(file_path, "r", encoding="utf-8") as file:
                for line in file:
                    matches = re.search(pattern, line)
                    if matches:
                        plate_numbers.append(matches.group(1))
            if plate_numbers:
                return plate_numbers[0]

        # Fallback for Pa7a3Hin raw-train structure:
        # .../track_x/hr-003.jpg -> .../track_x/annotations.json
        ann_path = Path(file_path).parent / "annotations.json"
        if ann_path.exists():
            with open(ann_path, "r", encoding="utf-8") as f:
                ann = json.load(f)
            plate = ann.get("plate_text", "")
            if plate:
                return plate

        raise FileNotFoundError(f"Label not found for sample. txt={file_path}, ann={ann_path}")
    
    def get_lbp(self, x):
        if local_binary_pattern is None:
            raise ModuleNotFoundError("scikit-image is required for LBP features.")
        radius = 2
        n_points = 8 * radius
        METHOD = 'uniform'

        lbp = local_binary_pattern(x, n_points, radius, METHOD)
        return lbp.astype(np.uint8)
    
    def collate_fn(self, datas):
        lrs = []
        hrs = []
        gts = []
        file_name = []
        for item in datas:      
            lr = self.Open_image(item['lr'])
            hr = self.Open_image(item['hr'])
            gt = self.extract_plate_numbers(Path(item['hr']).with_suffix('.txt'), pattern=r'plate: (\w+)')
  
            if self.test:
                file_name.append(item['hr'].split('/')[-1])
            
            if self.aug is not False:
                augment = np.random.choice(self.transform, replace = True)
                if augment is not None:
                    lr = augment(image=lr)["image"]

            if self.preprocessed:
                # Offline preprocessed path: images are already standardized.
                if lr.shape[:2] != (self.imgH, self.imgW):
                    lr = cv2.resize(lr, (self.imgW, self.imgH), interpolation=cv2.INTER_CUBIC)
                if hr.shape[:2] != (2 * self.imgH, 2 * self.imgW):
                    hr = cv2.resize(hr, (2 * self.imgW, 2 * self.imgH), interpolation=cv2.INTER_CUBIC)
                lr = transforms.ToTensor()(Image.fromarray(lr))
                hr = transforms.ToTensor()(Image.fromarray(hr))
            else:
                lr = resize_with_aspect_and_gray_padding(
                    lr,
                    out_h=self.imgH,
                    out_w=self.imgW,
                    gray_color=self.background,
                )
                lr = resize_fn(lr, (self.imgH, self.imgW))
                hr = K.enhance.equalize_clahe(transforms.ToTensor()(Image.fromarray(hr)).unsqueeze(0), clip_limit=4.0, grid_size=(2, 2))
                hr = K.utils.tensor_to_image(hr.mul(255.0).byte())
                hr = resize_with_aspect_and_gray_padding(
                    hr,
                    out_h=2 * self.imgH,
                    out_w=2 * self.imgW,
                    gray_color=self.background,
                )
                hr = resize_fn(hr, (2 * self.imgH, 2 * self.imgW))
            
            lrs.append(lr)
            hrs.append(hr)
            gts.append(gt)
            
        lr = torch.stack(lrs, dim=0)
        hr = torch.stack(hrs, dim=0)
        
        gt = gts
        del lrs
        del hrs
        del gts
        if self.test and not self.lbp:
            return {
                'lr': lr, 'hr': hr, 'gt': gt, 'name': file_name
                    }
        else:
            return {
                'lr': lr, 'hr': hr, 'gt': gt
                }
    
        
        
    def __len__(self):
        return len(self.dataset)
    
    def __getitem__(self, idx):
        return self.dataset[idx]        
