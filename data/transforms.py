import torch
import numpy as np
from monai.transforms import (
    Compose,
    LoadImaged,
    AddChanneld,
    ScaleIntensityd,
    RandRotate90d,
    RandFlipd,
    ToTensord,
    NormalizeIntensityd,
)

def get_transforms(mode='train'):
    """
    Get transforms for different modes
    
    Args:
        mode: 'train', 'val', or 'test'
        
    Returns:
        transform: Composed transforms
    """
    if mode == 'train':
        return Compose([
            ScaleIntensityd(keys=["image"]),
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
            RandRotate90d(keys=["image", "mask"], prob=0.5),
            RandFlipd(keys=["image", "mask"], prob=0.5),
            ToTensord(keys=["image", "mask"]),
        ])
    else:
        return Compose([
            ScaleIntensityd(keys=["image"]),
            NormalizeIntensityd(keys=["image"], nonzero=True, channel_wise=True),
            ToTensord(keys=["image", "mask"]),
        ])