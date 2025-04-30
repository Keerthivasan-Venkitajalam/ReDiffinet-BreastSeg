import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms
from skimage import io

class BreastSegmentationDataset(Dataset):
    def __init__(self, data_path, transform=None, mode='train'):
        """
        Args:
            data_path: Path to the dataset
            transform: Optional transforms to apply
            mode: 'train', 'val', or 'test'
        """
        self.data_path = data_path
        self.transform = transform
        self.mode = mode
        
        # Get list of files
        self.image_files = self._get_file_list('images')
        self.mask_files = self._get_file_list('masks')
        
    def _get_file_list(self, subfolder):
        path = os.path.join(self.data_path, self.mode, subfolder)
        return sorted([os.path.join(path, f) for f in os.listdir(path) if f.endswith(('.png', '.jpg', '.tif'))])
        
    def __len__(self):
        return len(self.image_files)
        
    def __getitem__(self, idx):
        # Load image
        image_path = self.image_files[idx]
        image = io.imread(image_path)
        if len(image.shape) == 2:  # Grayscale
            image = np.expand_dims(image, axis=2)
        
        # Load mask
        mask_path = self.mask_files[idx]
        mask = io.imread(mask_path)
        
        # Convert to tensors
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0
        mask = torch.from_numpy(mask).long()
        
        # Apply transformations
        if self.transform:
            data = {"image": image, "mask": mask}
            data = self.transform(data)
            image, mask = data["image"], data["mask"]
        
        return {
            "image": image,
            "mask": mask,
            "image_path": image_path
        }

def get_dataloader(config, mode='train'):
    from data.transforms import get_transforms
    
    dataset = BreastSegmentationDataset(
        data_path=config.data_path,
        transform=get_transforms(mode),
        mode=mode
    )
    
    return DataLoader(
        dataset,
        batch_size=config.batch_size if mode == 'train' else 1,
        shuffle=mode == 'train',
        num_workers=config.num_workers,
        pin_memory=True
    )