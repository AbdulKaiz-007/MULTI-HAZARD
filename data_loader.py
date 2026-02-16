"""
Data Loader Module for Multi-Hazard Disaster Detection
Handles loading satellite images and multi-class segmentation masks
"""

import os
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2


class MultiHazardDataset(Dataset):
    """
    Dataset for multi-hazard disaster detection
    
    Mask Classes:
        0: Background
        1: Flood
        2: Fire
        3: Building Damage
    """
    
    def __init__(self, image_dir, mask_dir, transform=None, img_size=(256, 256)):
        """
        Args:
            image_dir (str): Directory containing satellite images
            mask_dir (str): Directory containing segmentation masks
            transform (albumentations.Compose): Augmentation pipeline
            img_size (tuple): Target image size (height, width)
        """
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        
        # Get list of images
        self.images = sorted([f for f in os.listdir(image_dir) 
                             if f.endswith(('.png', '.jpg', '.jpeg', '.tif'))])
        
        # Default transforms if none provided
        if transform is None:
            self.transform = A.Compose([
                A.Resize(height=img_size[0], width=img_size[1]),
                A.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225]),
                ToTensorV2()
            ])
        else:
            self.transform = transform
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        """
        Returns:
            image (torch.Tensor): Normalized image tensor [C, H, W]
            mask (torch.Tensor): Segmentation mask [H, W] with class indices
        """
        # Load image
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        image = np.array(Image.open(img_path).convert('RGB'))
        
        # Load mask (assuming same filename)
        mask_name = img_name.replace('.jpg', '.png').replace('.jpeg', '.png')
        mask_path = os.path.join(self.mask_dir, mask_name)
        
        if os.path.exists(mask_path):
            mask = np.array(Image.open(mask_path).convert('L'))  # Grayscale
        else:
            # Create empty mask if not found
            mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        
        # Apply transformations
        if self.transform:
            transformed = self.transform(image=image, mask=mask)
            image = transformed['image']
            mask = transformed['mask']
        
        # Convert mask to long tensor for CrossEntropyLoss
        # Check if mask is already a tensor (from ToTensorV2)
        if not isinstance(mask, torch.Tensor):
            mask = torch.from_numpy(mask).long()
        else:
            mask = mask.long()
        
        return image, mask


def get_train_transforms(img_size=(256, 256)):
    """Training augmentation pipeline"""
    return A.Compose([
        A.Resize(height=img_size[0], width=img_size[1]),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, 
                          rotate_limit=15, p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], 
                   std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


def get_val_transforms(img_size=(256, 256)):
    """Validation augmentation pipeline"""
    return A.Compose([
        A.Resize(height=img_size[0], width=img_size[1]),
        A.Normalize(mean=[0.485, 0.456, 0.406], 
                   std=[0.229, 0.224, 0.225]),
        ToTensorV2()
    ])


def create_dataloaders(train_img_dir, train_mask_dir, 
                       val_img_dir, val_mask_dir,
                       batch_size=8, img_size=(256, 256), 
                       num_workers=4):
    """
    Create train and validation dataloaders
    
    Args:
        train_img_dir (str): Training images directory
        train_mask_dir (str): Training masks directory
        val_img_dir (str): Validation images directory
        val_mask_dir (str): Validation masks directory
        batch_size (int): Batch size
        img_size (tuple): Image dimensions
        num_workers (int): Number of workers for data loading
    
    Returns:
        train_loader, val_loader: DataLoader objects
    """
    train_dataset = MultiHazardDataset(
        train_img_dir, train_mask_dir,
        transform=get_train_transforms(img_size),
        img_size=img_size
    )
    
    val_dataset = MultiHazardDataset(
        val_img_dir, val_mask_dir,
        transform=get_val_transforms(img_size),
        img_size=img_size
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,  # Set to 0 for Windows compatibility
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,  # Set to 0 for Windows compatibility
        pin_memory=True
    )
    
    return train_loader, val_loader


if __name__ == "__main__":
    # Example usage
    print("Multi-Hazard Dataset Loader")
    print("=" * 50)
    
    # Example paths (update with your actual paths)
    train_img_dir = "data/train/images"
    train_mask_dir = "data/train/masks"
    
    if os.path.exists(train_img_dir) and os.path.exists(train_mask_dir):
        dataset = MultiHazardDataset(train_img_dir, train_mask_dir)
        print(f"Dataset size: {len(dataset)}")
        
        if len(dataset) > 0:
            image, mask = dataset[0]
            print(f"Image shape: {image.shape}")
            print(f"Mask shape: {mask.shape}")
            print(f"Unique mask values: {torch.unique(mask)}")
    else:
        print("Please create data directories:")
        print(f"  - {train_img_dir}")
        print(f"  - {train_mask_dir}")
