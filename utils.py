"""
Utility Functions for Multi-Hazard Detection
Includes IoU calculation, visualization, and helper functions
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import cv2


# Class names and colors
CLASS_NAMES = ['Background', 'Flood', 'Fire', 'Damage']
CLASS_COLORS = [
    [0, 0, 0],        # Background - Black
    [0, 0, 255],      # Flood - Blue
    [255, 0, 0],      # Fire - Red
    [255, 255, 0]     # Damage - Yellow
]


def calculate_iou(pred, target, num_classes=4, smooth=1e-6):
    """
    Calculate Intersection over Union (IoU) for each class
    
    Args:
        pred (torch.Tensor): Predicted masks [B, H, W] or [H, W]
        target (torch.Tensor): Ground truth masks [B, H, W] or [H, W]
        num_classes (int): Number of classes
        smooth (float): Smoothing factor to avoid division by zero
    
    Returns:
        iou_per_class (dict): IoU for each class
        mean_iou (float): Mean IoU across all classes
    """
    # Ensure tensors are on CPU
    pred = pred.cpu()
    target = target.cpu()
    
    # Flatten if batched
    if len(pred.shape) == 3:
        pred = pred.view(-1)
        target = target.view(-1)
    else:
        pred = pred.view(-1)
        target = target.view(-1)
    
    iou_per_class = {}
    ious = []
    
    for cls in range(num_classes):
        pred_cls = (pred == cls)
        target_cls = (target == cls)
        
        intersection = (pred_cls & target_cls).sum().float()
        union = (pred_cls | target_cls).sum().float()
        
        iou = (intersection + smooth) / (union + smooth)
        iou_per_class[CLASS_NAMES[cls]] = iou.item()
        ious.append(iou.item())
    
    mean_iou = np.mean(ious)
    
    return iou_per_class, mean_iou


def calculate_pixel_accuracy(pred, target):
    """
    Calculate pixel-wise accuracy
    
    Args:
        pred (torch.Tensor): Predicted masks
        target (torch.Tensor): Ground truth masks
    
    Returns:
        accuracy (float): Pixel accuracy
    """
    pred = pred.cpu().view(-1)
    target = target.cpu().view(-1)
    
    correct = (pred == target).sum().float()
    total = target.numel()
    
    accuracy = (correct / total).item()
    return accuracy


def mask_to_rgb(mask, colors=CLASS_COLORS):
    """
    Convert class mask to RGB visualization
    
    Args:
        mask (np.ndarray or torch.Tensor): Class mask [H, W]
        colors (list): List of RGB colors for each class
    
    Returns:
        rgb_mask (np.ndarray): RGB mask [H, W, 3]
    """
    if isinstance(mask, torch.Tensor):
        mask = mask.cpu().numpy()
    
    h, w = mask.shape
    rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)
    
    for cls_idx, color in enumerate(colors):
        rgb_mask[mask == cls_idx] = color
    
    return rgb_mask


def overlay_mask_on_image(image, mask, alpha=0.5):
    """
    Overlay segmentation mask on original image
    
    Args:
        image (np.ndarray): Original image [H, W, 3]
        mask (np.ndarray): Class mask [H, W]
        alpha (float): Transparency factor
    
    Returns:
        overlay (np.ndarray): Overlaid image
    """
    # Convert mask to RGB
    mask_rgb = mask_to_rgb(mask)
    
    # Ensure image is uint8
    if image.dtype != np.uint8:
        if image.max() <= 1.0:
            image = (image * 255).astype(np.uint8)
        else:
            image = image.astype(np.uint8)
    
    # Resize mask if needed
    if image.shape[:2] != mask_rgb.shape[:2]:
        mask_rgb = cv2.resize(mask_rgb, (image.shape[1], image.shape[0]))
    
    # Blend images
    overlay = cv2.addWeighted(image, 1 - alpha, mask_rgb, alpha, 0)
    
    return overlay


def visualize_prediction(image, pred_mask, gt_mask=None, save_path=None):
    """
    Visualize prediction results
    
    Args:
        image (np.ndarray): Original image
        pred_mask (np.ndarray): Predicted mask
        gt_mask (np.ndarray): Ground truth mask (optional)
        save_path (str): Path to save visualization
    """
    if gt_mask is not None:
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Original image
        axes[0, 0].imshow(image)
        axes[0, 0].set_title('Original Image')
        axes[0, 0].axis('off')
        
        # Ground truth
        axes[0, 1].imshow(mask_to_rgb(gt_mask))
        axes[0, 1].set_title('Ground Truth')
        axes[0, 1].axis('off')
        
        # Prediction
        axes[1, 0].imshow(mask_to_rgb(pred_mask))
        axes[1, 0].set_title('Prediction')
        axes[1, 0].axis('off')
        
        # Overlay
        overlay = overlay_mask_on_image(image, pred_mask)
        axes[1, 1].imshow(overlay)
        axes[1, 1].set_title('Prediction Overlay')
        axes[1, 1].axis('off')
    else:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original image
        axes[0].imshow(image)
        axes[0].set_title('Original Image')
        axes[0].axis('off')
        
        # Prediction
        axes[1].imshow(mask_to_rgb(pred_mask))
        axes[1].set_title('Prediction')
        axes[1].axis('off')
        
        # Overlay
        overlay = overlay_mask_on_image(image, pred_mask)
        axes[2].imshow(overlay)
        axes[2].set_title('Prediction Overlay')
        axes[2].axis('off')
    
    # Add legend
    legend_elements = [plt.Rectangle((0, 0), 1, 1, fc=np.array(color)/255.0, 
                                    label=name) 
                      for name, color in zip(CLASS_NAMES, CLASS_COLORS)]
    fig.legend(handles=legend_elements, loc='lower center', ncol=4)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    else:
        plt.show()
    
    plt.close()


def denormalize_image(image, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    """
    Denormalize image tensor for visualization
    
    Args:
        image (torch.Tensor or np.ndarray): Normalized image [C, H, W]
        mean (list): Mean used for normalization
        std (list): Std used for normalization
    
    Returns:
        denorm_image (np.ndarray): Denormalized image [H, W, C]
    """
    if isinstance(image, torch.Tensor):
        image = image.cpu().numpy()
    
    # Convert from [C, H, W] to [H, W, C]
    if image.shape[0] == 3 or image.shape[0] == 4:
        image = np.transpose(image[:3], (1, 2, 0))
    
    # Denormalize
    mean = np.array(mean)
    std = np.array(std)
    denorm_image = (image * std + mean) * 255.0
    denorm_image = np.clip(denorm_image, 0, 255).astype(np.uint8)
    
    return denorm_image


def save_checkpoint(model, optimizer, epoch, loss, save_path):
    """
    Save model checkpoint
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        epoch (int): Current epoch
        loss (float): Current loss
        save_path (str): Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss
    }
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")


def load_checkpoint(model, optimizer, checkpoint_path):
    """
    Load model checkpoint
    
    Args:
        model: PyTorch model
        optimizer: Optimizer
        checkpoint_path (str): Path to checkpoint
    
    Returns:
        epoch (int): Saved epoch
        loss (float): Saved loss
    """
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    
    print(f"Checkpoint loaded from {checkpoint_path}")
    print(f"Resuming from epoch {epoch} with loss {loss:.4f}")
    
    return epoch, loss


if __name__ == "__main__":
    print("Utility Functions Test")
    print("=" * 50)
    
    # Test IoU calculation
    pred = torch.randint(0, 4, (256, 256))
    target = torch.randint(0, 4, (256, 256))
    
    iou_per_class, mean_iou = calculate_iou(pred, target)
    print("\nIoU per class:")
    for cls_name, iou_val in iou_per_class.items():
        print(f"  {cls_name}: {iou_val:.4f}")
    print(f"Mean IoU: {mean_iou:.4f}")
    
    # Test pixel accuracy
    accuracy = calculate_pixel_accuracy(pred, target)
    print(f"\nPixel Accuracy: {accuracy:.4f}")
    
    # Test mask visualization
    test_mask = np.random.randint(0, 4, (256, 256))
    rgb_mask = mask_to_rgb(test_mask)
    print(f"\nRGB mask shape: {rgb_mask.shape}")
