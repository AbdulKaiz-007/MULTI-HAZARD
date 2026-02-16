"""
Edge-Aware Feature Enhancement Module
Uses Canny edge detection to extract edge features and fuse with RGB channels
"""

import cv2
import numpy as np
import torch
import torch.nn as nn


class EdgeExtractor:
    """
    Extracts edge features using Canny edge detection
    and fuses them with RGB channels
    """
    
    def __init__(self, low_threshold=50, high_threshold=150, blur_kernel=5):
        """
        Args:
            low_threshold (int): Lower threshold for Canny edge detection
            high_threshold (int): Upper threshold for Canny edge detection
            blur_kernel (int): Gaussian blur kernel size (must be odd)
        """
        self.low_threshold = low_threshold
        self.high_threshold = high_threshold
        self.blur_kernel = blur_kernel
    
    def extract_edges(self, image):
        """
        Extract edges from a single image
        
        Args:
            image (np.ndarray): RGB image [H, W, 3] or [C, H, W]
        
        Returns:
            edge_map (np.ndarray): Binary edge map [H, W]
        """
        # Handle tensor input
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        
        # Convert from [C, H, W] to [H, W, C] if needed
        if image.shape[0] == 3 and len(image.shape) == 3:
            image = np.transpose(image, (1, 2, 0))
        
        # Ensure uint8 format
        if image.dtype != np.uint8:
            # Denormalize if normalized
            if image.max() <= 1.0:
                image = (image * 255).astype(np.uint8)
            else:
                image = image.astype(np.uint8)
        
        # Convert to grayscale
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray = image
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (self.blur_kernel, self.blur_kernel), 0)
        
        # Apply Canny edge detection
        edges = cv2.Canny(blurred, self.low_threshold, self.high_threshold)
        
        return edges
    
    def normalize_edges(self, edge_map):
        """
        Normalize edge map to [0, 1] range
        
        Args:
            edge_map (np.ndarray): Binary edge map
        
        Returns:
            normalized_edges (np.ndarray): Normalized edge map
        """
        return edge_map.astype(np.float32) / 255.0
    
    def fuse_with_rgb(self, rgb_image, edge_map):
        """
        Fuse edge map with RGB image to create 4-channel input
        
        Args:
            rgb_image (np.ndarray or torch.Tensor): RGB image [3, H, W] or [H, W, 3]
            edge_map (np.ndarray): Edge map [H, W]
        
        Returns:
            fused (torch.Tensor): 4-channel tensor [4, H, W]
        """
        # Convert to tensor if needed
        if isinstance(rgb_image, np.ndarray):
            rgb_image = torch.from_numpy(rgb_image).float()
        
        # Ensure [C, H, W] format
        if rgb_image.shape[-1] == 3:
            rgb_image = rgb_image.permute(2, 0, 1)
        
        # Convert edge map to tensor
        edge_tensor = torch.from_numpy(edge_map).float().unsqueeze(0)
        
        # Ensure same spatial dimensions
        if rgb_image.shape[1:] != edge_tensor.shape[1:]:
            edge_tensor = torch.nn.functional.interpolate(
                edge_tensor.unsqueeze(0), 
                size=rgb_image.shape[1:],
                mode='bilinear',
                align_corners=False
            ).squeeze(0)
        
        # Concatenate along channel dimension
        fused = torch.cat([rgb_image, edge_tensor], dim=0)
        
        return fused
    
    def __call__(self, image):
        """
        Complete edge extraction and fusion pipeline
        
        Args:
            image (np.ndarray or torch.Tensor): Input RGB image
        
        Returns:
            fused (torch.Tensor): 4-channel tensor [4, H, W]
        """
        # Extract edges
        edges = self.extract_edges(image)
        
        # Normalize edges
        edges_norm = self.normalize_edges(edges)
        
        # Fuse with RGB
        fused = self.fuse_with_rgb(image, edges_norm)
        
        return fused


class EdgeAwareConv(nn.Module):
    """
    Edge-aware convolutional layer that processes 4-channel input
    (RGB + Edge) and outputs standard feature maps
    """
    
    def __init__(self, out_channels=64):
        """
        Args:
            out_channels (int): Number of output channels
        """
        super(EdgeAwareConv, self).__init__()
        
        self.conv = nn.Sequential(
            nn.Conv2d(4, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
        """
        Args:
            x (torch.Tensor): 4-channel input [B, 4, H, W]
        
        Returns:
            features (torch.Tensor): Feature maps [B, out_channels, H, W]
        """
        return self.conv(x)


if __name__ == "__main__":
    print("Edge-Aware Feature Enhancement Module")
    print("=" * 50)
    
    # Test with random image
    test_image = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    
    # Create edge extractor
    edge_extractor = EdgeExtractor()
    
    # Extract edges
    edges = edge_extractor.extract_edges(test_image)
    print(f"Edge map shape: {edges.shape}")
    print(f"Edge map range: [{edges.min()}, {edges.max()}]")
    
    # Fuse with RGB
    fused = edge_extractor(test_image)
    print(f"Fused tensor shape: {fused.shape}")
    print(f"Expected: [4, 256, 256]")
    
    # Test edge-aware conv layer
    edge_conv = EdgeAwareConv(out_channels=64)
    output = edge_conv(fused.unsqueeze(0))
    print(f"Edge-aware conv output shape: {output.shape}")
    print(f"Expected: [1, 64, 256, 256]")
