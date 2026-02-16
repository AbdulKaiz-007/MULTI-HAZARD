"""
Multi-Hazard Segmentation Model
ResNet50 encoder + U-Net decoder with edge-aware input
"""

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp


class EdgeAwareMultiHazardModel(nn.Module):
    """
    Edge-aware multi-hazard segmentation model
    
    Architecture:
        - Input: 4 channels (RGB + Edge)
        - Encoder: ResNet50 (pretrained on ImageNet)
        - Decoder: U-Net
        - Output: 4 classes (background, flood, fire, damage)
    """
    
    def __init__(self, num_classes=4, encoder_name='resnet50', 
                 encoder_weights='imagenet', in_channels=4):
        """
        Args:
            num_classes (int): Number of segmentation classes
            encoder_name (str): Encoder architecture
            encoder_weights (str): Pretrained weights ('imagenet' or None)
            in_channels (int): Number of input channels (4 for RGB+Edge)
        """
        super(EdgeAwareMultiHazardModel, self).__init__()
        
        self.num_classes = num_classes
        self.in_channels = in_channels
        
        # Create U-Net model with ResNet50 encoder
        # Note: We'll modify the first conv layer to accept 4 channels
        self.model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,  # Temporarily set to 3 for pretrained weights
            classes=num_classes,
            activation=None  # We'll apply softmax/argmax separately
        )
        
        # Modify first convolutional layer to accept 4 channels
        if in_channels != 3:
            self._modify_first_conv(in_channels, encoder_weights)
    
    def _modify_first_conv(self, in_channels, encoder_weights):
        """
        Modify the first convolutional layer to accept custom input channels
        while preserving pretrained weights for RGB channels
        """
        # Get the first conv layer
        first_conv = self.model.encoder.conv1
        
        # Create new conv layer with desired input channels
        new_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=first_conv.out_channels,
            kernel_size=first_conv.kernel_size,
            stride=first_conv.stride,
            padding=first_conv.padding,
            bias=first_conv.bias is not None
        )
        
        # Copy pretrained weights for RGB channels
        if encoder_weights:
            with torch.no_grad():
                # Copy RGB weights
                new_conv.weight[:, :3, :, :] = first_conv.weight
                
                # Initialize edge channel weights (average of RGB weights)
                if in_channels > 3:
                    new_conv.weight[:, 3:, :, :] = first_conv.weight.mean(dim=1, keepdim=True)
                
                # Copy bias if exists
                if first_conv.bias is not None:
                    new_conv.bias = first_conv.bias
        
        # Replace the first conv layer
        self.model.encoder.conv1 = new_conv
    
    def forward(self, x):
        """
        Forward pass
        
        Args:
            x (torch.Tensor): Input tensor [B, 4, H, W]
        
        Returns:
            output (torch.Tensor): Segmentation logits [B, num_classes, H, W]
        """
        return self.model(x)
    
    def predict(self, x):
        """
        Prediction with argmax
        
        Args:
            x (torch.Tensor): Input tensor [B, 4, H, W]
        
        Returns:
            predictions (torch.Tensor): Class predictions [B, H, W]
        """
        logits = self.forward(x)
        predictions = torch.argmax(logits, dim=1)
        return predictions


def create_model(num_classes=4, encoder_name='resnet50', 
                encoder_weights='imagenet', in_channels=4):
    """
    Factory function to create the model
    
    Args:
        num_classes (int): Number of segmentation classes
        encoder_name (str): Encoder architecture
        encoder_weights (str): Pretrained weights
        in_channels (int): Number of input channels
    
    Returns:
        model (EdgeAwareMultiHazardModel): Initialized model
    """
    model = EdgeAwareMultiHazardModel(
        num_classes=num_classes,
        encoder_name=encoder_name,
        encoder_weights=encoder_weights,
        in_channels=in_channels
    )
    return model


def count_parameters(model):
    """Count trainable parameters"""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


if __name__ == "__main__":
    print("Multi-Hazard Segmentation Model")
    print("=" * 50)
    
    # Create model
    model = create_model(
        num_classes=4,
        encoder_name='resnet50',
        encoder_weights='imagenet',
        in_channels=4
    )
    
    print(f"Model created successfully!")
    print(f"Total parameters: {count_parameters(model):,}")
    
    # Test forward pass
    batch_size = 2
    test_input = torch.randn(batch_size, 4, 256, 256)
    
    model.eval()
    with torch.no_grad():
        output = model(test_input)
        predictions = model.predict(test_input)
    
    print(f"\nTest forward pass:")
    print(f"Input shape: {test_input.shape}")
    print(f"Output logits shape: {output.shape}")
    print(f"Predictions shape: {predictions.shape}")
    print(f"Unique predicted classes: {torch.unique(predictions)}")
    
    # Model summary
    print(f"\nModel Architecture:")
    print(f"  - Encoder: ResNet50 (pretrained on ImageNet)")
    print(f"  - Decoder: U-Net")
    print(f"  - Input channels: 4 (RGB + Edge)")
    print(f"  - Output classes: 4 (background, flood, fire, damage)")
