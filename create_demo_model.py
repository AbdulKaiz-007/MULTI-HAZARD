"""
Create a demo pre-trained model for testing the Streamlit app
This creates a model with random weights for demonstration purposes
"""

import torch
import os
from model import create_model
from utils import save_checkpoint

def create_demo_model(output_path='checkpoints/demo_model.pth'):
    """
    Create a demo model with initialized weights
    
    Args:
        output_path: Path to save the demo model
    """
    print("Creating demo model...")
    
    # Create model
    model = create_model(
        num_classes=4,
        encoder_name='resnet50',
        encoder_weights='imagenet',
        in_channels=4
    )
    
    # Create dummy optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    
    # Create checkpoint directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save model
    checkpoint = {
        'epoch': 0,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': 0.5,
        'note': 'Demo model with ImageNet pretrained encoder - not trained on disaster data'
    }
    
    torch.save(checkpoint, output_path)
    
    print(f"[OK] Demo model created: {output_path}")
    print("\nNote: This is a DEMO model with ImageNet pretrained weights.")
    print("      It has NOT been trained on disaster detection data.")
    print("      Predictions will be random/incorrect.")
    print("\nTo get real predictions:")
    print("  1. Prepare dataset: python scripts/prepare_dataset.py --mode sample")
    print("  2. Train model: python train.py")
    print("  3. Use trained model: checkpoints/best_model.pth")
    
    return output_path


if __name__ == "__main__":
    create_demo_model()
