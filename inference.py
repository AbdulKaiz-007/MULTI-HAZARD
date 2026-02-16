"""
Inference Script for Multi-Hazard Disaster Detection
Performs prediction on single images and saves results
"""

import os
import torch
import numpy as np
from PIL import Image
import argparse

from model import create_model
from edge_module import EdgeExtractor
from utils import (mask_to_rgb, overlay_mask_on_image, 
                  visualize_prediction, CLASS_NAMES)


class MultiHazardPredictor:
    """Predictor for multi-hazard segmentation"""
    
    def __init__(self, model_path, device='cuda', num_classes=4):
        """
        Args:
            model_path (str): Path to trained model checkpoint
            device (str): Device to run inference on
            num_classes (int): Number of classes
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.num_classes = num_classes
        
        # Create model
        self.model = create_model(
            num_classes=num_classes,
            encoder_name='resnet50',
            encoder_weights=None,  # We'll load trained weights
            in_channels=4
        )
        
        # Load checkpoint
        self.load_checkpoint(model_path)
        self.model = self.model.to(self.device)
        self.model.eval()
        
        # Create edge extractor
        self.edge_extractor = EdgeExtractor()
        
        print(f"Model loaded from {model_path}")
        print(f"Running on: {self.device}")
    
    def load_checkpoint(self, checkpoint_path):
        """Load model weights from checkpoint"""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        if 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)
    
    def preprocess_image(self, image_path, img_size=(256, 256)):
        """
        Preprocess image for inference
        
        Args:
            image_path (str): Path to input image
            img_size (tuple): Target image size
        
        Returns:
            image_tensor (torch.Tensor): Preprocessed image [1, 4, H, W]
            original_image (np.ndarray): Original image for visualization
        """
        # Load image
        image = Image.open(image_path).convert('RGB')
        original_image = np.array(image)
        
        # Resize
        image = image.resize((img_size[1], img_size[0]))
        image_np = np.array(image)
        
        # Extract edges and fuse
        fused = self.edge_extractor(image_np)
        
        # Add batch dimension
        image_tensor = fused.unsqueeze(0)
        
        return image_tensor, original_image
    
    def predict(self, image_path, img_size=(256, 256)):
        """
        Perform prediction on single image
        
        Args:
            image_path (str): Path to input image
            img_size (tuple): Target image size
        
        Returns:
            prediction (np.ndarray): Predicted mask [H, W]
            original_image (np.ndarray): Original image
        """
        # Preprocess
        image_tensor, original_image = self.preprocess_image(image_path, img_size)
        image_tensor = image_tensor.to(self.device)
        
        # Predict
        with torch.no_grad():
            output = self.model(image_tensor)
            prediction = torch.argmax(output, dim=1).squeeze(0)
        
        # Convert to numpy
        prediction = prediction.cpu().numpy()
        
        return prediction, original_image
    
    def predict_and_save(self, image_path, output_dir='outputs', 
                        img_size=(256, 256)):
        """
        Predict and save results
        
        Args:
            image_path (str): Path to input image
            output_dir (str): Directory to save outputs
            img_size (tuple): Target image size
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Get prediction
        prediction, original_image = self.predict(image_path, img_size)
        
        # Resize prediction to original size
        pred_resized = Image.fromarray(prediction.astype(np.uint8))
        pred_resized = pred_resized.resize(
            (original_image.shape[1], original_image.shape[0]),
            Image.NEAREST
        )
        pred_resized = np.array(pred_resized)
        
        # Get base filename
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        
        # Save mask
        mask_rgb = mask_to_rgb(pred_resized)
        mask_path = os.path.join(output_dir, f'{base_name}_mask.png')
        Image.fromarray(mask_rgb).save(mask_path)
        print(f"Mask saved to: {mask_path}")
        
        # Save overlay
        overlay = overlay_mask_on_image(original_image, pred_resized, alpha=0.5)
        overlay_path = os.path.join(output_dir, f'{base_name}_overlay.png')
        Image.fromarray(overlay).save(overlay_path)
        print(f"Overlay saved to: {overlay_path}")
        
        # Save visualization
        viz_path = os.path.join(output_dir, f'{base_name}_visualization.png')
        visualize_prediction(original_image, pred_resized, save_path=viz_path)
        
        # Print statistics
        self.print_statistics(pred_resized)
        
        return pred_resized, mask_rgb, overlay
    
    def print_statistics(self, prediction):
        """Print prediction statistics"""
        total_pixels = prediction.size
        
        print("\nPrediction Statistics:")
        print("-" * 50)
        
        for class_idx, class_name in enumerate(CLASS_NAMES):
            class_pixels = np.sum(prediction == class_idx)
            percentage = (class_pixels / total_pixels) * 100
            print(f"{class_name:15s}: {class_pixels:8d} pixels ({percentage:5.2f}%)")
        
        print("-" * 50)


def main():
    """Main inference function"""
    parser = argparse.ArgumentParser(
        description='Multi-Hazard Disaster Detection - Inference'
    )
    parser.add_argument('--image', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--model', type=str, default='checkpoints/best_model.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default='outputs',
                       help='Output directory')
    parser.add_argument('--size', type=int, default=256,
                       help='Image size for inference')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    print("Multi-Hazard Disaster Detection - Inference")
    print("=" * 70)
    print(f"Input image: {args.image}")
    print(f"Model: {args.model}")
    print(f"Output directory: {args.output}")
    print("=" * 70)
    
    # Create predictor
    predictor = MultiHazardPredictor(
        model_path=args.model,
        device=args.device,
        num_classes=4
    )
    
    # Run prediction
    predictor.predict_and_save(
        image_path=args.image,
        output_dir=args.output,
        img_size=(args.size, args.size)
    )
    
    print("\nInference completed!")


if __name__ == "__main__":
    # If no arguments provided, show usage example
    import sys
    if len(sys.argv) == 1:
        print("Multi-Hazard Disaster Detection - Inference")
        print("=" * 70)
        print("\nUsage:")
        print("  python inference.py --image <path_to_image> --model <path_to_model>")
        print("\nExample:")
        print("  python inference.py --image test.jpg --model checkpoints/best_model.pth")
        print("\nOptions:")
        print("  --image    Path to input image (required)")
        print("  --model    Path to model checkpoint (default: checkpoints/best_model.pth)")
        print("  --output   Output directory (default: outputs)")
        print("  --size     Image size for inference (default: 256)")
        print("  --device   Device to use (default: cuda)")
    else:
        main()
