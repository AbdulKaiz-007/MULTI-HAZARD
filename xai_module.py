"""
Explainability Module using Grad-CAM
Generates visual explanations for model predictions
"""

import torch
import torch.nn.functional as F
import numpy as np
import cv2
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import SemanticSegmentationTarget
import matplotlib.pyplot as plt
from PIL import Image

from model import create_model
from edge_module import EdgeExtractor
from utils import CLASS_NAMES, CLASS_COLORS


class SegmentationModelWrapper(torch.nn.Module):
    """Wrapper for segmentation model to work with Grad-CAM"""
    
    def __init__(self, model):
        super(SegmentationModelWrapper, self).__init__()
        self.model = model
    
    def forward(self, x):
        return self.model(x)


class MultiHazardExplainer:
    """Explainability module for multi-hazard segmentation"""
    
    def __init__(self, model, device='cuda'):
        """
        Args:
            model: Trained segmentation model
            device (str): Device to run on
        """
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        
        # Wrap model
        self.wrapped_model = SegmentationModelWrapper(model)
        
        # Target layer for Grad-CAM (last decoder layer)
        # For U-Net in segmentation_models_pytorch, use decoder
        self.target_layers = [self.model.model.decoder.blocks[-1]]
        
        # Create edge extractor
        self.edge_extractor = EdgeExtractor()
    
    def preprocess_image(self, image_path, img_size=(256, 256)):
        """
        Preprocess image for Grad-CAM
        
        Args:
            image_path (str): Path to input image
            img_size (tuple): Target size
        
        Returns:
            image_tensor: Preprocessed tensor
            rgb_image: Normalized RGB for visualization
            original_image: Original image
        """
        # Load image
        image = Image.open(image_path).convert('RGB')
        original_image = np.array(image)
        
        # Resize
        image = image.resize((img_size[1], img_size[0]))
        image_np = np.array(image)
        
        # Normalize for visualization (0-1 range)
        rgb_image = image_np.astype(np.float32) / 255.0
        
        # Extract edges and fuse
        fused = self.edge_extractor(image_np)
        
        # Add batch dimension
        image_tensor = fused.unsqueeze(0).to(self.device)
        
        return image_tensor, rgb_image, original_image
    
    def generate_gradcam(self, image_path, target_class=None, 
                        img_size=(256, 256)):
        """
        Generate Grad-CAM heatmap for specific class
        
        Args:
            image_path (str): Path to input image
            target_class (int): Target class index (None for predicted class)
            img_size (tuple): Image size
        
        Returns:
            cam_image: Grad-CAM visualization
            prediction: Model prediction
        """
        # Preprocess
        image_tensor, rgb_image, original_image = self.preprocess_image(
            image_path, img_size
        )
        
        # Get prediction
        with torch.no_grad():
            output = self.model(image_tensor)
            prediction = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
        
        # Determine target class
        if target_class is None:
            # Use most common predicted class (excluding background)
            unique, counts = np.unique(prediction, return_counts=True)
            class_counts = dict(zip(unique, counts))
            # Remove background
            if 0 in class_counts:
                del class_counts[0]
            if class_counts:
                target_class = max(class_counts, key=class_counts.get)
            else:
                target_class = 1  # Default to flood
        
        print(f"Generating Grad-CAM for class: {CLASS_NAMES[target_class]}")
        
        # Create semantic segmentation target
        targets = [SemanticSegmentationTarget(target_class, 
                                             prediction == target_class)]
        
        # Generate Grad-CAM
        with GradCAM(model=self.wrapped_model, 
                    target_layers=self.target_layers) as cam:
            grayscale_cam = cam(input_tensor=image_tensor, targets=targets)[0]
        
        # Create visualization
        cam_image = show_cam_on_image(rgb_image, grayscale_cam, 
                                     use_rgb=True)
        
        return cam_image, prediction, target_class
    
    def generate_multi_class_gradcam(self, image_path, img_size=(256, 256),
                                    output_path=None):
        """
        Generate Grad-CAM for all detected hazard classes
        
        Args:
            image_path (str): Path to input image
            img_size (tuple): Image size
            output_path (str): Path to save visualization
        
        Returns:
            results (dict): Dictionary of class-wise Grad-CAM results
        """
        # Preprocess
        image_tensor, rgb_image, original_image = self.preprocess_image(
            image_path, img_size
        )
        
        # Get prediction
        with torch.no_grad():
            output = self.model(image_tensor)
            prediction = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
        
        # Find detected classes (excluding background)
        detected_classes = np.unique(prediction)
        detected_classes = [c for c in detected_classes if c != 0]
        
        if not detected_classes:
            print("No hazards detected in image")
            return None
        
        print(f"Detected classes: {[CLASS_NAMES[c] for c in detected_classes]}")
        
        # Generate Grad-CAM for each class
        results = {}
        
        for class_idx in detected_classes:
            targets = [SemanticSegmentationTarget(class_idx, 
                                                 prediction == class_idx)]
            
            with GradCAM(model=self.wrapped_model, 
                        target_layers=self.target_layers) as cam:
                grayscale_cam = cam(input_tensor=image_tensor, targets=targets)[0]
            
            cam_image = show_cam_on_image(rgb_image, grayscale_cam, use_rgb=True)
            
            results[CLASS_NAMES[class_idx]] = {
                'cam_image': cam_image,
                'grayscale_cam': grayscale_cam
            }
        
        # Create visualization
        self.visualize_multi_class_gradcam(
            original_image, prediction, results, output_path
        )
        
        return results
    
    def visualize_multi_class_gradcam(self, original_image, prediction, 
                                     results, output_path=None):
        """
        Visualize Grad-CAM results for multiple classes
        
        Args:
            original_image: Original image
            prediction: Model prediction
            results: Dictionary of Grad-CAM results
            output_path: Path to save visualization
        """
        num_classes = len(results)
        
        fig, axes = plt.subplots(2, num_classes + 1, 
                                figsize=(5 * (num_classes + 1), 10))
        
        if num_classes == 1:
            axes = axes.reshape(2, -1)
        
        # Original image
        axes[0, 0].imshow(original_image)
        axes[0, 0].set_title('Original Image', fontsize=12, fontweight='bold')
        axes[0, 0].axis('off')
        
        # Prediction
        from utils import mask_to_rgb
        pred_rgb = mask_to_rgb(prediction)
        axes[1, 0].imshow(pred_rgb)
        axes[1, 0].set_title('Prediction', fontsize=12, fontweight='bold')
        axes[1, 0].axis('off')
        
        # Grad-CAM for each class
        for idx, (class_name, result) in enumerate(results.items(), start=1):
            # Heatmap
            axes[0, idx].imshow(result['grayscale_cam'], cmap='jet')
            axes[0, idx].set_title(f'{class_name}\nHeatmap', 
                                  fontsize=12, fontweight='bold')
            axes[0, idx].axis('off')
            
            # Overlay
            axes[1, idx].imshow(result['cam_image'])
            axes[1, idx].set_title(f'{class_name}\nGrad-CAM', 
                                  fontsize=12, fontweight='bold')
            axes[1, idx].axis('off')
        
        plt.tight_layout()
        
        if output_path:
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Grad-CAM visualization saved to: {output_path}")
        else:
            plt.show()
        
        plt.close()


def main():
    """Example usage"""
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Multi-Hazard Disaster Detection - Grad-CAM Explainability'
    )
    parser.add_argument('--image', type=str, required=True,
                       help='Path to input image')
    parser.add_argument('--model', type=str, default='checkpoints/best_model.pth',
                       help='Path to model checkpoint')
    parser.add_argument('--output', type=str, default='gradcam_output.png',
                       help='Output path for Grad-CAM visualization')
    parser.add_argument('--class', type=int, default=None,
                       help='Target class (None for all detected classes)')
    parser.add_argument('--device', type=str, default='cuda',
                       help='Device (cuda/cpu)')
    
    args = parser.parse_args()
    
    print("Multi-Hazard Disaster Detection - Grad-CAM")
    print("=" * 70)
    
    # Load model
    device = args.device if torch.cuda.is_available() else 'cpu'
    model = create_model(num_classes=4, in_channels=4)
    
    checkpoint = torch.load(args.model, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    # Create explainer
    explainer = MultiHazardExplainer(model, device=device)
    
    # Generate Grad-CAM
    if args.__dict__['class'] is not None:
        cam_image, prediction, target_class = explainer.generate_gradcam(
            args.image, target_class=args.__dict__['class']
        )
        
        # Save result
        Image.fromarray(cam_image).save(args.output)
        print(f"Grad-CAM saved to: {args.output}")
    else:
        explainer.generate_multi_class_gradcam(
            args.image, output_path=args.output
        )


if __name__ == "__main__":
    import sys
    if len(sys.argv) == 1:
        print("Multi-Hazard Disaster Detection - Grad-CAM")
        print("=" * 70)
        print("\nUsage:")
        print("  python xai_module.py --image <path> --model <path>")
        print("\nExample:")
        print("  python xai_module.py --image test.jpg --model checkpoints/best_model.pth")
    else:
        main()
