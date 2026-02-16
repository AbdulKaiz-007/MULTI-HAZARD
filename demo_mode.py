"""
Demo Mode Script - Test the system without training
Creates synthetic predictions for demonstration purposes
"""

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import sys

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8')

from utils import mask_to_rgb, overlay_mask_on_image, CLASS_NAMES
from cascading_logic import CascadingRiskAnalyzer


def create_demo_prediction(image_path, output_dir='demo_outputs'):
    """
    Create a synthetic prediction for demonstration
    
    Args:
        image_path (str): Path to input image
        output_dir (str): Directory to save outputs
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load image
    image = Image.open(image_path).convert('RGB')
    image_np = np.array(image)
    h, w = image_np.shape[:2]
    
    print(f"Creating demo prediction for image: {image_path}")
    print(f"Image size: {w}x{h}")
    
    # Create synthetic prediction mask
    prediction = np.zeros((h, w), dtype=np.uint8)
    
    # Add some synthetic hazard regions
    # Flood region (blue) - bottom left
    flood_h, flood_w = h // 3, w // 3
    prediction[h-flood_h:h, 0:flood_w] = 1
    
    # Fire region (red) - top right
    fire_h, fire_w = h // 4, w // 4
    prediction[0:fire_h, w-fire_w:w] = 2
    
    # Damage region (yellow) - center
    damage_h, damage_w = h // 5, w // 5
    center_y, center_x = h // 2, w // 2
    prediction[center_y-damage_h//2:center_y+damage_h//2,
               center_x-damage_w//2:center_x+damage_w//2] = 3
    
    # Generate visualizations
    mask_rgb = mask_to_rgb(prediction)
    overlay = overlay_mask_on_image(image_np, prediction, alpha=0.5)
    
    # Save outputs
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    
    mask_path = os.path.join(output_dir, f'{base_name}_demo_mask.png')
    Image.fromarray(mask_rgb).save(mask_path)
    print(f"[OK] Mask saved: {mask_path}")
    
    overlay_path = os.path.join(output_dir, f'{base_name}_demo_overlay.png')
    Image.fromarray(overlay).save(overlay_path)
    print(f"[OK] Overlay saved: {overlay_path}")
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(image_np)
    axes[0].set_title('Original Image', fontsize=14, fontweight='bold')
    axes[0].axis('off')
    
    axes[1].imshow(mask_rgb)
    axes[1].set_title('Demo Prediction Mask', fontsize=14, fontweight='bold')
    axes[1].axis('off')
    
    axes[2].imshow(overlay)
    axes[2].set_title('Demo Overlay', fontsize=14, fontweight='bold')
    axes[2].axis('off')
    
    plt.tight_layout()
    viz_path = os.path.join(output_dir, f'{base_name}_demo_visualization.png')
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[OK] Visualization saved: {viz_path}")
    
    # Print statistics
    print("\n" + "="*60)
    print("DEMO PREDICTION STATISTICS")
    print("="*60)
    
    total_pixels = prediction.size
    for class_idx, class_name in enumerate(CLASS_NAMES):
        class_pixels = np.sum(prediction == class_idx)
        percentage = (class_pixels / total_pixels) * 100
        print(f"{class_name:15s}: {class_pixels:8d} pixels ({percentage:5.2f}%)")
    
    # Run risk analysis
    print("\n" + "="*60)
    print("DEMO RISK ANALYSIS")
    print("="*60)
    
    analyzer = CascadingRiskAnalyzer(image_size=prediction.shape)
    analysis = analyzer.analyze(prediction, rainfall_intensity=60)
    report = analyzer.generate_report(analysis)
    print(report)
    
    # Save report
    report_path = os.path.join(output_dir, f'{base_name}_demo_report.txt')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"\n[OK] Report saved: {report_path}")
    
    return prediction, mask_rgb, overlay, analysis


def create_sample_image(output_path='sample_satellite.jpg', size=(512, 512)):
    """
    Create a sample satellite-like image for testing
    
    Args:
        output_path (str): Path to save the sample image
        size (tuple): Image dimensions
    """
    # Create a synthetic satellite-like image
    np.random.seed(42)
    
    # Base terrain (greenish-brown)
    image = np.random.randint(80, 120, (*size, 3), dtype=np.uint8)
    image[:, :, 1] += 40  # More green
    
    # Add some texture
    noise = np.random.randint(-20, 20, size, dtype=np.int16)
    for c in range(3):
        image[:, :, c] = np.clip(image[:, :, c] + noise, 0, 255)
    
    # Add some features (roads, buildings)
    # Horizontal road
    image[size[0]//2-5:size[0]//2+5, :] = [100, 100, 100]
    
    # Vertical road
    image[:, size[1]//2-5:size[1]//2+5] = [100, 100, 100]
    
    # Some building-like rectangles
    for i in range(5):
        x = np.random.randint(50, size[1]-100)
        y = np.random.randint(50, size[0]-100)
        w = np.random.randint(30, 60)
        h = np.random.randint(30, 60)
        image[y:y+h, x:x+w] = [150, 150, 150]
    
    # Save
    Image.fromarray(image).save(output_path)
    print(f"[OK] Sample image created: {output_path}")
    
    return output_path


def main():
    """Main demo function"""
    print("="*60)
    print("MULTI-HAZARD AI - DEMO MODE")
    print("="*60)
    print("\nThis demo creates synthetic predictions without a trained model.")
    print("Useful for testing the system and understanding outputs.\n")
    
    # Check if sample image exists
    sample_image = 'sample_satellite.jpg'
    
    if not os.path.exists(sample_image):
        print("Creating sample satellite image...")
        create_sample_image(sample_image)
        print()
    
    # Create demo prediction
    print("Generating demo prediction...")
    create_demo_prediction(sample_image)
    
    print("\n" + "="*60)
    print("DEMO COMPLETED!")
    print("="*60)
    print("\nOutputs saved to 'demo_outputs/' directory:")
    print("  - *_demo_mask.png - Segmentation mask")
    print("  - *_demo_overlay.png - Overlay on original")
    print("  - *_demo_visualization.png - Complete visualization")
    print("  - *_demo_report.txt - Risk analysis report")
    print("\n[!] Note: These are SYNTHETIC predictions for demonstration only.")
    print("    Train a real model for actual disaster detection!")


if __name__ == "__main__":
    main()
