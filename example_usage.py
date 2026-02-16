"""
Complete End-to-End Example
Demonstrates the full pipeline from training to inference to explainability
"""

import os
import numpy as np
import torch
from PIL import Image

# Import all modules
from data_loader import create_dataloaders
from edge_module import EdgeExtractor
from model import create_model
from train import Trainer
from inference import MultiHazardPredictor
from xai_module import MultiHazardExplainer
from cascading_logic import CascadingRiskAnalyzer
import torch.nn as nn
import torch.optim as optim


def example_training():
    """Example: Training the model"""
    print("\n" + "="*70)
    print("EXAMPLE 1: TRAINING")
    print("="*70)
    
    # Configuration
    config = {
        'train_img_dir': 'data/train/images',
        'train_mask_dir': 'data/train/masks',
        'val_img_dir': 'data/val/images',
        'val_mask_dir': 'data/val/masks',
        'num_classes': 4,
        'batch_size': 4,
        'num_epochs': 5,  # Small number for demo
        'learning_rate': 1e-4,
        'img_size': (256, 256),
        'device': 'cuda' if torch.cuda.is_available() else 'cpu'
    }
    
    print(f"Device: {config['device']}")
    
    # Check if data exists
    if not os.path.exists(config['train_img_dir']):
        print("⚠️  Training data not found. Please prepare your dataset first.")
        print("   See README.md for dataset structure.")
        return
    
    # Create data loaders
    train_loader, val_loader = create_dataloaders(
        config['train_img_dir'],
        config['train_mask_dir'],
        config['val_img_dir'],
        config['val_mask_dir'],
        batch_size=config['batch_size'],
        img_size=config['img_size']
    )
    
    # Create model
    model = create_model(num_classes=config['num_classes'], in_channels=4)
    
    # Create edge extractor
    edge_extractor = EdgeExtractor()
    
    # Setup training
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    
    # Create trainer
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        device=config['device'],
        edge_extractor=edge_extractor
    )
    
    # Train
    trainer.train(num_epochs=config['num_epochs'], save_dir='checkpoints')
    
    print("✅ Training completed!")


def example_inference():
    """Example: Running inference on a single image"""
    print("\n" + "="*70)
    print("EXAMPLE 2: INFERENCE")
    print("="*70)
    
    # Check if model exists
    model_path = 'checkpoints/best_model.pth'
    if not os.path.exists(model_path):
        print("⚠️  Model checkpoint not found. Please train the model first.")
        return
    
    # Check if test image exists
    test_image = 'test_image.jpg'
    if not os.path.exists(test_image):
        print("⚠️  Test image not found. Please provide a test image.")
        return
    
    # Create predictor
    predictor = MultiHazardPredictor(
        model_path=model_path,
        device='cuda' if torch.cuda.is_available() else 'cpu'
    )
    
    # Run prediction
    predictor.predict_and_save(
        image_path=test_image,
        output_dir='outputs'
    )
    
    print("✅ Inference completed! Check 'outputs/' directory.")


def example_gradcam():
    """Example: Generating Grad-CAM explanations"""
    print("\n" + "="*70)
    print("EXAMPLE 3: GRAD-CAM EXPLAINABILITY")
    print("="*70)
    
    # Check if model exists
    model_path = 'checkpoints/best_model.pth'
    if not os.path.exists(model_path):
        print("⚠️  Model checkpoint not found. Please train the model first.")
        return
    
    # Check if test image exists
    test_image = 'test_image.jpg'
    if not os.path.exists(test_image):
        print("⚠️  Test image not found. Please provide a test image.")
        return
    
    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = create_model(num_classes=4, in_channels=4)
    
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    # Create explainer
    explainer = MultiHazardExplainer(model, device=device)
    
    # Generate Grad-CAM for all detected classes
    explainer.generate_multi_class_gradcam(
        test_image,
        output_path='gradcam_visualization.png'
    )
    
    print("✅ Grad-CAM completed! Check 'gradcam_visualization.png'.")


def example_risk_analysis():
    """Example: Cascading risk analysis"""
    print("\n" + "="*70)
    print("EXAMPLE 4: CASCADING RISK ANALYSIS")
    print("="*70)
    
    # Create sample prediction mask for demonstration
    # In practice, this would come from model inference
    prediction_mask = np.zeros((256, 256), dtype=np.uint8)
    
    # Simulate detected hazards
    prediction_mask[50:120, 50:180] = 1   # Flood region
    prediction_mask[130:200, 80:220] = 2  # Fire region
    prediction_mask[210:250, 100:200] = 3 # Damage region
    
    print("Using simulated prediction mask for demonstration...")
    
    # Create analyzer
    analyzer = CascadingRiskAnalyzer(image_size=(256, 256))
    
    # Perform analysis
    analysis = analyzer.analyze(
        prediction_mask,
        rainfall_intensity=60  # Optional: rainfall data
    )
    
    # Generate and print report
    report = analyzer.generate_report(analysis)
    print(report)
    
    print("\n✅ Risk analysis completed!")


def example_complete_pipeline():
    """Example: Complete pipeline from image to risk assessment"""
    print("\n" + "="*70)
    print("EXAMPLE 5: COMPLETE PIPELINE")
    print("="*70)
    
    # Check prerequisites
    model_path = 'checkpoints/best_model.pth'
    test_image = 'test_image.jpg'
    
    if not os.path.exists(model_path):
        print("⚠️  Model checkpoint not found.")
        return
    
    if not os.path.exists(test_image):
        print("⚠️  Test image not found.")
        return
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Step 1: Inference
    print("\n📍 Step 1: Running inference...")
    predictor = MultiHazardPredictor(model_path=model_path, device=device)
    prediction, original_image = predictor.predict(test_image)
    
    # Step 2: Risk Analysis
    print("\n📍 Step 2: Analyzing cascading risks...")
    analyzer = CascadingRiskAnalyzer(image_size=prediction.shape)
    analysis = analyzer.analyze(prediction)
    report = analyzer.generate_report(analysis)
    print(report)
    
    # Step 3: Grad-CAM
    print("\n📍 Step 3: Generating Grad-CAM explanations...")
    model = create_model(num_classes=4, in_channels=4)
    checkpoint = torch.load(model_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    explainer = MultiHazardExplainer(model, device=device)
    explainer.generate_multi_class_gradcam(
        test_image,
        output_path='complete_pipeline_gradcam.png'
    )
    
    print("\n✅ Complete pipeline executed successfully!")
    print("   - Prediction saved to 'outputs/'")
    print("   - Grad-CAM saved to 'complete_pipeline_gradcam.png'")
    print("   - Risk analysis displayed above")


def main():
    """Main function with menu"""
    print("\n" + "="*70)
    print("MULTI-HAZARD DISASTER DETECTION - EXAMPLE USAGE")
    print("="*70)
    
    print("\nAvailable Examples:")
    print("  1. Training")
    print("  2. Inference")
    print("  3. Grad-CAM Explainability")
    print("  4. Cascading Risk Analysis")
    print("  5. Complete Pipeline")
    print("  6. Run All Examples")
    
    choice = input("\nSelect example (1-6): ").strip()
    
    if choice == '1':
        example_training()
    elif choice == '2':
        example_inference()
    elif choice == '3':
        example_gradcam()
    elif choice == '4':
        example_risk_analysis()
    elif choice == '5':
        example_complete_pipeline()
    elif choice == '6':
        print("\n🚀 Running all examples...")
        example_risk_analysis()  # This one works without dependencies
        # Uncomment others if you have the required data
        # example_training()
        # example_inference()
        # example_gradcam()
        # example_complete_pipeline()
    else:
        print("Invalid choice. Please select 1-6.")


if __name__ == "__main__":
    main()
