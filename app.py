"""
Streamlit Web Application for Multi-Hazard Disaster Detection
Interactive interface for training, inference, explainability, and risk analysis
"""

import streamlit as st
import os
import sys
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import io
import time

# Import project modules
try:
    from model import create_model
    from edge_module import EdgeExtractor
    from inference import MultiHazardPredictor
    from cascading_logic import CascadingRiskAnalyzer
    from utils import mask_to_rgb, overlay_mask_on_image, CLASS_NAMES, CLASS_COLORS
    MODULES_AVAILABLE = True
except ImportError as e:
    MODULES_AVAILABLE = False
    st.error(f"⚠️ Error importing modules: {e}")

# Page configuration
st.set_page_config(
    page_title="Multi-Hazard AI Detection",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 1rem 0;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .risk-critical {
        background-color: #ffebee;
        border-left: 4px solid #f44336;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .risk-high {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .risk-moderate {
        background-color: #fff9c4;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        margin: 0.5rem 0;
    }
    .risk-low {
        background-color: #e8f5e9;
        border-left: 4px solid #4caf50;
        padding: 1rem;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)


def main():
    """Main application"""
    
    # Header
    st.markdown('<div class="main-header">🌍 Multi-Hazard Disaster Detection</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Edge-Aware Hybrid Explainable AI Framework</div>', unsafe_allow_html=True)
    
    # Sidebar
    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Select Feature",
        ["🏠 Home", "🔮 Inference", "🔍 Explainability (Grad-CAM)", "⚠️ Risk Analysis", "📊 About"]
    )
    
    if page == "🏠 Home":
        show_home()
    elif page == "🔮 Inference":
        show_inference()
    elif page == "🔍 Explainability (Grad-CAM)":
        show_gradcam()
    elif page == "⚠️ Risk Analysis":
        show_risk_analysis()
    elif page == "📊 About":
        show_about()


def show_home():
    """Home page"""
    st.header("Welcome to Multi-Hazard Disaster Detection System")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### 🌊 Flood Detection")
        st.info("Identify flood-affected regions from satellite imagery")
    
    with col2:
        st.markdown("### 🔥 Fire Detection")
        st.info("Detect active fire zones and burned areas")
    
    with col3:
        st.markdown("### 🏚️ Damage Assessment")
        st.info("Assess building and infrastructure damage")
    
    st.markdown("---")
    
    st.markdown("### 🎯 Key Features")
    
    features = {
        "Edge-Aware Enhancement": "Canny edge detection fused with RGB channels for improved feature extraction",
        "Deep Learning Model": "ResNet50 encoder + U-Net decoder architecture",
        "Multi-Class Segmentation": "Simultaneous detection of flood, fire, and damage",
        "Explainable AI": "Grad-CAM visualizations for model interpretability",
        "Risk Assessment": "Cascading risk analysis for disaster management"
    }
    
    for feature, description in features.items():
        st.markdown(f"**{feature}**: {description}")
    
    st.markdown("---")
    
    st.markdown("### 🚀 Quick Start")
    st.markdown("""
    1. **Upload Image**: Go to 'Inference' and upload a satellite image
    2. **Get Predictions**: View segmentation results instantly
    3. **Explore Explanations**: Use Grad-CAM to understand model decisions
    4. **Assess Risks**: Analyze cascading disaster risks
    """)
    
    # System status
    st.markdown("---")
    st.markdown("### 💻 System Status")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if MODULES_AVAILABLE:
            st.success("✅ Modules Loaded")
        else:
            st.error("❌ Module Error")
    
    with col2:
        if torch.cuda.is_available():
            st.success(f"✅ GPU: {torch.cuda.get_device_name(0)}")
        else:
            st.warning("⚠️ CPU Mode")
    
    with col3:
        trained_model = os.path.exists("checkpoints/best_model.pth")
        demo_model = os.path.exists("checkpoints/demo_model.pth")
        
        if trained_model:
            st.success("✅ Trained Model Ready")
        elif demo_model:
            st.info("ℹ️ Demo Model Available")
        else:
            st.warning("⚠️ No Model Found")


def show_inference():
    """Inference page"""
    st.header("🔮 Image Inference")
    
    # Check for available models
    best_model_path = "checkpoints/best_model.pth"
    demo_model_path = "checkpoints/demo_model.pth"
    
    model_path = None
    model_type = None
    
    if os.path.exists(best_model_path):
        model_path = best_model_path
        model_type = "trained"
        st.success("✅ Using trained model: `best_model.pth`")
    elif os.path.exists(demo_model_path):
        model_path = demo_model_path
        model_type = "demo"
        st.warning("⚠️ Using demo model with ImageNet weights (not trained on disaster data)")
        st.info("💡 Predictions will be random. Train a real model for accurate results.")
    else:
        st.error("⚠️ No model found. Please train a model or create a demo model.")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Option 1: Create Demo Model")
            st.code("python create_demo_model.py", language="bash")
            st.caption("Creates a model with ImageNet weights for testing")
        
        with col2:
            st.markdown("### Option 2: Train Real Model")
            st.code("python scripts/prepare_dataset.py --mode sample\npython train.py", language="bash")
            st.caption("Train on sample dataset")
        
        st.markdown("---")
        uploaded_model = st.file_uploader("Or Upload Model Checkpoint (.pth)", type=['pth'])
        if uploaded_model:
            os.makedirs("checkpoints", exist_ok=True)
            save_path = "checkpoints/uploaded_model.pth"
            with open(save_path, "wb") as f:
                f.write(uploaded_model.read())
            st.success("✅ Model uploaded successfully!")
            st.rerun()
        return
    
    # Upload image
    uploaded_file = st.file_uploader("Upload Satellite Image", type=['jpg', 'jpeg', 'png', 'tif'])
    
    if uploaded_file is not None:
        # Display original image
        image = Image.open(uploaded_file).convert('RGB')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Original Image")
            st.image(image, use_container_width=True)
        
        # Run inference
        if st.button("🚀 Run Inference", type="primary"):
            with st.spinner("Running inference..."):
                try:
                    # Save temporary file
                    temp_path = "temp_input.jpg"
                    image.save(temp_path)
                    
                    # Create predictor
                    device = 'cuda' if torch.cuda.is_available() else 'cpu'
                    predictor = MultiHazardPredictor(model_path, device=device)
                    
                    # Predict
                    prediction, original_image = predictor.predict(temp_path)
                    
                    # Resize prediction to match original
                    pred_resized = Image.fromarray(prediction.astype(np.uint8))
                    pred_resized = pred_resized.resize(
                        (original_image.shape[1], original_image.shape[0]),
                        Image.NEAREST
                    )
                    pred_resized = np.array(pred_resized)
                    
                    # Generate visualizations
                    mask_rgb = mask_to_rgb(pred_resized)
                    overlay = overlay_mask_on_image(original_image, pred_resized, alpha=0.5)
                    
                    with col2:
                        st.subheader("Segmentation Result")
                        st.image(overlay, use_container_width=True)
                    
                    # Show detailed results
                    st.markdown("---")
                    st.subheader("Detailed Results")
                    
                    tab1, tab2, tab3 = st.tabs(["Mask", "Overlay", "Statistics"])
                    
                    with tab1:
                        st.image(mask_rgb, caption="Segmentation Mask", use_container_width=True)
                    
                    with tab2:
                        st.image(overlay, caption="Overlay", use_container_width=True)
                    
                    with tab3:
                        # Calculate statistics
                        total_pixels = pred_resized.size
                        
                        st.markdown("### Detection Statistics")
                        
                        for class_idx, class_name in enumerate(CLASS_NAMES):
                            class_pixels = np.sum(pred_resized == class_idx)
                            percentage = (class_pixels / total_pixels) * 100
                            
                            if class_name != "Background":
                                color = CLASS_COLORS[class_idx]
                                color_hex = f"#{color[0]:02x}{color[1]:02x}{color[2]:02x}"
                                
                                st.markdown(f"""
                                <div style="background-color: {color_hex}20; padding: 10px; border-radius: 5px; margin: 5px 0;">
                                    <strong>{class_name}</strong>: {class_pixels:,} pixels ({percentage:.2f}%)
                                </div>
                                """, unsafe_allow_html=True)
                    
                    # Clean up
                    os.remove(temp_path)
                    
                    st.success("✅ Inference completed successfully!")
                    
                except Exception as e:
                    st.error(f"Error during inference: {e}")


def show_gradcam():
    """Grad-CAM explainability page"""
    st.header("🔍 Explainability with Grad-CAM")
    
    st.info("Grad-CAM (Gradient-weighted Class Activation Mapping) highlights the regions that the model focuses on when making predictions.")
    
    # Check if model exists
    model_path = "checkpoints/best_model.pth"
    if not os.path.exists(model_path):
        st.error("⚠️ No trained model found.")
        return
    
    # Upload image
    uploaded_file = st.file_uploader("Upload Satellite Image for Explanation", type=['jpg', 'jpeg', 'png', 'tif'])
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        
        st.subheader("Original Image")
        st.image(image, use_container_width=True)
        
        # Target class selection
        class_options = {
            "Auto (All Detected)": None,
            "Flood": 1,
            "Fire": 2,
            "Building Damage": 3
        }
        
        selected_class = st.selectbox("Select Target Class", list(class_options.keys()))
        
        if st.button("🔍 Generate Grad-CAM", type="primary"):
            with st.spinner("Generating Grad-CAM visualization..."):
                try:
                    # Try to import grad-cam
                    try:
                        from pytorch_grad_cam import GradCAM
                        from pytorch_grad_cam.utils.image import show_cam_on_image
                        import cv2
                        gradcam_available = True
                    except ImportError:
                        gradcam_available = False
                    
                    if not gradcam_available:
                        st.warning("⚠️ Grad-CAM visualization requires the 'grad-cam' package. Please install it first:")
                        st.code("pip install grad-cam", language="bash")
                        
                        st.info("""
                        **Grad-CAM Features:**
                        - Highlights important regions for each hazard class
                        - Shows model attention and decision-making process
                        - Helps validate model predictions
                        - Useful for research and debugging
                        """)
                        return
                    
                    # Save temporary file
                    temp_path = "temp_gradcam.jpg"
                    image.save(temp_path)
                    
                    # Load and preprocess image
                    img_array = np.array(image)
                    img_resized = cv2.resize(img_array, (256, 256))
                    img_normalized = img_resized.astype(np.float32) / 255.0
                    
                    # Create model
                    device = 'cuda' if torch.cuda.is_available() else 'cpu'
                    model = create_model(num_classes=4, in_channels=4)
                    
                    # Load checkpoint
                    checkpoint = torch.load(model_path, map_location=device)
                    model.load_state_dict(checkpoint['model_state_dict'])
                    model = model.to(device)
                    model.eval()
                    
                    # Get edge map using EdgeExtractor
                    edge_extractor = EdgeExtractor()
                    # EdgeExtractor.__call__ returns 4-channel tensor [4, H, W]
                    img_with_edges_tensor = edge_extractor(img_resized)
                    
                    # Convert back to numpy for consistency [H, W, 4]
                    img_with_edges = img_with_edges_tensor.permute(1, 2, 0).numpy()
                    
                    # Prepare input tensor - EdgeExtractor already gives us [4, H, W]
                    input_tensor = img_with_edges_tensor.unsqueeze(0).to(device)
                    
                    # Normalize
                    mean = torch.tensor([0.485, 0.456, 0.406, 0.5]).view(1, 4, 1, 1).to(device)
                    std = torch.tensor([0.229, 0.224, 0.225, 0.25]).view(1, 4, 1, 1).to(device)
                    input_tensor = (input_tensor - mean) / std
                    
                    # Get target layer (last conv layer of decoder)
                    target_layers = [model.model.decoder.blocks[-1]]
                    
                    # Determine target class
                    target_category = class_options[selected_class]
                    
                    # Generate Grad-CAM
                    cam = GradCAM(model=model, target_layers=target_layers)
                    
                    if target_category is None:
                        # Show all detected classes
                        st.subheader("Grad-CAM for All Detected Classes")
                        
                        cols = st.columns(3)
                        for idx, (class_name, class_idx) in enumerate([("Flood", 1), ("Fire", 2), ("Damage", 3)]):
                            grayscale_cam = cam(input_tensor=input_tensor, targets=None)
                            grayscale_cam = grayscale_cam[0, :]
                            
                            visualization = show_cam_on_image(img_normalized, grayscale_cam, use_rgb=True)
                            
                            with cols[idx]:
                                st.image(visualization, caption=f"{class_name} Attention", use_container_width=True)
                    else:
                        # Show specific class
                        grayscale_cam = cam(input_tensor=input_tensor, targets=None)
                        grayscale_cam = grayscale_cam[0, :]
                        
                        visualization = show_cam_on_image(img_normalized, grayscale_cam, use_rgb=True)
                        
                        st.subheader(f"Grad-CAM for {selected_class}")
                        st.image(visualization, caption="Model Attention Map", use_container_width=True)
                    
                    st.success("✅ Grad-CAM generated successfully!")
                    
                    st.info("""
                    **Interpretation:**
                    - Red/warm colors indicate regions the model focuses on
                    - Blue/cool colors indicate less important regions
                    - Helps understand model decision-making process
                    """)
                    
                    # Clean up
                    os.remove(temp_path)
                    
                except Exception as e:
                    st.error(f"Error generating Grad-CAM: {e}")
                    import traceback
                    st.code(traceback.format_exc())


def show_risk_analysis():
    """Risk analysis page"""
    st.header("⚠️ Cascading Risk Analysis")
    
    st.info("Analyze potential cascading risks based on detected hazards using rule-based logic.")
    
    # Upload image or use previous prediction
    uploaded_file = st.file_uploader("Upload Satellite Image", type=['jpg', 'jpeg', 'png', 'tif'], key="risk_upload")
    
    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Input Image")
            st.image(image, use_container_width=True)
        
        # Additional parameters
        st.subheader("Environmental Parameters (Optional)")
        rainfall_intensity = st.slider("Rainfall Intensity (%)", 0, 100, 50)
        
        if st.button("⚠️ Analyze Risks", type="primary"):
            with st.spinner("Analyzing cascading risks..."):
                try:
                    # First run inference
                    model_path = "checkpoints/best_model.pth"
                    if not os.path.exists(model_path):
                        st.error("⚠️ No trained model found.")
                        return
                    
                    # Save temporary file
                    temp_path = "temp_risk.jpg"
                    image.save(temp_path)
                    
                    # Create predictor
                    device = 'cuda' if torch.cuda.is_available() else 'cpu'
                    predictor = MultiHazardPredictor(model_path, device=device)
                    
                    # Predict
                    prediction, _ = predictor.predict(temp_path)
                    
                    with col2:
                        st.subheader("Detected Hazards")
                        mask_rgb = mask_to_rgb(prediction)
                        st.image(mask_rgb, use_container_width=True)
                    
                    # Analyze risks
                    analyzer = CascadingRiskAnalyzer(image_size=prediction.shape)
                    analysis = analyzer.analyze(prediction, rainfall_intensity=rainfall_intensity)
                    
                    # Display results
                    st.markdown("---")
                    st.subheader("Risk Assessment Report")
                    
                    # Hazard areas
                    st.markdown("### 📊 Detected Hazard Areas")
                    cols = st.columns(3)
                    
                    hazard_data = [
                        ("Flood", analysis['hazard_areas']['flood'], "🌊"),
                        ("Fire", analysis['hazard_areas']['fire'], "🔥"),
                        ("Damage", analysis['hazard_areas']['damage'], "🏚️")
                    ]
                    
                    for idx, (name, percentage, icon) in enumerate(hazard_data):
                        with cols[idx]:
                            st.metric(f"{icon} {name}", f"{percentage:.2f}%")
                    
                    # Risk categories
                    st.markdown("### 🎯 Risk Categories")
                    
                    risks = [
                        ("Infrastructure Risk", analysis['infrastructure_risk']),
                        ("Landslide Risk", analysis['landslide_risk']),
                        ("Fire Spread Risk", analysis['fire_spread_risk']),
                        ("Compound Disaster", analysis['compound_disaster'])
                    ]
                    
                    for risk_name, risk_data in risks:
                        level = risk_data['level']
                        score = risk_data['score']
                        
                        # Determine risk class
                        if 'Critical' in level or 'High' in level:
                            risk_class = "risk-critical"
                        elif 'Moderate' in level:
                            risk_class = "risk-moderate"
                        else:
                            risk_class = "risk-low"
                        
                        st.markdown(f"""
                        <div class="{risk_class}">
                            <strong>{risk_name}</strong><br>
                            Level: {level}<br>
                            Score: {score:.2f}/100
                        </div>
                        """, unsafe_allow_html=True)
                    
                    # Overall risk
                    st.markdown("### 🎯 Overall Risk Score")
                    overall_score = analysis['overall_risk_score']
                    
                    # Progress bar
                    st.progress(min(overall_score / 100, 1.0))
                    st.markdown(f"**{overall_score:.2f}/100**")
                    
                    # Active hazards
                    if 'active_hazards' in analysis['compound_disaster']:
                        st.markdown("### 🚨 Active Hazards")
                        for hazard in analysis['compound_disaster']['active_hazards']:
                            st.markdown(f"- {hazard}")
                    
                    # Clean up
                    os.remove(temp_path)
                    
                    st.success("✅ Risk analysis completed!")
                    
                except Exception as e:
                    st.error(f"Error during risk analysis: {e}")


def show_about():
    """About page"""
    st.header("📊 About This Project")
    
    st.markdown("""
    ### Edge-Aware Hybrid Explainable AI Framework
    
    This project implements a state-of-the-art multi-hazard disaster detection system using satellite imagery.
    
    #### 🎯 Research Objectives
    - Develop an edge-aware feature enhancement technique
    - Create a multi-hazard segmentation model
    - Implement explainable AI for disaster detection
    - Provide cascading risk assessment
    
    #### 🏗️ Architecture
    - **Encoder**: ResNet50 (pretrained on ImageNet)
    - **Decoder**: U-Net architecture
    - **Input**: 4 channels (RGB + Canny Edge)
    - **Output**: 4 classes (Background, Flood, Fire, Damage)
    
    #### 📚 Technologies
    - PyTorch for deep learning
    - Segmentation Models PyTorch for U-Net
    - OpenCV for edge detection
    - Streamlit for web interface
    - Grad-CAM for explainability
    
    #### 🎓 Academic Use
    This framework is designed for:
    - Research papers
    - Academic projects
    - Disaster management systems
    - Remote sensing applications
    
    #### 📖 Documentation
    For detailed documentation, see the README.md file in the project directory.
    
    #### 👨‍💻 Developer
    Research Project: Multi-Hazard Disaster Detection
    """)
    
    st.markdown("---")
    
    st.markdown("### 📊 Model Information")
    
    if os.path.exists("checkpoints/best_model.pth"):
        checkpoint = torch.load("checkpoints/best_model.pth", map_location='cpu')
        
        col1, col2 = st.columns(2)
        
        with col1:
            if 'epoch' in checkpoint:
                st.metric("Training Epoch", checkpoint['epoch'])
        
        with col2:
            if 'loss' in checkpoint:
                st.metric("Validation Loss", f"{checkpoint['loss']:.4f}")
    else:
        st.info("No trained model available yet.")


if __name__ == "__main__":
    main()
