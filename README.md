# Edge-Aware Hybrid Explainable AI Framework for Multi-Hazard Disaster Detection

## 📋 Project Overview

This project implements a **multi-hazard semantic segmentation model** for disaster detection using satellite imagery. The framework integrates edge-aware feature enhancement, deep learning segmentation, and explainable AI techniques.

### Detected Hazards
- 🌊 **Flood regions**
- 🔥 **Fire regions**
- 🏚️ **Building damage**

### Key Features
1. **Edge-aware feature enhancement** using Canny edge detection
2. **ResNet50 encoder** with U-Net decoder architecture
3. **Multi-class segmentation** (4 classes)
4. **Grad-CAM explainability** for model interpretability
5. **Cascading risk logic** for disaster assessment

---

## 📁 Project Structure

```
multi_hazard_ai/
├── data_loader.py          # Dataset loading and augmentation
├── edge_module.py          # Edge extraction and fusion
├── model.py                # ResNet50 + U-Net model
├── train.py                # Training script
├── inference.py            # Inference script
├── xai_module.py           # Grad-CAM explainability
├── cascading_logic.py      # Risk assessment logic
├── utils.py                # Helper functions
└── requirements.txt        # Dependencies
```

---

## 🚀 Installation

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Prepare Dataset

Organize your data in the following structure:

```
data/
├── train/
│   ├── images/         # Training satellite images
│   └── masks/          # Training segmentation masks
└── val/
    ├── images/         # Validation images
    └── masks/          # Validation masks
```

**Mask Format:**
- Grayscale PNG images
- Pixel values: 0 (background), 1 (flood), 2 (fire), 3 (damage)

---

## 🎓 Training

### Basic Training

```bash
python train.py
```

### Custom Configuration

Edit the `config` dictionary in `train.py`:

```python
config = {
    'train_img_dir': 'data/train/images',
    'train_mask_dir': 'data/train/masks',
    'val_img_dir': 'data/val/images',
    'val_mask_dir': 'data/val/masks',
    'num_classes': 4,
    'batch_size': 8,
    'num_epochs': 50,
    'learning_rate': 1e-4,
    'img_size': (256, 256),
    'device': 'cuda',
    'save_dir': 'checkpoints'
}
```

### Training Output

- **Checkpoints**: Saved to `checkpoints/` directory
- **Best model**: `checkpoints/best_model.pth`
- **Metrics**: IoU, pixel accuracy, loss

---

## 🔮 Inference

### Single Image Prediction

```bash
python inference.py --image path/to/image.jpg --model checkpoints/best_model.pth
```

### Options

```bash
python inference.py \
  --image test.jpg \
  --model checkpoints/best_model.pth \
  --output outputs \
  --size 256 \
  --device cuda
```

### Output Files

1. `*_mask.png` - Colored segmentation mask
2. `*_overlay.png` - Mask overlaid on original image
3. `*_visualization.png` - Complete visualization

---

## 🔍 Grad-CAM Explainability

### Generate Grad-CAM Heatmaps

```bash
python xai_module.py --image test.jpg --model checkpoints/best_model.pth
```

### For Specific Class

```bash
python xai_module.py --image test.jpg --model checkpoints/best_model.pth --class 1
```

**Class indices:**
- 0: Background
- 1: Flood
- 2: Fire
- 3: Damage

### Output

- Multi-class Grad-CAM visualization showing attention maps for each detected hazard
- Heatmaps highlighting important regions for model decisions

---

## ⚠️ Cascading Risk Analysis

### Programmatic Usage

```python
from cascading_logic import CascadingRiskAnalyzer
import numpy as np

# Load prediction mask
prediction = np.load('prediction.npy')

# Create analyzer
analyzer = CascadingRiskAnalyzer(image_size=(256, 256))

# Perform analysis
analysis = analyzer.analyze(prediction, rainfall_intensity=50)

# Generate report
report = analyzer.generate_report(analysis)
print(report)
```

### Risk Categories

1. **Infrastructure Risk**: Flood + Building damage
2. **Landslide Risk**: Flood + Fire + Rainfall
3. **Fire Spread Risk**: Fire + Damaged buildings
4. **Compound Disaster**: Multiple hazards present

---

## 🧪 Testing Individual Modules

### Test Data Loader

```bash
python data_loader.py
```

### Test Edge Module

```bash
python edge_module.py
```

### Test Model

```bash
python model.py
```

### Test Utils

```bash
python utils.py
```

### Test Cascading Logic

```bash
python cascading_logic.py
```

---

## 💻 Google Colab Setup

### 1. Upload Project

```python
from google.colab import drive
drive.mount('/content/drive')

# Navigate to project directory
%cd /content/drive/MyDrive/multi_hazard_ai
```

### 2. Install Dependencies

```python
!pip install -q -r requirements.txt
```

### 3. Check GPU

```python
import torch
print(f"GPU Available: {torch.cuda.is_available()}")
print(f"GPU Name: {torch.cuda.get_device_name(0)}")
```

### 4. Train

```python
!python train.py
```

### 5. Inference

```python
!python inference.py --image test.jpg --model checkpoints/best_model.pth
```

---

## 📊 Model Architecture

```
Input: RGB Image (3 channels)
   ↓
Edge Extraction (Canny)
   ↓
Fusion: RGB + Edge (4 channels)
   ↓
ResNet50 Encoder (Pretrained)
   ↓
U-Net Decoder
   ↓
Output: Segmentation Map (4 classes)
```

### Model Details

- **Encoder**: ResNet50 (pretrained on ImageNet)
- **Decoder**: U-Net architecture
- **Input**: 4 channels (RGB + Edge)
- **Output**: 4 classes
- **Loss**: CrossEntropyLoss
- **Optimizer**: Adam (lr=1e-4)

---

## 📈 Performance Metrics

The model is evaluated using:

1. **Mean IoU (mIoU)**: Intersection over Union averaged across classes
2. **Pixel Accuracy**: Percentage of correctly classified pixels
3. **Class-wise IoU**: IoU for each hazard class

---

## 🎯 Expected Results

### Training
- Convergence within 30-50 epochs
- Best mIoU: 0.60-0.80 (depending on dataset quality)
- Training time: ~2-4 hours on Google Colab GPU

### Inference
- Processing time: ~0.1-0.5 seconds per image (GPU)
- Output: Segmentation masks with 4 classes

### Grad-CAM
- Visual explanations showing model attention
- Class-specific heatmaps

---

## 🔧 Troubleshooting

### CUDA Out of Memory
- Reduce batch size in `train.py`
- Reduce image size to 128x128

### No GPU Available
- Model will run on CPU (slower)
- Reduce batch size to 2-4

### Import Errors
- Ensure all dependencies are installed
- Use Python 3.8+

---

## 📚 Citation

If you use this code for your research, please cite:

```
@article{your_paper,
  title={Edge-Aware Hybrid Explainable AI Framework for Multi-Hazard Disaster Detection Using Satellite Imagery},
  author={Your Name},
  year={2026}
}
```

---

## 📝 License

This project is for academic and research purposes.

---

## 🤝 Contributing

This is a research project. Feel free to extend it with:
- Additional hazard types
- Advanced architectures (Vision Transformers)
- Temporal analysis (video sequences)
- Real-time deployment

---

## ✅ Quick Start Checklist

- [ ] Install dependencies
- [ ] Prepare dataset
- [ ] Run training
- [ ] Test inference
- [ ] Generate Grad-CAM
- [ ] Analyze cascading risks

---

**Happy Researching! 🚀**
