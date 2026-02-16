# Quick Start Guide - No Trained Model

## 🎯 You Have 3 Options

### Option 1: Use Demo Mode (Recommended for Testing)
The Streamlit app can demonstrate functionality with synthetic data without needing a trained model or dataset.

**Coming soon**: Demo mode with sample predictions

---

### Option 2: Download a Pre-trained Model
If you have access to a pre-trained checkpoint:

1. Go to the **Inference** page in the Streamlit app
2. Click "Upload Model Checkpoint (.pth)"
3. Upload your `.pth` file
4. The app will use it for predictions

---

### Option 3: Train Your Own Model

#### Step 1: Prepare Dataset

Create this folder structure:
```
multi_hazard_ai/
└── data/
    ├── train/
    │   ├── images/     # Put training satellite images here
    │   └── masks/      # Put corresponding masks here
    └── val/
        ├── images/     # Put validation images here
        └── masks/      # Put validation masks here
```

**Mask Requirements:**
- Format: Grayscale PNG images
- Pixel values:
  - `0` = Background
  - `1` = Flood
  - `2` = Fire  
  - `3` = Building Damage
- Same filename as corresponding image

#### Step 2: Run Training

```bash
python train.py
```

**Training Configuration** (edit in `train.py`):
- Batch size: 8 (reduce to 4 if out of memory)
- Epochs: 50
- Learning rate: 0.0001
- Image size: 256x256

**Expected Time:**
- With GPU: 2-4 hours
- With CPU: 8-12 hours (not recommended)

#### Step 3: Use Trained Model

After training completes:
- Model saved to: `checkpoints/best_model.pth`
- Refresh the Streamlit app
- Upload images for inference

---

## 🧪 Testing Without Training

### Test Individual Modules

You can test each module independently:

```bash
# Test edge detection
python edge_module.py

# Test model architecture
python model.py

# Test utilities
python utils.py

# Test risk analysis (works without model!)
python cascading_logic.py
```

### Use Risk Analysis Feature

The **Risk Analysis** feature in the Streamlit app can work with synthetic data to demonstrate functionality.

---

## 📊 Where to Get Satellite Imagery

### Free Datasets:
1. **Sentinel Hub** - https://www.sentinel-hub.com/
2. **NASA Earthdata** - https://earthdata.nasa.gov/
3. **Google Earth Engine** - https://earthengine.google.com/
4. **Copernicus Open Access Hub** - https://scihub.copernicus.eu/

### Disaster-Specific Datasets:
- **xBD (Building Damage)** - https://xview2.org/
- **FloodNet** - Flood detection dataset
- **EFFIS** - European Forest Fire Information System

---

## 🎓 For Academic Demonstration

If you need to demonstrate the system for your research without real data:

1. Create synthetic test images
2. Generate corresponding masks manually
3. Use small dataset (10-20 images) for quick training
4. Focus on architecture and methodology rather than accuracy

---

## ⚡ Quick Demo Setup (5 minutes)

1. Create minimal dataset (5 images + masks)
2. Reduce epochs to 5 in `train.py`
3. Run quick training
4. Test inference in Streamlit app

This won't give good results but will demonstrate the complete pipeline!

---

## 💡 Current App Capabilities

**Without a trained model, you can still:**
- ✅ Explore the UI and navigation
- ✅ See the system architecture
- ✅ Read about features
- ✅ Upload a pre-trained checkpoint
- ❌ Run actual inference (needs model)
- ❌ Generate Grad-CAM (needs model)
- ❌ Analyze real images (needs model)

---

## 🚀 Next Steps

**Choose your path:**

1. **Just exploring?** → Browse the Streamlit app UI
2. **Have a checkpoint?** → Upload it in the app
3. **Have dataset?** → Run `python train.py`
4. **Need dataset?** → Check the resources above
5. **Quick demo?** → Create 5 sample images and train

---

**Need help with any of these options? Let me know!**
