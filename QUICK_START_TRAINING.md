# Quick Start: Training with Real Data

## 🚀 Option 1: Use Sample Dataset (5 minutes)

Perfect for testing the complete pipeline!

```bash
# Create sample dataset
python scripts/prepare_dataset.py --mode sample --output data --samples 20

# Train the model (quick test with 5 epochs)
python train.py
```

**What this does:**
- Creates 20 synthetic images with masks
- 16 for training, 4 for validation
- Demonstrates the complete workflow
- Takes ~10-15 minutes on CPU

---

## 🌍 Option 2: Download Real Datasets

### Step 1: Create Sample Dataset First
```bash
python scripts/prepare_dataset.py --mode sample --output data --samples 50
```

### Step 2: Train Initial Model
```bash
python train.py
```

### Step 3: Test in Streamlit
```bash
streamlit run app.py
```

### Step 4: Download Real Data (when ready)
Follow instructions in `DATASET_SOURCES.md`

---

## 📊 Training Configuration

Edit `train.py` to adjust:

```python
config = {
    'batch_size': 4,        # Reduce if out of memory
    'num_epochs': 10,       # Start with 10 for testing
    'learning_rate': 1e-4,
    'img_size': (256, 256), # Reduce to (128, 128) if needed
}
```

---

## ✅ Verification Checklist

Before training, verify your dataset:

```bash
python scripts/prepare_dataset.py --mode verify --output data
```

**Expected output:**
```
TRAIN:
  Images: 16
  Masks: 16
  Mask shape: (256, 256)
  Unique values: [0 1 2 3]
```

---

## 🎯 Next Steps After Training

1. **Check Results**:
   - Model saved to `checkpoints/best_model.pth`
   - View training logs

2. **Test Inference**:
   ```bash
   python inference.py --image sample_satellite.jpg --model checkpoints/best_model.pth
   ```

3. **Use Streamlit App**:
   - Upload images
   - View predictions
   - Analyze risks

4. **Generate Grad-CAM**:
   ```bash
   python xai_module.py --image test.jpg --model checkpoints/best_model.pth
   ```

---

## 💡 Tips

- **Start small**: Train with 10-20 images first
- **Monitor GPU**: Use `nvidia-smi` to check usage
- **Save checkpoints**: Model saves every epoch
- **Check IoU**: Good models achieve 0.6-0.8 mIoU

---

**Ready to start? Run the sample dataset command above!**
