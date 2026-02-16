# 🌐 Streamlit Web Application Guide

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install streamlit plotly grad-cam
```

Or install all requirements:

```bash
pip install -r requirements.txt
```

### 2. Run the Application

```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

---

## 📱 Features

### 🏠 Home Page
- Overview of the system
- Key features description
- System status (GPU, modules, model)
- Quick start guide

### 🔮 Inference
- Upload satellite images
- Run multi-hazard detection
- View segmentation results
- See detection statistics
- Download results

### 🔍 Explainability (Grad-CAM)
- Generate visual explanations
- Understand model decisions
- Class-specific attention maps
- Interpretable AI insights

### ⚠️ Risk Analysis
- Cascading risk assessment
- Infrastructure risk evaluation
- Landslide risk prediction
- Fire spread analysis
- Compound disaster detection
- Environmental parameter integration

### 📊 About
- Project information
- Architecture details
- Model statistics
- Documentation links

---

## 🎨 User Interface

### Navigation
- **Sidebar**: Select different features
- **Main Area**: Interactive content
- **Responsive Design**: Works on desktop and tablet

### Color Coding
- **Blue**: Flood regions
- **Red**: Fire regions
- **Yellow**: Building damage
- **Black**: Background

### Risk Levels
- 🟢 **Low**: Green background
- 🟡 **Moderate**: Yellow background
- 🟠 **High**: Orange background
- 🔴 **Critical**: Red background

---

## 💡 Usage Tips

### For Inference
1. Upload a satellite image (JPG, PNG, TIF)
2. Click "Run Inference"
3. View results in tabs:
   - **Mask**: Pure segmentation
   - **Overlay**: Mask on image
   - **Statistics**: Detection percentages

### For Risk Analysis
1. Upload an image
2. Adjust environmental parameters (optional)
3. Click "Analyze Risks"
4. Review comprehensive risk report

### Model Upload
If no trained model exists:
1. Go to Inference page
2. Upload your `.pth` checkpoint file
3. System will save it automatically

---

## 🔧 Troubleshooting

### Port Already in Use
```bash
streamlit run app.py --server.port 8502
```

### Module Import Errors
Ensure all dependencies are installed:
```bash
pip install -r requirements.txt
```

### GPU Not Detected
- Application works on CPU (slower)
- For GPU: Install CUDA-compatible PyTorch

### Model Not Found
- Train the model first using `train.py`
- Or upload a pre-trained checkpoint

---

## 🌐 Deployment Options

### Local Development
```bash
streamlit run app.py
```

### Streamlit Cloud
1. Push code to GitHub
2. Connect to Streamlit Cloud
3. Deploy automatically

### Docker
```dockerfile
FROM python:3.9
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py"]
```

### Heroku
Add `setup.sh` and `Procfile`:

**setup.sh**:
```bash
mkdir -p ~/.streamlit/
echo "[server]
headless = true
port = $PORT
enableCORS = false
" > ~/.streamlit/config.toml
```

**Procfile**:
```
web: sh setup.sh && streamlit run app.py
```

---

## 📊 Performance

### Recommended Specs
- **RAM**: 8GB minimum, 16GB recommended
- **GPU**: Optional but recommended for faster inference
- **Storage**: 2GB for model and dependencies

### Optimization
- Use smaller image sizes for faster processing
- Enable GPU acceleration when available
- Cache model loading with `@st.cache_resource`

---

## 🎯 Advanced Features

### Custom Styling
Edit the CSS in `app.py` to customize appearance:
```python
st.markdown("""
<style>
    /* Your custom CSS here */
</style>
""", unsafe_allow_html=True)
```

### Add New Pages
Create new functions and add to navigation:
```python
def show_new_feature():
    st.header("New Feature")
    # Your code here

# In main():
page = st.sidebar.radio(
    "Select Feature",
    ["Home", "Inference", "New Feature"]
)
```

---

## 📝 Configuration

### Streamlit Config
Create `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#1f77b4"
backgroundColor = "#ffffff"
secondaryBackgroundColor = "#f0f2f6"
textColor = "#262730"
font = "sans serif"

[server]
maxUploadSize = 200
```

---

## 🔐 Security Notes

### For Production
- Add authentication
- Limit file upload sizes
- Validate input images
- Use HTTPS
- Set up rate limiting

---

## 📚 Resources

- [Streamlit Documentation](https://docs.streamlit.io)
- [Streamlit Gallery](https://streamlit.io/gallery)
- [Streamlit Forum](https://discuss.streamlit.io)

---

## ✅ Checklist

- [ ] Install dependencies
- [ ] Run application locally
- [ ] Upload test image
- [ ] Test inference feature
- [ ] Test risk analysis
- [ ] Customize styling (optional)
- [ ] Deploy to cloud (optional)

---

**Enjoy your Multi-Hazard AI Web Application! 🚀**
