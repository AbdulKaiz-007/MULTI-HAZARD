# 🌍 Dataset Sources for Multi-Hazard Disaster Detection

This document provides direct links and instructions for downloading datasets for your research project.

---

## ✅ 1️⃣ **xView2 (Building Damage Assessment)**

**Purpose**: Building damage detection from pre/post disaster satellite imagery

**Links**:
- 🔗 Main Portal: https://xview2.org
- 🔗 GitHub: https://github.com/DIUx-xView/xview2
- 🔗 Dataset Download: https://xview2.org/dataset

**What You Get**:
- Pre- and post-disaster satellite images
- Pixel-level damage annotations (4 classes)
- Multiple disaster types (earthquakes, floods, fires, etc.)
- ~850,000 building annotations

**How to Use**:
1. Register at xView2.org
2. Download the dataset (requires agreement to terms)
3. Use our `download_xview2.py` script to organize data

---

## ✅ 2️⃣ **Sentinel-1 SAR (Flood Detection)**

**Purpose**: Synthetic Aperture Radar for flood detection (works through clouds)

**Links**:
- 🔗 Copernicus Hub: https://scihub.copernicus.eu/
- 🔗 AWS Open Data: https://registry.opendata.aws/sentinel-1/
- 🔗 Google Earth Engine: https://earthengine.google.com/

**What You Get**:
- SAR imagery (C-band)
- All-weather, day/night capability
- Level-1 GRD products for flood mapping

**How to Use**:
1. Create account at Copernicus SciHub
2. Use our `download_sentinel.py` script
3. Or use Google Earth Engine Python API

---

## ✅ 3️⃣ **Sentinel-2 Optical (Fire Detection)**

**Purpose**: Multi-spectral optical imagery for fire and vegetation analysis

**Links**:
- 🔗 Copernicus Hub: https://scihub.copernicus.eu/
- 🔗 AWS Open Data: https://registry.opendata.aws/sentinel-2/
- 🔗 Google Earth Engine: https://earthengine.google.com/

**What You Get**:
- 13 spectral bands (visible, NIR, SWIR)
- 10m to 60m resolution
- Thermal bands for fire detection

**How to Use**:
1. Same as Sentinel-1
2. Use bands 11 & 12 (SWIR) for fire detection
3. Calculate NDVI, NBR indices

---

## ✅ 4️⃣ **SRTM Digital Elevation Model**

**Purpose**: Terrain data for slope analysis and cascading risk

**Links**:
- 🔗 USGS EarthExplorer: https://earthexplorer.usgs.gov/
- 🔗 OpenTopography: https://opentopography.org/

**What You Get**:
- 30m or 90m resolution DEM
- Global coverage
- Elevation, slope, aspect data

**How to Use**:
1. Search by coordinates or region name
2. Download SRTM tiles
3. Use for landslide risk assessment

---

## ✅ 5️⃣ **Meteorological Data**

**Purpose**: Rainfall and weather data for cascading risk logic

**Links**:
- 🔗 NASA GPM: https://precip.gsfc.nasa.gov/
- 🔗 ERA5 Climate: https://cds.climate.copernicus.eu/
- 🔗 NOAA: https://www.ncdc.noaa.gov/

**What You Get**:
- Precipitation estimates
- Wind speed/direction
- Temperature data

---

## 📦 Quick Start Scripts

We've created Python scripts to help you download and prepare data:

### 1. Download xView2
```bash
python scripts/download_xview2.py --output data/xview2
```

### 2. Download Sentinel Data
```bash
python scripts/download_sentinel.py --region "coordinates" --date-range "2023-01-01,2023-12-31"
```

### 3. Prepare Training Data
```bash
python scripts/prepare_dataset.py --source data/xview2 --output data/train
```

---

## 🎯 Recommended Workflow

### For Quick Testing (1-2 hours):
1. Download 50-100 images from xView2
2. Use pre-labeled damage masks
3. Train for 10 epochs
4. Test inference

### For Full Research (1-2 days):
1. Download complete xView2 dataset
2. Download Sentinel-2 for fire regions
3. Download Sentinel-1 for flood regions
4. Combine datasets
5. Train for 50 epochs

---

## 📊 Dataset Statistics

| Dataset | Size | Images | Classes | Resolution |
|---------|------|--------|---------|------------|
| xView2 | ~10GB | ~850K buildings | 4 damage levels | 0.3-0.8m |
| Sentinel-1 | Varies | On-demand | Continuous | 10m |
| Sentinel-2 | Varies | On-demand | 13 bands | 10-60m |
| SRTM | ~1GB/tile | Elevation | Continuous | 30m |

---

## 🔐 Authentication Required

### Copernicus SciHub
1. Register: https://scihub.copernicus.eu/dhus/#/self-registration
2. Verify email
3. Use credentials in download scripts

### Google Earth Engine
1. Sign up: https://earthengine.google.com/signup/
2. Create project
3. Install Python API: `pip install earthengine-api`

---

## 💡 Tips

- **Start small**: Download 10-20 images first to test pipeline
- **Use Google Colab**: Free GPU + easy dataset access
- **Check licenses**: Most datasets are free for academic use
- **Cite properly**: Include dataset citations in your paper

---

## 📚 Dataset Citations

### xView2
```
Gupta et al. (2019). xBD: A Dataset for Assessing Building Damage from Satellite Imagery. 
CVPR Workshops.
```

### Sentinel
```
European Space Agency (ESA). Copernicus Sentinel-1/2 Data. 
https://scihub.copernicus.eu/
```

---

## 🚀 Next Steps

1. ✅ Choose your dataset source
2. ✅ Run download scripts (see `/scripts` folder)
3. ✅ Prepare data in correct format
4. ✅ Run training: `python train.py`
5. ✅ Test in Streamlit app

---

**Ready to download? Check the `/scripts` folder for automated download tools!**
