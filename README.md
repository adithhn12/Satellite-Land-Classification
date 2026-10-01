# Satellite Land Cover Classification & Temporal Change Detection System

An AI-powered web application and GIS framework for classifying satellite images into 10 EuroSAT land-cover categories. Features 2-stage ResNet50V2 transfer learning, multi-crop scale-preserving test-time augmentation (TTA), overlapping spatial grid classification, Grad-CAM saliency visualizers, temporal change detection, external dataset evaluation workflows, and downloadable PDF reports.

---

## 🏗️ System Architecture & ML Pipeline

* **Model Base**: ResNet50V2 pre-trained on ImageNet.
* **Input Preprocessing**: Rescaled to $[-1, 1]$ range via `keras.layers.Rescaling(scale=2.0, offset=-1.0)`.
* **Data Augmentation**: Spatial-preserving rotations ($0^\circ, 90^\circ, 180^\circ, 270^\circ$), horizontal & vertical flips, moderate zoom, and contrast adjustments to handle satellite camera orientations and sensor variations.
* **Transfer Learning Strategy**:
  1. *Stage 1 (Head Warmup)*: Base model frozen; custom classification head trained for 6 epochs (`lr = 1e-3`).
  2. *Stage 2 (Top Layer Fine-Tuning)*: Unfreezes upper 30 layers of ResNet50V2; fine-tunes with small learning rate (`lr = 1e-5`) and early stopping.
* **Out-of-Distribution Generalization**: Single image inference applies **Test-Time Augmentation (TTA)** and **Multi-Crop Tile Ensembling** (5 crops: Center + 4 Corners) on images $>128 \times 128$ px to prevent feature scale compression blur on large satellite composites.
* **Overlapping Grid Classification**: Slices arbitrary large regional imagery using overlapping windows with customizable stride. Performs spatial probability accumulation and 3x3 Gaussian neighborhood smoothing (preserving sharp boundaries for high-confidence cells $>70\%$).
* **Uncertainty Calibration**: Flags predictions with peak confidence $<45\%$ or small probability margins ($<12\%$) as **Uncertain / Low Confidence** rather than presenting uncalibrated softmax scores as reliable.

---

## 🌍 EuroSAT Land Cover Classes (10 Categories)

1. `AnnualCrop`: Plantations harvested annually (e.g., wheat, maize).
2. `Forest`: Dense tree canopy and woodland reserves.
3. `HerbaceousVegetation`: Natural pastures, shrubs, and wild grass.
4. `Highway`: Transportation routes, asphalt, and concrete roads.
5. `Industrial`: Commercial complexes, manufacturing units, and warehouses.
6. `Pasture`: Managed grazing fields and meadows.
7. `PermanentCrop`: Long-term orchards, vineyards, and plantations.
8. `Residential`: Suburban housing, towns, and residential tracts.
9. `River`: Flowing freshwater streams and rivers.
10. `SeaLake`: Inland lakes, seas, and marine coastal areas.

---

## 🧪 EuroSAT Benchmark vs. External Generalization Performance

> [!IMPORTANT]
> High accuracy on the EuroSAT benchmark split does **NOT** equal real-world external accuracy on arbitrary satellite photos from different sensors or resolutions.

* **EuroSAT Validation Benchmark**: Measures performance on held-out EuroSAT $64 \times 64$ test patches.
* **External Dataset Validation**: Evaluates generalization on custom outside images stored in `external_validation/<ClassName>/`. Run external validation via the UI or backend endpoint `/eval_external` to report separate external accuracy, per-class F1-scores, and confidence distributions.

---

## 🚀 How to Run the Project

### 1. Backend Server (Flask + PyTorch / Keras)
```powershell
cd backend
# Run with virtual environment Python
.\venv\Scripts\python.exe app.py
```
* The API server will start on [http://localhost:5000](http://localhost:5000).

### 2. Frontend Application (React + Vite + Tailwind)
```powershell
cd frontend
npm run dev
```
* Open [http://localhost:5173/](http://localhost:5173/) in your web browser.

---

## 📂 Project Structure

```
project/
├── backend/
│   ├── app.py              # Flask API server & PDF report generator
│   ├── train.py            # 2-Stage ResNet50V2 transfer learning & metrics
│   ├── predict.py          # Multi-crop TTA inference, Grid & Grad-CAM engine
│   ├── change_detection.py # Temporal transition impact rule engine
│   ├── database.py         # SQLite persistence layer (satellite_system.db)
│   ├── eval_external.py    # Out-of-distribution external dataset evaluator
│   ├── model/              # Trained weights (resnet50v2_eurosat.keras) & metrics.json
│   └── static/             # Static uploads, heatmaps, and downloadable PDFs
├── external_validation/    # Custom external test folders per class
├── frontend/
│   ├── index.html          # SPA HTML template
│   ├── package.json        # Dependencies (React, Chart.js, Lucide)
│   └── src/                # React components & Tailwind styles
└── README.md
```

---

## ⚠️ Known Limitations & Scope

1. **Domain Shift**: High-resolution imagery from Google Earth or NAIP may have different spectral signatures compared to Sentinel-2 RGB bands. Multi-crop TTA mitigates scale blur, but domain adaptation fine-tuning on regional data is recommended for specialized geographic regions.
2. **Computational Scale**: Multi-crop TTA and overlapping grid slicing trade off slight inference time for spatial precision.
