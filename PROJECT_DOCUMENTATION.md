# Satellite Land Classification & Change Detection System
## Comprehensive Technical Architecture & File Reference Guide (v2.0)

---

## 📂 1. Workspace Root Structure

```
Satellite Land Classification/
├── EuroSAT/               # Local Satellite Image Dataset (RGB)
└── project/               # Main Application Codebase & Deployment Artifacts
```

---

## 🌍 2. Dataset Directory (`EuroSAT/`)

The `EuroSAT` folder contains the Sentinel-2 satellite image dataset divided into 10 European land-cover classes, along with pre-defined train/validation/test CSV split files.

* **`AnnualCrop/`**: 3,000 $64 \times 64$ RGB satellite images of annual crop fields (e.g., wheat, maize).
* **`Forest/`**: 3,000 $64 \times 64$ RGB satellite images of dense woodland and canopy forests.
* **`HerbaceousVegetation/`**: 3,000 $64 \times 64$ RGB satellite images of grasslands, pastures, and shrublands.
* **`Highway/`**: 2,500 $64 \times 64$ RGB satellite images of roads, expressways, and transit infrastructure.
* **`Industrial/`**: 2,500 $64 \times 64$ RGB satellite images of commercial buildings, warehouses, and factories.
* **`Pasture/`**: 2,000 $64 \times 64$ RGB satellite images of livestock grazing lands.
* **`PermanentCrop/`**: 2,500 $64 \times 64$ RGB satellite images of orchards, vineyards, and fruit trees.
* **`Residential/`**: 3,000 $64 \times 64$ RGB satellite images of suburban houses, apartment complexes, and urban housing.
* **`River/`**: 2,500 $64 \times 64$ RGB satellite images of flowing rivers and streams.
* **`SeaLake/`**: 3,000 $64 \times 64$ RGB satellite images of open seas, ocean shorelines, and lakes.
* **`train.csv`**: Contains relative file paths and integer labels for **18,900 training images** (70% split).
* **`validation.csv`**: Contains relative file paths and integer labels for **5,400 validation images** (20% split).
* **`test.csv`**: Contains relative file paths and integer labels for **2,700 evaluation images** (10% split).

---

## 💻 3. Application Directory (`project/`)

```
project/
├── .gitignore               # Git exclusion rules (excludes venv, node_modules, temp files)
├── Dockerfile               # Multi-stage production Docker build configuration
├── docker-compose.yml       # Production container orchestration config
├── README.md                # Project architecture overview & run guide
├── PROJECT_DOCUMENTATION.md # Detailed file-by-file technical documentation
├── external_validation/     # Custom out-of-distribution external dataset validation folders
├── backend/                 # Python Flask & PyTorch/Keras Backend
└── frontend/                # React + Vite + Tailwind CSS Frontend
```

---

## 🐍 4. Backend Directory (`project/backend/`)

The backend is built using Python, Flask, PyTorch, Keras 3, OpenCV, SQLite, and ReportLab.

```
backend/
├── app.py                 # Flask REST API routes, error handlers & PDF generator
├── train.py               # 2-Stage ResNet50V2 transfer learning & data augmentation
├── predict.py             # Inference engine, TTA multi-crop, grid slicer & Grad-CAM
├── change_detection.py    # Temporal land-cover transition & impact rule engine
├── eval_external.py       # Independent external dataset evaluator
├── database.py            # SQLite database access layer (satellite_system.db)
├── satellite_system.db    # SQLite database persistent storage
├── requirements.txt       # Production Python dependency manifest
├── model/                 # Trained model weights & metrics JSON files
├── static/                # Uploaded original files, heatmaps & PDF reports
└── venv/                  # Python virtual environment
```

### File Details:

#### 📜 `app.py`
* **Purpose**: Primary Flask REST API web server.
* **What it does**:
  1. Initializes Flask application, enables CORS (Cross-Origin Resource Sharing), and configures logging.
  2. Ensures static folders (`uploads/`, `heatmaps/`, `reports/`) exist.
  3. Exposes API endpoints:
     - `POST /predict`: Single image classification with Test-Time Augmentation (TTA), multi-crop scale-preserving ensembling, Grad-CAM heatmap generation, and SQLite DB logging.
     - `POST /batch_predict`: Concurrent multi-image processing with uncertainty flags.
     - `POST /change_detection`: Evaluates historical vs. current satellite image pairs and computes ecological transition impact rules.
     - `POST /grid_predict`: Overlapping multi-scale sliding window grid classifier with 3x3 Gaussian spatial probability smoothing.
     - `GET /history` & `DELETE /history/<id>`: Database record retrieval and physical static file cleanup.
     - `GET /analytics`: Returns real stored production benchmark metrics, external evaluation metrics, and prediction trends over time (NO hardcoded fake values).
     - `GET /eval_external`: Evaluates model on images inside `external_validation/<ClassName>/`.
     - `POST /train`: Initiates Production training (`resnet50v2_eurosat.keras`) or Demo trial training (`demo_model.keras`).
     - `GET /download_report/<id>`: Compiles a multi-page PDF summary report using ReportLab.

#### 📜 `train.py`
* **Purpose**: 2-Stage ResNet50V2 transfer learning and satellite data augmentation training pipeline.
* **What it does**:
  1. Sets reproducible random seeds (`keras.utils.set_random_seed(42)`).
  2. Defines `build_model()` with a satellite-tailored **Data Augmentation** pipeline (`RandomFlip`, `RandomRotation 90°`, `RandomZoom`, `RandomContrast`).
  3. Implements **2-Stage Transfer Learning**:
     - *Stage 1 (Head Warmup)*: Base model frozen; classification head trained with Adam (`lr = 1e-3`).
     - *Stage 2 (Fine-Tuning)*: Unfreezes top 30 layers of ResNet50V2; fine-tunes with small learning rate (`lr = 1e-5`) and `ReduceLROnPlateau`.
  4. **Production vs. Demo Separation**:
     - Production mode trains on full EuroSAT dataset splits and saves to `model/resnet50v2_eurosat.keras` and `model/metrics.json`.
     - Demo mode trains a trial model saved to `model/demo_model.keras` without overwriting the production model.
  5. Evaluates real test set metrics (Accuracy, Precision, Recall, F1, Per-class breakdown, Confusion Matrix).

#### 📜 `predict.py`
* **Purpose**: High-performance inference engine, scale-preserving multi-crop ensembling, spatial grid classifier, and Grad-CAM generator.
* **What it does**:
  1. **Safe Model Loading**: `get_or_load_model()` loads the production model file `resnet50v2_eurosat.keras`. Fails with a clear `FileNotFoundError` if no trained model exists (NEVER silently creates an untrained model).
  2. **`predict_single_image()`**: Runs Test-Time Augmentation (TTA) and **5-patch multi-crop tile ensembling** (Center + 4 Corners) on images $>128 \times 128$ px to prevent resolution compression blur.
  3. **Uncertainty Calibration**: Flags predictions with peak confidence $<45\%$ or small class probability margins ($<12\%$) as `is_low_confidence: True`.
  4. **`generate_gradcam_heatmap()`**: Computes PyTorch backward gradients of target class outputs with respect to input pixels to construct a **Grad-CAM Saliency Map overlay**.
  5. **`predict_image_grid()`**: Slices large composite satellite imagery using overlapping sliding windows with a customizable stride. Accumulates probability distributions over a 2D spatial matrix and applies **3x3 Gaussian Spatial Neighborhood Smoothing** (preserving sharp borders for cells $>70\%$ confidence).

#### 📜 `eval_external.py`
* **Purpose**: Independent evaluation framework for out-of-distribution external datasets.
* **What it does**:
  - Scans `external_validation/<ClassName>/` folders for custom external images.
  - Runs inference using `predict_single_image()`, compares predictions against directory labels, and computes external accuracy, per-class metrics, confusion matrices, and confidence distributions.
  - Saves report to `model/external_metrics.json`.

#### 📜 `change_detection.py`
* **Purpose**: Temporal land-use transition and environmental impact rule engine.
* **What it does**:
  - Compares classification signatures between historical and current satellite imagery.
  - Computes socio-environmental impact descriptions (*Deforestation*, *Industrial Expansion*, *Agricultural Land Loss*, *Water Resource Risk*, *Afforestation*).

#### 📜 `database.py`
* **Purpose**: SQLite database layer for persistent record logging (`satellite_system.db`).

#### 📁 `model/`
* **`resnet50v2_eurosat.keras`**: Production ResNet50V2 model weight parameters (~96.6 MB).
* **`metrics.json`**: Real test evaluation metrics (Accuracy, Precision, Recall, F1, Confusion Matrix, History).
* **`demo_model.keras`** & **`demo_metrics.json`**: Temporary trial training artifacts.
* **`external_metrics.json`**: Out-of-distribution external evaluation metrics.

---

## 🎨 5. Frontend Directory (`project/frontend/`)

Built with React 19, Vite, Tailwind CSS, Lucide Icons, and Chart.js.

```
frontend/
├── index.html             # Entry HTML document
├── package.json           # Frontend dependencies & scripts
├── vite.config.js         # Vite bundler config
├── tailwind.config.js     # Tailwind CSS config
└── src/                   # React source code
    ├── main.jsx           # React app DOM mounting point
    ├── index.css          # Global CSS, fonts & animations
    └── App.jsx            # Main Interactive React Dashboard Component
```

### Core Features in `App.jsx`:
1. **Uncertainty Warning Badges**: Displays orange alert tags ("Uncertain / Low Confidence") for predictions with low confidence or ambiguous candidate probability margins.
2. **Grid Cell Inspector**: Overlapping grid slicing view with cell-level confidence inspection, low-confidence cell hatching, and multi-candidate probability sidebars.
3. **Training Controller**: Separate controls for "Run Demo Trial Training" vs "Train Production Model" with confirmation dialogs.
4. **External Dataset Evaluation Panel**: Trigger and view separate out-of-distribution accuracy metrics alongside EuroSAT benchmark metrics.
5. **Batch Processing & CSV Export**: Concurrent multi-image processing with downloadable CSV spreadsheets.
6. **PDF Report Downloads**: ReportLab PDF summaries containing satellite images and Grad-CAM saliency maps.

---

## 🚢 6. Deployment Artifacts

* **[`Dockerfile`](file:///c:/Users/ASUS/Documents/Coding/MCA/Sem%202/Satellite%20Land%20Classification/project/Dockerfile)**: Multi-stage Docker build file compiling Vite frontend static assets and hosting them via Gunicorn WSGI Python server.
* **[`docker-compose.yml`](file:///c:/Users/ASUS/Documents/Coding/MCA/Sem%202/Satellite%20Land%20Classification/project/docker-compose.yml)**: Docker Compose orchestration for local container testing or cloud deployment (Render, Railway, AWS ECS).
* **[`backend/requirements.txt`](file:///c:/Users/ASUS/Documents/Coding/MCA/Sem%202/Satellite%20Land%20Classification/project/backend/requirements.txt)**: Pinned production dependencies (`torch`, `keras`, `opencv-python-headless`, `reportlab`, `gunicorn`).

---

## 🛠️ Summary Architecture Diagram

```mermaid
flowchart TD
    User([User Browser]) <--> Frontend[React 19 + Vite Frontend (Port 5173)]
    Frontend <--> API[Flask REST API Server / Gunicorn (Port 5000)]
    API <--> Predict[predict.py / TTA Multi-Crop & Overlapping Grid Engine]
    API <--> Train[train.py / 2-Stage ResNet50V2 Transfer Learning]
    API <--> ExtEval[eval_external.py / External Dataset Evaluator]
    API <--> Change[change_detection.py / Transition Impact Engine]
    API <--> DB[(SQLite Database / satellite_system.db)]
    Predict <--> Heatmap[PyTorch Grad-CAM Saliency Generator]
    Train <--> EuroSAT[(EuroSAT Dataset / 27,000 Images)]
    ExtEval <--> ExtDir[(external_validation/ Directory)]
    API <--> ReportLab[ReportLab PDF Generator]
```
