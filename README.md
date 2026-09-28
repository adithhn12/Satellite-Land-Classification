# Satellite Land Classification and Change Detection System

An AI-powered web application for classifying satellite images into EuroSAT land-cover categories, providing explainability (Saliency Map visualizer), change detection, prediction history logging, analytics charts, and PDF report downloads.

---

## Folder Structure

* `backend/`: Flask server, SQLite database, PyTorch/Keras 3 models, static uploads/heatmaps, and PDF generator.
* `frontend/`: Vite React app styled with Tailwind CSS, using Chart.js for data visualization.
* `dataset/`: Location of the local EuroSAT dataset (dynamically located at `../../EuroSAT`).

---

## Setup and Installation

### 1. Prerequisites
Ensure you have the following installed:
* Python (v3.9 or higher)
* Node.js (v18 or higher) and npm

### 2. Python Packages Installation
Install the required packages in your Python environment:
```bash
python -m pip install keras torch torchvision numpy pandas opencv-python flask flask-cors reportlab scikit-learn
```

### 3. Frontend Packages Installation
Navigate to the frontend folder and install npm packages:
```bash
cd frontend
npm install
```

---

## How to Run

Both servers are currently configured to run on your local machine:

### Start the Flask Backend (Port 5000)
```powershell
cd backend
$env:KERAS_BACKEND="torch"
python app.py
```

### Start the Vite React Frontend (Port 5173)
```bash
cd frontend
npm run dev
```

Open [http://localhost:5173/](http://localhost:5173/) in your web browser.

---

## Key Features

1. **Dashboard Home**: Review classification statistics, active categories, and initiate model training (Quick Demo vs Full).
2. **Land Classification**: Upload single images, view predicted class, confidence, top-3 candidates, and inspect the saliency explainability map overlay.
3. **Batch Prediction**: Process multiple satellite images concurrently, view results, and export them as a CSV spreadsheet.
4. **Change Detection**: Perform side-by-side comparison of historical and recent satellite captures. The system evaluates changes and displays ecological or socio-economic impacts.
5. **Prediction Logs**: Retain all records in an SQLite database. Search, filter, delete entries, and download compiled PDF reports containing satellite visual configurations.
6. **Analytics Panel**: Charts representing land type distributions, training history accuracies, and category frequency logs.
