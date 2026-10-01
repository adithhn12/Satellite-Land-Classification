import os
os.environ["KERAS_BACKEND"] = "torch"

import json
import cv2
import numpy as np
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score

import predict as pr

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(BASE_DIR, "..")
EXTERNAL_DIR = os.path.join(PROJECT_DIR, "external_validation")

CLASSES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", 
    "Industrial", "Pasture", "PermanentCrop", "Residential", 
    "River", "SeaLake"
]

# Ensure external validation directory structure exists
def ensure_external_dir_structure():
    os.makedirs(EXTERNAL_DIR, exist_ok=True)
    for cname in CLASSES:
        os.makedirs(os.path.join(EXTERNAL_DIR, cname), exist_ok=True)

ensure_external_dir_structure()

OUTPUT_METRICS_PATH = os.path.join(BASE_DIR, "model", "external_metrics.json")

def evaluate_external_dataset():
    """
    Evaluates the model on custom images placed in external_validation/<ClassName>/.
    Reports separate external accuracy, per-class metrics, confusion matrix, and confidence distribution.
    """
    ensure_external_dir_structure()
    
    y_true = []
    y_pred = []
    confidences = []
    image_records = []
    
    for class_idx, class_name in enumerate(CLASSES):
        class_folder = os.path.join(EXTERNAL_DIR, class_name)
        if not os.path.exists(class_folder):
            continue
            
        filenames = os.listdir(class_folder)
        for fname in filenames:
            ext = os.path.splitext(fname)[1].lower()
            if ext not in ['.jpg', '.jpeg', '.png', '.tif', '.tiff']:
                continue
                
            img_path = os.path.join(class_folder, fname)
            try:
                result = pr.predict_single_image(img_path)
                pred_label = result["prediction"]
                pred_idx = CLASSES.index(pred_label) if pred_label in CLASSES else -1
                
                y_true.append(class_idx)
                y_pred.append(pred_idx)
                confidences.append(result["confidence"])
                
                image_records.append({
                    "filename": fname,
                    "true_class": class_name,
                    "predicted_class": pred_label,
                    "confidence": result["confidence"],
                    "is_correct": (pred_label == class_name),
                    "is_low_confidence": result.get("is_low_confidence", False)
                })
            except Exception as e:
                print(f"Error evaluating external image {img_path}: {e}")

    if not y_true:
        return {
            "status": "no_external_data",
            "message": "No external validation images found. Please place custom labeled images in external_validation/<ClassName>/ folders to run external evaluation.",
            "external_dir": EXTERNAL_DIR
        }

    accuracy = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="weighted", zero_division=0)
    per_class_p, per_class_r, per_class_f, _ = precision_recall_fscore_support(y_true, y_pred, average=None, zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=list(range(10)))
    
    avg_conf = float(np.mean(confidences)) if confidences else 0.0
    low_conf_count = sum(1 for rec in image_records if rec["is_low_confidence"])
    
    per_class_summary = {}
    for idx, cname in enumerate(CLASSES):
        per_class_summary[cname] = {
            "precision": f"{per_class_p[idx]*100:.1f}%" if idx < len(per_class_p) else "0.0%",
            "recall": f"{per_class_r[idx]*100:.1f}%" if idx < len(per_class_r) else "0.0%",
            "f1_score": f"{per_class_f[idx]*100:.1f}%" if idx < len(per_class_f) else "0.0%"
        }

    report = {
        "status": "success",
        "total_images": len(y_true),
        "external_accuracy": f"{accuracy * 100:.1f}%",
        "precision": f"{precision * 100:.1f}%",
        "recall": f"{recall * 100:.1f}%",
        "f1_score": f"{f1 * 100:.1f}%",
        "avg_confidence": round(avg_conf, 1),
        "low_confidence_count": low_conf_count,
        "confusion_matrix": cm.tolist(),
        "per_class_metrics": per_class_summary,
        "records": image_records[:100] # Cap preview log
    }
    
    os.makedirs(os.path.dirname(OUTPUT_METRICS_PATH), exist_ok=True)
    with open(OUTPUT_METRICS_PATH, "w") as f:
        json.dump(report, f, indent=4)
        
    return report

def get_external_metrics():
    if os.path.exists(OUTPUT_METRICS_PATH):
        try:
            with open(OUTPUT_METRICS_PATH, "r") as f:
                return json.load(f)
        except Exception:
            pass
    return evaluate_external_dataset()

if __name__ == "__main__":
    res = evaluate_external_dataset()
    print("External Evaluation Summary:", res)
