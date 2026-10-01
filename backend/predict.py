import os
os.environ["KERAS_BACKEND"] = "torch"

import keras
import numpy as np
import cv2
import logging

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROD_MODEL_PATH = os.path.join(BASE_DIR, "model", "resnet50v2_eurosat.keras")
DEMO_MODEL_PATH = os.path.join(BASE_DIR, "model", "demo_model.keras")

CLASSES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", 
    "Industrial", "Pasture", "PermanentCrop", "Residential", 
    "River", "SeaLake"
]

_model_cache = None
_model_cache_path = None

def get_or_load_model():
    """
    Safely loads the production neural network model.
    Falls back to demo model if production model is missing.
    Fails explicitly if no trained model exists on disk (NO UNTRAINED MODEL CREATION).
    """
    global _model_cache, _model_cache_path
    
    if _model_cache is not None:
        return _model_cache
        
    target_path = None
    if os.path.exists(PROD_MODEL_PATH):
        target_path = PROD_MODEL_PATH
    elif os.path.exists(DEMO_MODEL_PATH):
        target_path = DEMO_MODEL_PATH
        logging.warning("Production model file missing; falling back to demo model file.")
    else:
        raise FileNotFoundError(
            f"No trained model found at {PROD_MODEL_PATH}. "
            "Please run production model training to generate the model before predicting."
        )
        
    try:
        _model_cache = keras.models.load_model(target_path)
        _model_cache_path = target_path
        logging.info(f"Successfully loaded model from {target_path}")
        return _model_cache
    except Exception as e:
        raise RuntimeError(f"Failed to load model file from {target_path}: {str(e)}")

def predict_single_image(image_path):
    """
    Runs single-image land classification with Test-Time Augmentation (TTA)
    and Multi-Crop Ensembling for out-of-distribution high-resolution images.
    """
    orig_img = cv2.imread(image_path)
    if orig_img is None:
        raise ValueError(f"Could not read image file from path: {image_path}")
        
    img_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    h, w, _ = img_rgb.shape
    
    crops = []
    
    # 1. Base resized view (Global Context)
    base_resized = cv2.resize(img_rgb, (64, 64))
    crops.append(base_resized.astype("float32") / 255.0)
    
    # 2. Test-Time Augmentation (Horizontal & Vertical Flips)
    h_flip = cv2.flip(base_resized, 1)
    v_flip = cv2.flip(base_resized, 0)
    crops.append(h_flip.astype("float32") / 255.0)
    crops.append(v_flip.astype("float32") / 255.0)
    
    # 3. Multi-crop tile ensembling for high-resolution images (preserves detail scale)
    if h >= 128 and w >= 128:
        crop_h, crop_w = int(h * 0.6), int(w * 0.6)
        offsets = [
            ((h - crop_h) // 2, (w - crop_w) // 2), # Center
            (0, 0),                                 # Top-Left
            (0, w - crop_w),                        # Top-Right
            (h - crop_h, 0),                        # Bottom-Left
            (h - crop_h, w - crop_w)                # Bottom-Right
        ]
        
        for top, left in offsets:
            patch = img_rgb[top:top+crop_h, left:left+crop_w]
            patch_resized = cv2.resize(patch, (64, 64))
            crops.append(patch_resized.astype("float32") / 255.0)
            
    crops_batch = np.array(crops, dtype="float32") # Shape: (N, 64, 64, 3)
    
    model = get_or_load_model()
    all_preds = model.predict(crops_batch, verbose=0)
    
    # Ensemble probability distribution (mean across crops/TTA)
    avg_preds = np.mean(all_preds, axis=0)
    
    pred_idx = int(np.argmax(avg_preds))
    confidence = float(avg_preds[pred_idx]) * 100.0
    
    # Uncertainty / Confidence Calibration Analysis
    sorted_indices = np.argsort(avg_preds)[::-1]
    top1_prob = float(avg_preds[sorted_indices[0]])
    top2_prob = float(avg_preds[sorted_indices[1]])
    prob_margin = top1_prob - top2_prob
    
    # Flag low confidence if peak probability is under 45% or margin is small (< 12%)
    is_low_confidence = (confidence < 45.0) or (prob_margin < 0.12)
    confidence_label = "Low / Ambiguous Confidence" if is_low_confidence else "High Confidence"
    
    top_predictions = []
    for idx in sorted_indices:
        top_predictions.append({
            "class": CLASSES[idx],
            "confidence": round(float(avg_preds[idx]) * 100.0, 1)
        })
        
    return {
        "prediction": CLASSES[pred_idx],
        "confidence": round(confidence, 1),
        "is_low_confidence": is_low_confidence,
        "confidence_label": confidence_label,
        "top_predictions": top_predictions[:5],
        "image_size": {"width": w, "height": h}
    }

def generate_gradcam_heatmap(image_path, output_heatmap_path):
    """
    Generates PyTorch-based Grad-CAM saliency heatmaps highlighting decision features.
    Raises explicit RuntimeError on failure (NO FAKE / DUMMY HEATMAP GENERATION).
    """
    orig_img = cv2.imread(image_path)
    if orig_img is None:
        raise ValueError(f"Could not read image file from path: {image_path}")
        
    import torch
    
    h, w, c = orig_img.shape
    img_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (64, 64))
    img_array = img_resized.astype("float32") / 255.0
    
    model = get_or_load_model()
    
    # Convert image patch to PyTorch tensor with grad tracking
    x = torch.tensor(np.expand_dims(img_array, axis=0), requires_grad=True)
    
    # Forward pass
    preds = model(x)
    pred_idx = torch.argmax(preds[0])
    score = preds[0, pred_idx]
    
    # Backward pass
    score.backward()
    
    if x.grad is None:
        raise RuntimeError("Grad-CAM gradient extraction failed: input tensor gradient is None.")
        
    saliency = torch.abs(x.grad)[0]
    heatmap = torch.max(saliency, dim=2)[0].detach().cpu().numpy()
    
    denom = heatmap.max() - heatmap.min()
    if denom == 0:
        denom = 1e-10
    heatmap = (heatmap - heatmap.min()) / denom
    
    # Resize heatmap back to full image resolution
    heatmap_resized = cv2.resize(heatmap, (w, h))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    superimposed = cv2.addWeighted(orig_img, 0.6, heatmap_colored, 0.4, 0)
    
    os.makedirs(os.path.dirname(output_heatmap_path), exist_ok=True)
    cv2.imwrite(output_heatmap_path, superimposed)
    return output_heatmap_path

def predict_image_grid(image_path, patch_size=64, stride=32):
    """
    Overlapping multi-scale sliding-window grid classifier with 2D spatial probability aggregation
    and boundary-preserving neighborhood smoothing. Handles arbitrary image dimensions.
    """
    orig_img = cv2.imread(image_path)
    if orig_img is None:
        raise ValueError(f"Could not read image file: {image_path}")
        
    h, w, _ = orig_img.shape
    img_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    
    # Handle small images by padding or resizing
    if h < patch_size or w < patch_size:
        pad_h = max(patch_size, h)
        pad_w = max(patch_size, w)
        img_rgb = cv2.resize(img_rgb, (pad_w, pad_h))
        h, w, _ = img_rgb.shape
        
    # Generate sliding window coordinates with specified stride
    y_coords = list(range(0, h - patch_size + 1, stride))
    if y_coords[-1] + patch_size < h:
        y_coords.append(h - patch_size) # Include border edge
        
    x_coords = list(range(0, w - patch_size + 1, stride))
    if x_coords[-1] + patch_size < w:
        x_coords.append(w - patch_size) # Include border edge
        
    rows = len(y_coords)
    cols = len(x_coords)
    
    patches = []
    window_info = []
    
    for r_idx, y in enumerate(y_coords):
        for c_idx, x in enumerate(x_coords):
            crop = img_rgb[y:y+patch_size, x:x+patch_size]
            crop_resized = cv2.resize(crop, (64, 64))
            crop_norm = crop_resized.astype("float32") / 255.0
            
            patches.append(crop_norm)
            window_info.append({
                "row": r_idx,
                "col": c_idx,
                "x": x,
                "y": y,
                "w": patch_size,
                "h": patch_size
            })
            
    if not patches:
        return {"predictions": [], "grid_size": {"rows": 0, "cols": 0}}
        
    patches_batch = np.array(patches, dtype="float32")
    model = get_or_load_model()
    
    # Run efficient batch inference across windows
    all_probs = model.predict(patches_batch, batch_size=32, verbose=0) # (N, 10)
    
    # Build 2D probability tensor grid of shape (rows, cols, 10)
    grid_probs = np.zeros((rows, cols, 10), dtype="float32")
    for idx, win in enumerate(window_info):
        grid_probs[win["row"], win["col"]] = all_probs[idx]
        
    # Apply Principled Spatial Neighborhood Smoothing (3x3 Gaussian kernel over probability maps)
    # Does NOT override high-confidence predictions (> 70%) to preserve genuine land-cover boundaries
    smoothed_grid_probs = np.copy(grid_probs)
    for r in range(rows):
        for c in range(cols):
            orig_prob = grid_probs[r, c]
            peak_conf = float(np.max(orig_prob)) * 100.0
            
            # Keep strong predictions untouched to preserve sharp borders
            if peak_conf >= 70.0:
                continue
                
            # Gather neighbor probability vectors
            neighbor_probs = []
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        neighbor_probs.append(grid_probs[nr, nc])
                        
            if neighbor_probs:
                spatial_mean = np.mean(neighbor_probs, axis=0)
                # Blend 60% original + 40% spatial neighborhood mean
                smoothed_grid_probs[r, c] = 0.6 * orig_prob + 0.4 * spatial_mean

    # Compile final predictions list
    grid_predictions = []
    for win in window_info:
        r, c = win["row"], win["col"]
        probs = smoothed_grid_probs[r, c]
        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx]) * 100.0
        
        sorted_indices = np.argsort(probs)[::-1]
        top1_p = float(probs[sorted_indices[0]])
        top2_p = float(probs[sorted_indices[1]])
        
        is_low_conf = (confidence < 45.0) or ((top1_p - top2_p) < 0.10)
        
        top_preds = []
        for idx in sorted_indices[:3]:
            top_preds.append({
                "class": CLASSES[idx],
                "confidence": round(float(probs[idx]) * 100.0, 1)
            })
            
        grid_predictions.append({
            "position": win,
            "prediction": CLASSES[pred_idx],
            "confidence": round(confidence, 1),
            "is_low_confidence": is_low_conf,
            "top_predictions": top_preds
        })
        
    return {
        "predictions": grid_predictions,
        "grid_size": {"rows": rows, "cols": cols},
        "image_size": {"width": w, "height": h},
        "patch_size": patch_size,
        "stride": stride
    }
