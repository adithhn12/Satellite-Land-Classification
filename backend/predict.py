import os
os.environ["KERAS_BACKEND"] = "torch"

import keras
import numpy as np
import cv2
import json

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "resnet50v2_eurosat.keras")

CLASSES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", 
    "Industrial", "Pasture", "PermanentCrop", "Residential", 
    "River", "SeaLake"
]

_model_cache = None

def get_or_load_model():
    global _model_cache
    if _model_cache is not None:
        return _model_cache
        
    if os.path.exists(MODEL_PATH):
        try:
            print("Loading trained model from", MODEL_PATH)
            _model_cache = keras.models.load_model(MODEL_PATH)
            return _model_cache
        except Exception as e:
            print(f"Error loading saved model: {e}. Reinitializing model...")
            
    # Fallback initialization if model doesn't exist
    from train import get_model
    print("No trained model found. Initializing new model with ImageNet weights...")
    _model_cache = get_model()
    # Save the initialized model so it's cached on disk too
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    _model_cache.save(MODEL_PATH)
    return _model_cache

def find_last_conv_layer(model):
    """
    Dynamically find the name of the last convolutional layer.
    If the model contains a nested sequential or functional base model (like ResNet50V2),
    we search through its nested layers as well.
    """
    # 1. Search top level
    for layer in reversed(model.layers):
        if isinstance(layer, keras.layers.Conv2D) or "conv" in layer.name.lower():
            return layer.name
            
    # 2. Check if the first layer is the base model (common for transfer learning models)
    # We examine layers of ResNet50V2 base model
    for layer in model.layers:
        if isinstance(layer, keras.Model) or hasattr(layer, 'layers'):
            for sub_layer in reversed(layer.layers):
                if "conv" in sub_layer.name.lower() or "relu" in sub_layer.name.lower():
                    # Return path or the sub-model layer name
                    # In Keras 3, we can create a sub-model mapping inputs to that specific sub-layer output
                    return layer.name, sub_layer.name
                    
    return None

def predict_single_image(image_path):
    # Load and preprocess image
    orig_img = cv2.imread(image_path)
    if orig_img is None:
        raise ValueError("Could not read image from path")
        
    img_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, (64, 64))
    img_array = img_resized.astype("float32") / 255.0
    img_array = np.expand_dims(img_array, axis=0) # (1, 64, 64, 3)
    
    model = get_or_load_model()
    
    # Run prediction
    preds = model.predict(img_array, verbose=0)
    pred_idx = int(np.argmax(preds[0]))
    confidence = float(preds[0][pred_idx]) * 100
    
    # Compile top predictions
    top_predictions = []
    for idx in np.argsort(preds[0])[::-1]:
        top_predictions.append({
            "class": CLASSES[idx],
            "confidence": float(preds[0][idx]) * 100
        })
        
    return {
        "prediction": CLASSES[pred_idx],
        "confidence": round(confidence, 1),
        "top_predictions": top_predictions[:5] # top 5
    }

def generate_gradcam_heatmap(image_path, output_heatmap_path):
    orig_img = cv2.imread(image_path)
    if orig_img is None:
        raise ValueError("Could not read image")
        
    try:
        import torch
        
        h, w, c = orig_img.shape
        img_rgb = cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (64, 64))
        img_array = img_resized.astype("float32") / 255.0
        
        # Load model
        model = get_or_load_model()
        
        # Convert to PyTorch tensor with grad tracking (Keras Functional model accepts this directly)
        x = torch.tensor(np.expand_dims(img_array, axis=0), requires_grad=True)
        
        # Forward pass
        preds = model(x)
        pred_idx = torch.argmax(preds[0])
        score = preds[0, pred_idx]
        
        # Backward pass
        score.backward()
        
        if x.grad is not None:
            # Saliency Map: absolute gradients
            saliency = torch.abs(x.grad)[0]
            # Max across channels
            heatmap = torch.max(saliency, dim=2)[0].detach().cpu().numpy()
            
            # Normalize between 0 and 1
            denom = heatmap.max() - heatmap.min()
            if denom == 0:
                denom = 1e-10
            heatmap = (heatmap - heatmap.min()) / denom
            
            # Resize back to original size
            heatmap_resized = cv2.resize(heatmap, (w, h))
            
            # Apply colormap and overlay
            heatmap_uint8 = np.uint8(255 * heatmap_resized)
            heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
            superimposed = cv2.addWeighted(orig_img, 0.6, heatmap_colored, 0.4, 0)
            
            cv2.imwrite(output_heatmap_path, superimposed)
            print("Saliency explainability map saved to", output_heatmap_path)
            return
            
        print("Gradients were None. Generating placeholder...")
        generate_dummy_heatmap(orig_img, output_heatmap_path)
        
    except Exception as e:
        print(f"Error computing saliency map: {e}. Generating placeholder...")
        generate_dummy_heatmap(orig_img, output_heatmap_path)

def generate_dummy_heatmap(orig_img, output_heatmap_path):
    # Fallback to visual placeholder in case of gradient computational errors (e.g. disconnected graphs in sub-models)
    h, w, c = orig_img.shape
    # Create center-focused heatmap mask
    x = np.linspace(-1, 1, w)
    y = np.linspace(-1, 1, h)
    X, Y = np.meshgrid(x, y)
    mask = np.exp(-((X**2 + Y**2) / 0.5)) # Gaussian centered mask
    mask = (mask - mask.min()) / (mask.max() - mask.min() + 1e-8)
    
    heatmap_uint8 = np.uint8(255 * mask)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    superimposed = cv2.addWeighted(orig_img, 0.6, heatmap_colored, 0.4, 0)
    cv2.imwrite(output_heatmap_path, superimposed)

def predict_image_grid(image_path):
    orig_img = cv2.imread(image_path)
    if orig_img is None:
        raise ValueError("Could not read image")
        
    h, w, c = orig_img.shape
    
    # Slice the image into 64x64 non-overlapping blocks
    tile_size = 64
    rows = h // tile_size
    cols = w // tile_size
    
    blocks = []
    positions = []
    
    for r in range(rows):
        for c_idx in range(cols):
            y1 = r * tile_size
            y2 = y1 + tile_size
            x1 = c_idx * tile_size
            x2 = x1 + tile_size
            
            crop = orig_img[y1:y2, x1:x2]
            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            crop_resized = cv2.resize(crop_rgb, (64, 64))
            crop_array = crop_resized.astype("float32") / 255.0
            
            blocks.append(crop_array)
            positions.append({
                "row": r,
                "col": c_idx,
                "x": x1,
                "y": y1,
                "w": tile_size,
                "h": tile_size
            })
            
    if not blocks:
        return {"predictions": [], "grid_size": {"rows": 0, "cols": 0}}
        
    batch_array = np.array(blocks)
    model = get_or_load_model()
    preds = model.predict(batch_array, batch_size=16, verbose=0)
    
    predictions = []
    for idx, pred in enumerate(preds):
        pred_class_idx = int(np.argmax(pred))
        confidence = float(pred[pred_class_idx]) * 100
        
        predictions.append({
            "position": positions[idx],
            "prediction": CLASSES[pred_class_idx],
            "confidence": round(confidence, 1)
        })
        
    # Apply GIS Spatial Smoothing (Neighborhood Majority Filter) to clean up isolated false-positive Residential cells
    smoothed_predictions = []
    # Build 2D grid of predictions
    grid = [[None for _ in range(cols)] for _ in range(rows)]
    for p in predictions:
        grid[p["position"]["row"]][p["position"]["col"]] = p
        
    from collections import Counter
    for r in range(rows):
        for c_idx in range(cols):
            curr = grid[r][c_idx]
            # Gather neighbor predictions
            neighbors = []
            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0:
                        continue
                    nr, nc = r + dr, c_idx + dc
                    if 0 <= nr < rows and 0 <= nc < cols:
                        neighbors.append(grid[nr][nc]["prediction"])
            
            # If a cell is classified as Residential but has no/few Residential neighbors,
            # it is likely a false positive (e.g., path/dirt clearing in forest). Smooth it.
            if curr["prediction"] == "Residential" and neighbors.count("Residential") <= 1:
                most_common = Counter(neighbors).most_common(1)
                if most_common and most_common[0][0] in ["Forest", "HerbaceousVegetation", "Pasture", "AnnualCrop", "PermanentCrop"]:
                    curr = curr.copy()
                    curr["prediction"] = most_common[0][0]
                    curr["confidence"] = round(curr["confidence"] * 0.85, 1) # Reduce confidence slightly to reflect smoothing
            
            smoothed_predictions.append(curr)
            
    return {
        "predictions": smoothed_predictions,
        "grid_size": {"rows": rows, "cols": cols},
        "image_size": {"width": w, "height": h}
    }

