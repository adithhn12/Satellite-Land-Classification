import os
os.environ["KERAS_BACKEND"] = "torch"

import keras
import numpy as np
import pandas as pd
import cv2
import json
from sklearn.metrics import precision_recall_fscore_support, confusion_matrix, accuracy_score

# Set reproducible random seeds
keras.utils.set_random_seed(42)

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_DATASET_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "EuroSAT"))
PARENT_DATASET_DIR = os.path.normpath(os.path.join(BASE_DIR, "..", "..", "EuroSAT"))
DATASET_DIR = LOCAL_DATASET_DIR if os.path.exists(os.path.join(LOCAL_DATASET_DIR, "train.csv")) else PARENT_DATASET_DIR

MODEL_DIR = os.path.join(BASE_DIR, "model")
os.makedirs(MODEL_DIR, exist_ok=True)

PROD_MODEL_PATH = os.path.join(MODEL_DIR, "resnet50v2_eurosat.keras")
PROD_METRICS_PATH = os.path.join(MODEL_DIR, "metrics.json")

DEMO_MODEL_PATH = os.path.join(MODEL_DIR, "demo_model.keras")
DEMO_METRICS_PATH = os.path.join(MODEL_DIR, "demo_metrics.json")

# Class names and labels
CLASSES = [
    "AnnualCrop", "Forest", "HerbaceousVegetation", "Highway", 
    "Industrial", "Pasture", "PermanentCrop", "Residential", 
    "River", "SeaLake"
]

def load_image(img_relative_path, target_size=(64, 64)):
    img_path = os.path.join(DATASET_DIR, img_relative_path.replace('/', os.sep))
    if not os.path.exists(img_path):
        return None
    img = cv2.imread(img_path)
    if img is None:
        return None
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, target_size)
    img = img.astype("float32") / 255.0
    return img

def load_data_from_csv(csv_filename, sample_size=None):
    csv_path = os.path.join(DATASET_DIR, csv_filename)
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV dataset mapping not found at {csv_path}")
        
    df = pd.read_csv(csv_path)
    
    if sample_size is not None:
        df = df.groupby("ClassName").apply(lambda x: x.sample(min(len(x), sample_size), random_state=42)).reset_index(drop=True)
    
    images = []
    labels = []
    
    for idx, row in df.iterrows():
        img = load_image(row['Filename'])
        if img is not None:
            images.append(img)
            labels.append(row['Label'])
            
    return np.array(images, dtype="float32"), np.array(labels, dtype="int64")

def build_model(input_shape=(64, 64, 3), num_classes=10, unfreeze_layers=0):
    inputs = keras.Input(shape=input_shape)
    
    # Satellite Data Augmentation Pipeline
    # Using spatial rotations (0, 90, 180, 270 deg), flips, subtle contrast/zoom
    augmented = keras.Sequential([
        keras.layers.RandomFlip("horizontal_and_vertical"),
        keras.layers.RandomRotation(0.25), # 90 degree random rotations
        keras.layers.RandomZoom(0.1),
        keras.layers.RandomContrast(0.1),
    ], name="satellite_augmentation")(inputs)
    
    # Rescaling input [0, 1] -> [-1, 1] as required by ResNet50V2
    x = keras.layers.Rescaling(scale=2.0, offset=-1.0)(augmented)
    
    # Base ResNet50V2 model with ImageNet pre-trained weights
    base_model = keras.applications.ResNet50V2(
        input_shape=input_shape,
        include_top=False,
        weights="imagenet"
    )
    
    if unfreeze_layers > 0:
        base_model.trainable = True
        for layer in base_model.layers[:-unfreeze_layers]:
            layer.trainable = False
    else:
        base_model.trainable = False
        
    x = base_model(x, training=False)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(256, activation="relu")(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.35)(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)
    
    model = keras.Model(inputs, outputs)
    return model

def train_model(quick_demo=True, is_production=False):
    """
    Trains the satellite land cover model.
    - If is_production=True (or quick_demo=False), trains production model and saves to PROD_MODEL_PATH.
    - If quick_demo=True and is_production=False, trains trial demo model and saves to DEMO_MODEL_PATH.
    """
    target_model_path = PROD_MODEL_PATH if (is_production or not quick_demo) else DEMO_MODEL_PATH
    target_metrics_path = PROD_METRICS_PATH if (is_production or not quick_demo) else DEMO_METRICS_PATH
    mode_name = "Production Model" if (is_production or not quick_demo) else "Quick Demo Trial"
    
    print(f"--> Starting {mode_name} training...")
    
    if quick_demo and not is_production:
        x_train, y_train = load_data_from_csv("train.csv", sample_size=30)
        x_val, y_val = load_data_from_csv("validation.csv", sample_size=15)
        x_test, y_test = load_data_from_csv("test.csv", sample_size=15)
        epochs_head = 5
        epochs_fine = 0
        batch_size = 16
        lr_head = 1e-3
    else:
        # Full or larger balanced subset
        x_train, y_train = load_data_from_csv("train.csv", sample_size=400)
        x_val, y_val = load_data_from_csv("validation.csv", sample_size=100)
        x_test, y_test = load_data_from_csv("test.csv", sample_size=100)
        epochs_head = 6
        epochs_fine = 6
        batch_size = 32
        lr_head = 1e-3

    print(f"Dataset Loaded: Train shape={x_train.shape}, Val shape={x_val.shape}, Test shape={x_test.shape}")
    
    # Stage 1: Train Head with frozen base
    model = build_model(unfreeze_layers=0)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=lr_head),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(filepath=target_model_path, save_best_only=True, monitor="val_loss")
    ]
    
    history1 = model.fit(
        x_train, y_train,
        validation_data=(x_val, y_val),
        epochs=epochs_head,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=1
    )
    
    combined_history = {
        "accuracy": [float(v) for v in history1.history["accuracy"]],
        "val_accuracy": [float(v) for v in history1.history["val_accuracy"]],
        "loss": [float(v) for v in history1.history["loss"]],
        "val_loss": [float(v) for v in history1.history["val_loss"]]
    }
    
    # Stage 2: Fine-Tuning top 30 layers if requested
    if epochs_fine > 0:
        print("--> Stage 2: Fine-tuning top layers of ResNet50V2...")
        model = build_model(unfreeze_layers=30)
        # Load best weights from Stage 1
        model.load_weights(target_model_path)
        
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-5),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"]
        )
        
        callbacks_fine = [
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=2),
            keras.callbacks.EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
            keras.callbacks.ModelCheckpoint(filepath=target_model_path, save_best_only=True, monitor="val_loss")
        ]
        
        history2 = model.fit(
            x_train, y_train,
            validation_data=(x_val, y_val),
            epochs=epochs_fine,
            batch_size=batch_size,
            callbacks=callbacks_fine,
            verbose=1
        )
        
        for k in combined_history:
            combined_history[k].extend([float(v) for v in history2.history[k]])
            
    # Save final best model
    model.save(target_model_path)
    print(f"Model saved to {target_model_path}")
    
    # Evaluate on test set
    y_pred_probs = model.predict(x_test, verbose=0)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    test_acc = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average="weighted", zero_division=0)
    per_class_prec, per_class_rec, per_class_f1, _ = precision_recall_fscore_support(y_test, y_pred, average=None, zero_division=0)
    
    cm = confusion_matrix(y_test, y_pred, labels=list(range(10)))
    
    per_class_metrics = {}
    for idx, cname in enumerate(CLASSES):
        per_class_metrics[cname] = {
            "precision": f"{per_class_prec[idx]*100:.1f}%",
            "recall": f"{per_class_rec[idx]*100:.1f}%",
            "f1": f"{per_class_f1[idx]*100:.1f}%"
        } if idx < len(per_class_prec) else {}

    metrics = {
        "evaluated": True,
        "is_production": (is_production or not quick_demo),
        "accuracy": f"{test_acc * 100:.1f}%",
        "precision": f"{precision * 100:.1f}%",
        "recall": f"{recall * 100:.1f}%",
        "f1_score": f"{f1 * 100:.1f}%",
        "confusion_matrix": cm.tolist(),
        "per_class_metrics": per_class_metrics,
        "history": combined_history,
        "test_samples": len(y_test),
        "train_samples": len(y_train)
    }
    
    with open(target_metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
        
    print(f"Metrics saved to {target_metrics_path}")
    return metrics

def get_production_metrics():
    """
    Returns saved production model metrics. Returns unevaluated dictionary if missing (NO FAKE METRICS!).
    """
    if os.path.exists(PROD_METRICS_PATH):
        try:
            with open(PROD_METRICS_PATH, "r") as f:
                data = json.load(f)
                data["evaluated"] = True
                return data
        except Exception as e:
            print("Error loading metrics:", e)
            
    return {
        "evaluated": False,
        "message": "Production model evaluation metrics unavailable. Please run production model training to evaluate."
    }

if __name__ == "__main__":
    train_model(quick_demo=True, is_production=True)
