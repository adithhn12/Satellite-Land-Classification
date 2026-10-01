import os
import sys
import zipfile
import urllib.request

# Dynamic relative base paths
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
BACKEND_DIR = os.path.join(PROJECT_DIR, "backend")
MODEL_DIR = os.path.join(BACKEND_DIR, "model")
PROD_MODEL_PATH = os.path.join(MODEL_DIR, "resnet50v2_eurosat.keras")

# Dataset location checks (1. project/EuroSAT, 2. ../EuroSAT)
LOCAL_PROJECT_DATASET_DIR = os.path.join(PROJECT_DIR, "EuroSAT")
PARENT_WORKSPACE_DATASET_DIR = os.path.normpath(os.path.join(PROJECT_DIR, "..", "EuroSAT"))

REQUIRED_CSV_FILES = ["train.csv", "validation.csv", "test.csv", "label_map.json"]
MIN_MODEL_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB minimum for valid Keras model file

# GitHub Release download URLs (v1.0.0)
RELEASE_TAG = "v1.0.0"
REPO_URL = "https://github.com/adithhn12/Satellite-Land-Classification/releases/download"
MODEL_DOWNLOAD_URL = os.environ.get(
    "MODEL_DOWNLOAD_URL", 
    f"{REPO_URL}/{RELEASE_TAG}/resnet50v2_eurosat.keras"
)
DATASET_DOWNLOAD_URL = os.environ.get(
    "DATASET_DOWNLOAD_URL", 
    f"{REPO_URL}/{RELEASE_TAG}/EuroSAT.zip"
)

def check_valid_model(model_path):
    """Returns True if model_path exists and exceeds minimum expected model file size (10 MB)."""
    if not os.path.exists(model_path):
        return False
    try:
        return os.path.getsize(model_path) >= MIN_MODEL_SIZE_BYTES
    except Exception:
        return False

def check_valid_dataset(target_dir):
    """Returns True if target_dir exists and contains all required CSV and JSON split files."""
    if not os.path.exists(target_dir) or not os.path.isdir(target_dir):
        return False
    for req_file in REQUIRED_CSV_FILES:
        if not os.path.exists(os.path.join(target_dir, req_file)):
            return False
    return True

def download_file(url, target_path, description):
    """Downloads a file with progress reporting and cleanup on failure."""
    print(f"--> Downloading {description} from:\n    {url}")
    os.makedirs(os.path.dirname(target_path), exist_ok=True)
    
    def report_progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            percent = min(100, (downloaded / total_size) * 100)
            mb_downloaded = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            sys.stdout.write(f"\r    Progress: {percent:.1f}% ({mb_downloaded:.1f} MB / {mb_total:.1f} MB)")
            sys.stdout.flush()

    try:
        urllib.request.urlretrieve(url, target_path, reporthook=report_progress)
        print("\n    [OK] Download complete.")
    except Exception as e:
        print(f"\n    [ERROR] Download failed: {e}")
        if os.path.exists(target_path):
            os.remove(target_path)
        raise

def ensure_assets():
    print("==========================================================")
    print("  Satellite Land Classification - Asset Setup & Verifier  ")
    print("==========================================================\n")
    
    # 1. Check Trained Model
    print("[1/2] Checking Trained Production Model...")
    if check_valid_model(PROD_MODEL_PATH):
        file_size_mb = os.path.getsize(PROD_MODEL_PATH) / (1024 * 1024)
        print(f"      [OK] Model found at: {PROD_MODEL_PATH} ({file_size_mb:.1f} MB)")
    else:
        if os.path.exists(PROD_MODEL_PATH):
            print(f"      [!] Incomplete model file detected at: {PROD_MODEL_PATH}. Re-downloading...")
            os.remove(PROD_MODEL_PATH)
        else:
            print(f"      [!] Model missing at: {PROD_MODEL_PATH}")
        download_file(MODEL_DOWNLOAD_URL, PROD_MODEL_PATH, "Production ResNet50V2 Model Weight File")

    # 2. Check EuroSAT Dataset
    print("\n[2/2] Checking EuroSAT Dataset...")
    if check_valid_dataset(LOCAL_PROJECT_DATASET_DIR):
        print(f"      [OK] Dataset found at: {LOCAL_PROJECT_DATASET_DIR}")
    elif check_valid_dataset(PARENT_WORKSPACE_DATASET_DIR):
        print(f"      [OK] Dataset found at parent workspace: {PARENT_WORKSPACE_DATASET_DIR}")
    else:
        print("      [!] No valid EuroSAT dataset found locally.")
        zip_target_path = os.path.join(PROJECT_DIR, "EuroSAT.zip")
        download_file(DATASET_DOWNLOAD_URL, zip_target_path, "EuroSAT Dataset Zip Archive")
        
        print(f"--> Extracting EuroSAT.zip to {LOCAL_PROJECT_DATASET_DIR}...")
        with zipfile.ZipFile(zip_target_path, 'r') as zip_ref:
            zip_ref.extractall(PROJECT_DIR)
        print("    [OK] Extraction complete.")
        
        if os.path.exists(zip_target_path):
            os.remove(zip_target_path)

    print("\n==========================================================")
    print("  [SUCCESS] All assets verified! Application ready to run. ")
    print("==========================================================")

if __name__ == "__main__":
    ensure_assets()
