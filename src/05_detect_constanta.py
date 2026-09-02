import os
import zipfile
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import tifffile # <-- NEW IMPORT

def process_and_detect():
    # Setup paths
    raw_s1_dir = Path("data/raw/sentinel1")
    processed_dir = Path("data/processed/sentinel1_png")
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    print("1. Looking for Sentinel-1 raw data...")
    zip_files = list(raw_s1_dir.glob("*.zip"))
    if not zip_files:
        print("Error: No Sentinel-1 ZIP file found.")
        return
    
    target_zip = zip_files[0]
    extract_path = raw_s1_dir / target_zip.stem
    
    # Extract the ZIP if we haven't already
    if not extract_path.exists():
        print(f"2. Extracting {target_zip.name} (This might take a minute)...")
        with zipfile.ZipFile(target_zip, 'r') as zip_ref:
            zip_ref.extractall(raw_s1_dir)
    else:
        print("2. Sentinel-1 data already extracted.")
    
    # Find the VV polarization TIFF (best for ship detection)
    tiff_files = list(extract_path.rglob("measurement/*-vv-*.tiff"))
    if not tiff_files:
        print("Error: Could not find the VV measurement TIFF inside the SAFE folder.")
        return
        
    target_tiff = tiff_files[0]
    png_out_path = processed_dir / f"{target_zip.stem}_VV.png"
    
    # Convert the massive 16-bit TIFF into a usable PNG
    if not png_out_path.exists():
        print("3. Converting 16-bit SAR TIFF to 8-bit PNG...")
        
        # --- FIX: Using tifffile to handle ZSTD compressed Sentinel-1 data ---
        img = tifffile.imread(str(target_tiff))
        
        print("   Resizing image for memory safety...")
        scale_percent = 25 # Scale down to 25% of original size
        width = int(img.shape[1] * scale_percent / 100)
        height = int(img.shape[0] * scale_percent / 100)
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_AREA)

        # Normalize the pixel values (stretching contrast to make ships visible)
        p2, p98 = np.percentile(img, (2, 98))
        img_norm = np.clip(img, p2, p98)
        img_norm = (img_norm - p2) / (p98 - p2) * 255.0
        img_norm = img_norm.astype(np.uint8)
        
        cv2.imwrite(str(png_out_path), img_norm)
        print(f"   Saved processed PNG to {png_out_path}")
    else:
        print("3. Processed PNG already exists.")

    print("\n4. Loading our trained YOLOv8 model...")
    # Using the exact path generated during our training phase
    model_path = "runs/detect/models/yolov8_ship_detector_v1/weights/best.pt"
    
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}")
        return

    model = YOLO(model_path)
    
    print("\n5. Running Ship Detection on Constanța Port...")
    # Run the prediction and save the visual map
    results = model.predict(
        source=str(png_out_path),
        save=True,
        conf=0.25, # Confidence threshold
        name="constanta_ship_detections",
        project="data/processed"
    )
    
    print("\nSUCCESS! Pipeline complete end-to-end.")
    print("Check the 'data/processed/constanta_ship_detections' folder for your final mapped image!")

if __name__ == "__main__":
    process_and_detect()