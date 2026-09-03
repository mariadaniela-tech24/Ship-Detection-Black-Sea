import os
import zipfile
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import tifffile 

def process_and_detect():
    raw_s1_dir = Path("data/raw/sentinel1")
    processed_dir = Path("data/processed/sentinel1_tiles")
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    print("1. Looking for Sentinel-1 raw data...")
    zip_files = list(raw_s1_dir.glob("*.zip"))
    if not zip_files:
        print("Error: No Sentinel-1 ZIP file found.")
        return
    
    target_zip = zip_files[0]
    extract_path = raw_s1_dir / target_zip.stem
    
    if not extract_path.exists():
        print(f"2. Extracting {target_zip.name}...")
        with zipfile.ZipFile(target_zip, 'r') as zip_ref:
            zip_ref.extractall(raw_s1_dir)
    else:
        print("2. Sentinel-1 data already extracted.")
    
    tiff_files = list(extract_path.rglob("measurement/*-vv-*.tiff"))
    if not tiff_files:
        print("Error: Could not find the VV measurement TIFF.")
        return
        
    target_tiff = tiff_files[0]
    
    print("3. Loading our trained YOLOv8 model...")
    model_path = "runs/detect/models/yolov8_ship_detector_v2/weights/best.pt"
    if not os.path.exists(model_path):
        print(f"Error: Model not found at {model_path}.")
        return
    model = YOLO(model_path)

    print("4. Reading massive SAR image into memory...")
    img = tifffile.imread(str(target_tiff))
    img_h, img_w = img.shape
    print(f"   Original image size: {img_w}x{img_h} pixels!")

    print("\n5. Starting Sliding Window Scanner...")
    tile_size = 1024  # Size of the chunks we will feed to YOLO
    ships_found = 0
    
    # Scan across the image in a grid
    for y in range(0, img_h, tile_size):
        for x in range(0, img_w, tile_size):
            # Cut out the tile
            tile = img[y:y+tile_size, x:x+tile_size]
            
            # Skip tiles that are too small (edges of the image)
            if tile.shape[0] < 512 or tile.shape[1] < 512:
                continue
                
            # Normalize the radar data for this specific tile
            p2, p98 = np.percentile(tile, (2, 98))
            if p98 == p2: # Skip empty black tiles
                continue
                
            tile_norm = np.clip(tile, p2, p98)
            tile_norm = (tile_norm - p2) / (p98 - p2) * 255.0
            tile_norm = tile_norm.astype(np.uint8)
            
            # Convert to 3-channel image as YOLO expects RGB format
            tile_rgb = cv2.cvtColor(tile_norm, cv2.COLOR_GRAY2RGB)
            
            # Run YOLO on the tile
            results = model.predict(source=tile_rgb, conf=0.25, verbose=False)
            
            # If YOLO found a bounding box (a ship), save this tile!
            if len(results[0].boxes) > 0:
                ships_found += len(results[0].boxes)
                save_path = processed_dir / f"ship_tile_y{y}_x{x}.jpg"
                
                # Draw the boxes and save
                annotated_tile = results[0].plot() 
                cv2.imwrite(str(save_path), annotated_tile)
                print(f"   [!] Found ships at Y:{y}, X:{x} -> Saved tile.")

    print(f"\nSUCCESS! Scanner finished. Found {ships_found} total ships.")
    print(f"Check the '{processed_dir}' folder to see the cropped images with bounding boxes!")

if __name__ == "__main__":
    process_and_detect()