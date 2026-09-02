from ultralytics import YOLO
from pathlib import Path
import os

def train_ship_detector():
    # 1. Locate the YAML configuration file we just created
    yaml_path = Path("data/processed/hrsid_yolo/dataset.yaml").resolve()
    
    if not yaml_path.exists():
        print(f"Error: Could not find {yaml_path}")
        return

    print(f"Loading dataset configuration from: {yaml_path}")
    
    # 2. Load a pre-trained YOLOv8 Nano model
    # (It will automatically download the ~6MB yolov8n.pt weights file the first time)
    model = YOLO('yolov8n.pt') 
    
    # 3. Train the model
    print("Starting YOLOv8 training pipeline...")
    
    # NOTE: We are using very small numbers here for a "test run" so it doesn't take hours.
    # Once we confirm it works, we can increase epochs and imgsz for better accuracy.
    results = model.train(
        data=str(yaml_path),
        epochs=3,           # Only 3 passes over the data for a quick test
        imgsz=256,          # Smaller image size to prevent out-of-memory errors
        batch=4,            # Process 4 images at a time
        name='yolov8_ship_detector_v1',
        project='models',   # Saves outputs to a 'models/' folder in your project root
        device='cpu'        # Explicitly setting to CPU for local testing (change to 0 if you have an Nvidia GPU setup)
    )
    
    print("\nTraining complete! Check the 'models/yolov8_ship_detector_v1' folder for your results and graphs.")

if __name__ == "__main__":
    train_ship_detector()