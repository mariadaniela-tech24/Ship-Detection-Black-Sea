from ultralytics import YOLO
from pathlib import Path
import os

def train_ship_detector():
    yaml_path = Path("data/processed/hrsid_yolo/dataset.yaml").resolve()
    
    if not yaml_path.exists():
        print(f"Error: Could not find {yaml_path}")
        return

    print(f"Loading dataset configuration from: {yaml_path}")
    
    model = YOLO('yolov8n.pt') 
    
    print("Starting YOLOv8 training pipeline on GPU...")
    
    results = model.train(
        data=str(yaml_path),
        epochs=50,          
        imgsz=512,          
        batch=8,            
        name='yolov8_ship_detector_v2',
        project='models',
        device=0,           # '0' tells PyTorch to use your NVIDIA GPU
        patience=15         # Early stopping if the model stops improving
    )
    
    print("\nTraining complete! Your high-recall model is ready.")

if __name__ == "__main__":
    train_ship_detector()