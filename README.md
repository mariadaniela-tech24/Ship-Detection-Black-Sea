# 🚢 Sentinel-1 SAR Ship Detection Pipeline



## 🛰️ Project Overview

This repository presents a complete, end-to-end machine learning pipeline designed for **Earth Observation and Maritime Surveillance**. It automatically detects and maps maritime vessels across massive **Sentinel-1 Synthetic Aperture Radar (SAR)** satellite imagery.

**The Challenge:** Standard object detection models are designed for small, everyday photographs. When fed a raw, 428-megapixel (25,000 x 16,000 pixel) satellite map, standard algorithms aggressively downscale the input. When that happens, massive 300-meter cargo ships vanish into sub-pixel noise, resulting in zero detections.

**The Solution:** This project bridges the gap between deep learning and geospatial analysis by implementing a **Sliding-Window Inference Engine** paired with a custom-trained **YOLOv8** model. 

### 🔑 Key Capabilities:
* **Advanced SAR Processing:** Safely reads heavily compressed, complex 16-bit Sentinel-1 measurement TIFFs, dynamically normalizes radar backscatter percentiles, and converts them to 8-bit matrices without losing the physical signatures of the ships.
* **Gigapixel Inference:** Dynamically slices massive satellite maps into 1024x1024 high-resolution tiles. The engine acts as a scanner, running inference purely on valid radar data while ignoring empty landmasses.
* **High-Recall Model:** Powered by a YOLOv8 network trained from scratch on the **HRSID** (High-Resolution SAR Images Dataset) using GPU-accelerated PyTorch (CUDA). 
* **Proven Metrics:** Achieved **89.7% Precision** and **79.3% Recall** during training. In live inference on the Constanța port region, the sliding-window pipeline successfully mapped 128 individual vessels in a single pass.
### ✨ Detection Showcase
Here are real examples of the model successfully identifying vessels within complex inland waterways using the SAR backscatter signatures:

<p align="center">
  <img width="400" height="400" alt="Screenshot 2026-09-03 152320" src="https://github.com/user-attachments/assets/8f919f35-4ed8-4ab5-a143-ddbbb0ab0f86" />
<img width="400" height="400" alt="Screenshot 2026-09-03 152419" src="https://github.com/user-attachments/assets/fa450716-9e79-4ead-8edb-37fec038d1c0" />
</p>


---

## ⚙️ Installation Requirements

**1. Clone the repository**

    git clone https://github.com/YOUR_USERNAME/sentinel1-ship-detection.git
    cd sentinel1-ship-detection

**2. Setup the Environment (Windows/CUDA)**

    python -m venv venv
    .\venv\Scripts\activate
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
    pip install ultralytics tifffile imagecodecs opencv-python

---

## 🚀 Running the Pipeline

*   **Prepare the Data:** Place your Sentinel-1 raw `.zip` files into `data/raw/sentinel1/`.
*   **Train the Model:** Execute `python src/04_train_yolo.py` to initiate GPU-accelerated training over 50 epochs (512px resolution).
*   **Execute the Scanner:** Run `python src/05_detect_constanta.py` to deploy the sliding-window scanner across the entire SAR image.
*   **Run a Live Demo:** Execute `python src/06_demo.py` to test the model on a single radar tile and visualize the generated bounding boxes.

---
  > **ROSPIN Summer School Project**  
> *This project was developed as part of the ROSPIN Summer School. For more information and other projects, visit the [ROSPIN GitHub organization](https://github.com/Romanian-Space-Initiative).*


