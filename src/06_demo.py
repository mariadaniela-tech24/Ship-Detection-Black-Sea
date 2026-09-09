import os
import glob
import cv2
from ultralytics import YOLO

# --- Candidate model paths (checked in order) ---
MODEL_CANDIDATES = [
    "runs/detect/models/yolov8_ship_detector_v2/weights/best.pt",
    "runs/detect/models/yolov8_ship_detector_v1/weights/best.pt",
    "runs/detect/train/weights/best.pt",
]

# --- Fallback locations to search for a sample image if none is set below ---
IMAGE_SEARCH_DIRS = [
    "data/processed/hrsid_yolo/images/val",
    "data/processed/hrsid_yolo/images/train",
    "data/processed/sentinel1_tiles",
]

# Set this to a specific image if you want to force one; leave as None to auto-pick.
IMAGE_PATH = None

OUTPUT_PATH = "demo_output.jpg"


def find_model_path():
    for path in MODEL_CANDIDATES:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        "No trained weights found. Checked:\n  "
        + "\n  ".join(MODEL_CANDIDATES)
        + "\nDid you run 04_train_yolo.py, and does the run folder name match?"
    )


def find_image_path():
    if IMAGE_PATH and os.path.exists(IMAGE_PATH):
        return IMAGE_PATH
    if IMAGE_PATH:
        print(f"[WARN] IMAGE_PATH '{IMAGE_PATH}' not found, searching for a fallback...")

    for directory in IMAGE_SEARCH_DIRS:
        matches = glob.glob(os.path.join(directory, "*.jpg")) + glob.glob(os.path.join(directory, "*.png"))
        if matches:
            return matches[0]

    raise FileNotFoundError(
        "No sample image found. Checked:\n  " + "\n  ".join(IMAGE_SEARCH_DIRS)
        + "\nSet IMAGE_PATH manually to a real file, or run 03_prepare_yolo_data.py first."
    )


def run_demo():
    print("\n[INFO] Initializing AegisSAR Engine...")
    model_path = find_model_path()
    print(f"[INFO] Using model: {model_path}")
    model = YOLO(model_path)

    image_path = find_image_path()
    print(f"[INFO] Scanning radar tile: {image_path}")
    results = model(image_path)

    for r in results:
        annotated_frame = r.plot()
        n_detections = len(r.boxes)

        cv2.imwrite(OUTPUT_PATH, annotated_frame)
        print(f"[SUCCESS] {n_detections} target(s) found. Saved annotated image to: {OUTPUT_PATH}")

        # Try to open a live window; skip gracefully if no display is available
        # (this fails by design in Colab / headless / SSH sessions).
        try:
            window_name = "AegisSAR Live Demo - Press ANY KEY to close"
            cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(window_name, 800, 800)
            cv2.imshow(window_name, annotated_frame)
            print("[INFO] Press any key in the image window to close.")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except cv2.error as e:
            print(f"[INFO] No display available to show a live window ({e}). "
                  f"Open '{OUTPUT_PATH}' directly to view the result.")


if __name__ == "__main__":
    run_demo()