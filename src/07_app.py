import os
import glob
import random
import time

import cv2
import gradio as gr
from ultralytics import YOLO

# ---------------------------------------------------------------------------
# Model + sample locations (same auto-discovery approach as 06_demo.py)
# ---------------------------------------------------------------------------
MODEL_CANDIDATES = [
    "runs/detect/models/yolov8_ship_detector_v2/weights/best.pt",
    "runs/detect/models/yolov8_ship_detector_v1/weights/best.pt",
    "runs/detect/train/weights/best.pt",
]

SAMPLE_SEARCH_DIRS = [
    "data/processed/sentinel1_tiles",        # real Constanta detections (05_detect_constanta.py)
    "data/processed/hrsid_yolo/images/val",  # held-out HRSID validation tiles
    "data/processed/hrsid_yolo/images/train",
]

CONF_THRESHOLD = 0.25


def find_model_path():
    for path in MODEL_CANDIDATES:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        "No trained weights found. Checked:\n  " + "\n  ".join(MODEL_CANDIDATES)
    )


def collect_sample_images(max_samples=12):
    samples = []
    for directory in SAMPLE_SEARCH_DIRS:
        samples.extend(glob.glob(os.path.join(directory, "*.jpg")))
        samples.extend(glob.glob(os.path.join(directory, "*.png")))
    random.shuffle(samples)
    return samples[:max_samples]


print("[INFO] Loading model...")
MODEL_PATH = find_model_path()
model = YOLO(MODEL_PATH)
print(f"[INFO] Loaded: {MODEL_PATH}")

SAMPLE_IMAGES = collect_sample_images()
print(f"[INFO] Found {len(SAMPLE_IMAGES)} sample tiles for the gallery/random button.")


def run_detection(image):
    """image: RGB numpy array from Gradio's image input."""
    if image is None:
        return None, "Upload a tile or click **Try a Random Tile** first."

    start = time.time()
    results = model.predict(source=image, conf=CONF_THRESHOLD, verbose=False)
    elapsed_ms = (time.time() - start) * 1000

    r = results[0]
    annotated_rgb = cv2.cvtColor(r.plot(), cv2.COLOR_BGR2RGB)  # plot() returns BGR

    n = len(r.boxes)
    confs = [f"{c:.2f}" for c in r.boxes.conf.tolist()]
    summary = f"**{n} ship(s) detected** — {elapsed_ms:.0f} ms"
    if confs:
        summary += f"\n\nConfidences: {', '.join(confs)}"

    return annotated_rgb, summary


def load_random_sample():
    if not SAMPLE_IMAGES:
        return None
    path = random.choice(SAMPLE_IMAGES)
    return cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)


with gr.Blocks(title="OmniSea Intelligence - Black Sea Ship Detection") as demo:
    gr.Markdown(
        "# OmniSea Intelligence\n\n"
        "Upload a radar tile, or click **Try a Random Tile** to test on a real sample."
    )

    with gr.Row():
        with gr.Column():
            input_image = gr.Image(label="Input tile", type="numpy")
            with gr.Row():
                random_btn = gr.Button("🎲 Try a Random Tile", variant="secondary")
                detect_btn = gr.Button("🔍 Run Detection", variant="primary")
        with gr.Column():
            output_image = gr.Image(label="Detections")
            output_text = gr.Markdown()

    if SAMPLE_IMAGES:
        gr.Examples(examples=SAMPLE_IMAGES, inputs=input_image, label="Or click a sample tile")

    random_btn.click(fn=load_random_sample, outputs=input_image).then(
        fn=run_detection, inputs=input_image, outputs=[output_image, output_text]
    )
    detect_btn.click(fn=run_detection, inputs=input_image, outputs=[output_image, output_text])


if __name__ == "__main__":
    demo.launch(inbrowser=True)