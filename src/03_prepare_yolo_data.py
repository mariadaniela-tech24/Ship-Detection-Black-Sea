import os
import json
import shutil
from pathlib import Path

def convert_coco_to_yolo():
    raw_dir = Path("data/raw/hrsid")
    processed_dir = Path("data/processed/hrsid_yolo")
    
    ann_dir = raw_dir / "annotations"
    img_dir = raw_dir / "images"

    # Define splits to convert
    splits = {
        "train": ann_dir / "train2017.json",
        "val": ann_dir / "test2017.json"
    }

    print("Starting COCO to YOLO conversion...")

    for split_name, json_path in splits.items():
        if not json_path.exists():
            print(f"Skipping {split_name}: {json_path} not found.")
            continue

        print(f"Processing {split_name} split...")

        # Create output directories for images and labels
        split_img_dir = processed_dir / "images" / split_name
        split_lbl_dir = processed_dir / "labels" / split_name
        split_img_dir.mkdir(parents=True, exist_ok=True)
        split_lbl_dir.mkdir(parents=True, exist_ok=True)

        with open(json_path, "r") as f:
            coco_data = json.load(f)

        # Map image ID to metadata
        images = {img["id"]: img for img in coco_data["images"]}
        
        # Map image ID to annotations
        img_to_anns = {}
        for ann in coco_data["annotations"]:
            img_id = ann["image_id"]
            img_to_anns.setdefault(img_id, []).append(ann)

        # Convert annotations to YOLO format
        for img_id, img_info in images.items():
            file_name = img_info["file_name"]
            img_w = img_info["width"]
            img_h = img_info["height"]

            # Source image path
            src_img_path = img_dir / file_name
            if not src_img_path.exists():
                continue

            # Copy image to processed directory
            dst_img_path = split_img_dir / file_name
            shutil.copy(src_img_path, dst_img_path)

            # Write YOLO label file
            label_file_path = split_lbl_dir / f"{Path(file_name).stem}.txt"
            anns = img_to_anns.get(img_id, [])

            with open(label_file_path, "w") as label_file:
                for ann in anns:
                    # COCO bbox: [x_min, y_min, width, height]
                    bbox = ann["bbox"]
                    x_min, y_min, bw, bh = bbox[0], bbox[1], bbox[2], bbox[3]

                    # Calculate normalized YOLO format
                    x_center = (x_min + bw / 2.0) / img_w
                    y_center = (y_min + bh / 2.0) / img_h
                    norm_bw = bw / img_w
                    norm_bh = bh / img_h

                    # Class 0 for ship
                    label_file.write(f"0 {x_center:.6f} {y_center:.6f} {norm_bw:.6f} {norm_bh:.6f}\n")

    # Create dataset.yaml file required by YOLO
    yaml_content = f"""path: {processed_dir.resolve()}
train: images/train
val: images/val

names:
  0: ship
"""
    yaml_path = processed_dir / "dataset.yaml"
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print(f"Dataset conversion complete! YAML configuration created at: {yaml_path}")

if __name__ == "__main__":
    convert_coco_to_yolo()