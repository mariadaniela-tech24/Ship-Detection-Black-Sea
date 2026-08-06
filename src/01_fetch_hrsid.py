import os
import subprocess

os.environ['KAGGLE_API_TOKEN'] = "KGAT_81e5a3b84b4df0c39a6587934881e292"

def download_hrsid():
    output_dir = "data/raw/hrsid"
    print(f"Downloading HRSID dataset to {output_dir}...")

    subprocess.run([
        "kaggle", "datasets", "download", 
        "-d", "sarribere99/high-resolution-sar-images-dataset-hrsid", 
        "-p", output_dir, 
        "--unzip"
    ])
    print("Download and extraction complete!")

if __name__ == "__main__":
    download_hrsid()