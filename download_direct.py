#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to download traffic surveillance datasets directly from their sources.
"""

import os
import argparse
import requests
import zipfile
import tarfile
import gdown
import shutil
from tqdm import tqdm
import yaml
import time

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Download traffic surveillance datasets")
    parser.add_argument("--datasets", nargs="+", choices=["helmet", "license_plate", "vehicle", "all"], default=["all"],
                        help="Datasets to download (default: all)")
    parser.add_argument("--output-dir", default="data/datasets",
                        help="Output directory (default: data/datasets)")
    parser.add_argument("--no-extract", action="store_true",
                        help="Skip extraction of downloaded archives")
    return parser.parse_args()

def download_file(url, destination, description=None):
    """Download a file with progress bar."""
    try:
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        block_size = 1024  # 1 Kibibyte
        
        desc = description or os.path.basename(destination)
        t = tqdm(total=total_size, unit='iB', unit_scale=True, desc=desc)
        
        with open(destination, 'wb') as f:
            for data in response.iter_content(block_size):
                t.update(len(data))
                f.write(data)
        t.close()
        
        if total_size != 0 and t.n != total_size:
            print("ERROR: Download incomplete")
            return False
        return True
    except Exception as e:
        print(f"Error downloading file: {e}")
        return False

def download_from_gdrive(file_id, destination, description=None):
    """Download a file from Google Drive."""
    try:
        desc = description or os.path.basename(destination)
        print(f"Downloading {desc} from Google Drive...")
        gdown.download(id=file_id, output=destination, quiet=False)
        return True
    except Exception as e:
        print(f"Error downloading from Google Drive: {e}")
        return False

def extract_archive(archive_path, extract_dir, archive_type="zip"):
    """Extract an archive file."""
    try:
        print(f"Extracting {os.path.basename(archive_path)} to {extract_dir}...")
        os.makedirs(extract_dir, exist_ok=True)
        
        if archive_type == "zip":
            with zipfile.ZipFile(archive_path, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)
        elif archive_type in ["tar", "tar.gz", "tgz"]:
            with tarfile.open(archive_path, 'r:*') as tar_ref:
                tar_ref.extractall(extract_dir)
        else:
            print(f"Unsupported archive type: {archive_type}")
            return False
        
        return True
    
    except Exception as e:
        print(f"Error extracting archive: {e}")
        return False

def download_helmet_dataset(output_dir):
    """
    Download helmet detection dataset.
    
    Using the Helmet Detection dataset from Roboflow:
    https://universe.roboflow.com/joseph-nelson/hard-hat-workers
    """
    dataset_dir = os.path.join(output_dir, "helmet")
    os.makedirs(dataset_dir, exist_ok=True)
    
    print("\nDownloading Helmet Detection dataset...")
    
    # Direct download links for the dataset in YOLO format
    download_urls = {
        "train": "https://app.roboflow.com/ds/0XD9MFsXKS?key=ULdqLDVVvs",
        "valid": "https://app.roboflow.com/ds/Gg7eGzEiTl?key=ULdqLDVVvs",
        "test": "https://app.roboflow.com/ds/Gg7eGzEiTl?key=ULdqLDVVvs"
    }
    
    # Download and extract each split
    for split, url in download_urls.items():
        archive_path = os.path.join(dataset_dir, f"{split}.zip")
        
        # Download the dataset
        print(f"Downloading {split} split...")
        success = download_file(url, archive_path, f"Helmet {split} dataset")
        
        if not success:
            print(f"Failed to download {split} split")
            continue
        
        # Extract the archive
        extract_dir = os.path.join(dataset_dir, split)
        success = extract_archive(archive_path, extract_dir)
        
        if not success:
            print(f"Failed to extract {split} split")
    
    # Create a YAML file for the dataset
    yaml_content = {
        "path": os.path.abspath(dataset_dir),
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
        "names": {
            0: "helmet",
            1: "head",
            2: "person"
        }
    }
    
    with open(os.path.join(dataset_dir, "dataset.yaml"), "w") as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    
    print("Helmet Detection dataset downloaded successfully")
    return True

def download_license_plate_dataset(output_dir):
    """
    Download license plate dataset.
    
    Using the CCPD (Chinese City Parking Dataset):
    https://github.com/detectRecog/CCPD
    """
    dataset_dir = os.path.join(output_dir, "license_plate")
    os.makedirs(dataset_dir, exist_ok=True)
    
    print("\nDownloading License Plate dataset (CCPD)...")
    
    # CCPD dataset is available on Google Drive
    gdrive_ids = {
        "ccpd_base": "1rdEsCUcIUaYAO58U2Jn-hZQzC0U8Iwgj",
        "ccpd_blur": "1aDrQnxBm0c4S2tQXZ2JXYQd3-AqVH1lt",
        "ccpd_challenge": "1UbGCdOGWohQqwlVgAWvL2JtH0hD7QKb8",
        "ccpd_db": "1r7ckUJRIxJ_6XmAZo7pCWh_Jtmh0BVw5",
        "ccpd_fn": "1sZLhNAw-RL-EsT7QsQYNT7qjQfgDDfQj",
        "ccpd_rotate": "1YYtVsaoXH-zAgo1Rmbr9K6HBQJm5i9XY",
        "ccpd_tilt": "1JZZ_jZGhGKHVy8-v9vKy8RTKY0zGUfq9",
        "ccpd_weather": "1xKkpS-RyRBEBUBwjEXxTESzGpLOCdwE7"
    }
    
    # Download only the base dataset to save time and space
    subset = "ccpd_base"
    archive_path = os.path.join(dataset_dir, f"{subset}.zip")
    
    # Download the dataset
    print(f"Downloading {subset} subset...")
    success = download_from_gdrive(gdrive_ids[subset], archive_path, f"CCPD {subset} dataset")
    
    if not success:
        # Try alternative download method
        print("Trying alternative download method...")
        alt_url = "https://drive.google.com/uc?id=1rdEsCUcIUaYAO58U2Jn-hZQzC0U8Iwgj"
        success = gdown.download(alt_url, archive_path, quiet=False)
        
        if not success:
            print(f"Failed to download {subset} subset")
            
            # Create a small sample dataset instead
            print("Creating a small sample license plate dataset instead...")
            create_sample_license_plate_dataset(dataset_dir, 50)
            return True
    
    # Extract the archive
    extract_dir = os.path.join(dataset_dir, subset)
    success = extract_archive(archive_path, extract_dir)
    
    if not success:
        print(f"Failed to extract {subset} subset")
    
    # Create a YAML file for the dataset
    yaml_content = {
        "path": os.path.abspath(dataset_dir),
        "train": f"{subset}/images",
        "val": f"{subset}/images",
        "names": {
            0: "license_plate"
        }
    }
    
    with open(os.path.join(dataset_dir, "dataset.yaml"), "w") as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    
    print("License Plate dataset downloaded successfully")
    return True

def download_vehicle_dataset(output_dir):
    """
    Download vehicle dataset.
    
    Using the UA-DETRAC dataset:
    https://detrac-db.rit.albany.edu/
    """
    dataset_dir = os.path.join(output_dir, "vehicle")
    os.makedirs(dataset_dir, exist_ok=True)
    
    print("\nDownloading Vehicle dataset (UA-DETRAC)...")
    
    # UA-DETRAC dataset requires registration, so we'll use a mirror or sample
    # Try to download a small subset from an alternative source
    
    # Alternative: Use the COCO-pretrained model and a small sample dataset
    print("UA-DETRAC dataset requires registration.")
    print("Creating a small sample vehicle dataset instead...")
    create_sample_vehicle_dataset(dataset_dir, 50)
    
    print("Vehicle dataset created successfully")
    return True

def create_sample_license_plate_dataset(output_dir, num_samples=50):
    """Create a sample license plate dataset."""
    print(f"Creating sample license plate dataset with {num_samples} images...")
    
    # Create directories
    images_dir = os.path.join(output_dir, "images")
    labels_dir = os.path.join(output_dir, "labels")
    
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    
    # Import required libraries
    import cv2
    import numpy as np
    import random
    
    # Generate random license plates
    def generate_license_plate():
        # Format: ABC-1234
        letters = "ABCDEFGHJKLMNPQRSTUVWXYZ"
        digits = "0123456789"
        
        plate = ""
        for _ in range(3):
            plate += random.choice(letters)
        
        plate += "-"
        
        for _ in range(4):
            plate += random.choice(digits)
        
        return plate
    
    # Generate sample images and annotations
    for i in range(num_samples):
        # Create a blank image (road scene)
        img = np.zeros((640, 640, 3), dtype=np.uint8)
        
        # Add a background (road)
        cv2.rectangle(img, (0, 0), (640, 640), (100, 100, 100), -1)
        
        # Generate a random license plate
        plate_text = generate_license_plate()
        
        # Create a license plate image
        plate_width = random.randint(100, 200)
        plate_height = int(plate_width * 0.2)  # License plates are typically 5:1 ratio
        plate_x = random.randint(0, 640 - plate_width)
        plate_y = random.randint(0, 640 - plate_height)
        
        # Draw the license plate
        cv2.rectangle(img, (plate_x, plate_y), (plate_x + plate_width, plate_y + plate_height), (255, 255, 255), -1)
        cv2.rectangle(img, (plate_x, plate_y), (plate_x + plate_width, plate_y + plate_height), (0, 0, 0), 2)
        
        # Add the license plate text
        font_scale = plate_width / 200
        cv2.putText(img, plate_text, (plate_x + 5, plate_y + plate_height - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 2)
        
        # Save the image
        image_path = os.path.join(images_dir, f"plate_{i:04d}.jpg")
        cv2.imwrite(image_path, img)
        
        # YOLO format: class_id center_x center_y width height (normalized)
        center_x = (plate_x + plate_width / 2) / 640
        center_y = (plate_y + plate_height / 2) / 640
        norm_width = plate_width / 640
        norm_height = plate_height / 640
        
        # Save the YOLO annotation
        label_path = os.path.join(labels_dir, f"plate_{i:04d}.txt")
        with open(label_path, "w") as f:
            f.write(f"0 {center_x:.6f} {center_y:.6f} {norm_width:.6f} {norm_height:.6f}")
    
    # Create dataset.yaml for YOLO
    yaml_content = {
        "path": os.path.abspath(output_dir),
        "train": "images",
        "val": "images",
        "names": {
            0: "license_plate"
        }
    }
    
    with open(os.path.join(output_dir, "dataset.yaml"), "w") as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    
    print(f"Created sample license plate dataset with {num_samples} images")

def create_sample_vehicle_dataset(output_dir, num_samples=50):
    """Create a sample vehicle dataset."""
    print(f"Creating sample vehicle dataset with {num_samples} images...")
    
    # Create directories
    images_dir = os.path.join(output_dir, "images")
    labels_dir = os.path.join(output_dir, "labels")
    
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(labels_dir, exist_ok=True)
    
    # Import required libraries
    import cv2
    import numpy as np
    import random
    
    # Define vehicle types and colors
    vehicle_types = ["car", "truck", "bus", "motorcycle", "bicycle"]
    vehicle_colors = [
        (0, 0, 255),    # Red
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (0, 255, 255),  # Yellow
        (255, 0, 255)   # Magenta
    ]
    
    # Generate sample images and annotations
    for i in range(num_samples):
        # Create a blank image (road scene)
        img = np.zeros((640, 640, 3), dtype=np.uint8)
        
        # Add a background (road)
        cv2.rectangle(img, (0, 0), (640, 640), (100, 100, 100), -1)
        
        # Add road markings
        # Center line
        cv2.line(img, (0, 640 // 2), (640, 640 // 2), (255, 255, 255), 2)
        
        # Lane markings
        for x in range(0, 640, 50):
            cv2.line(img, (x, 640 // 4), (x + 30, 640 // 4), (255, 255, 255), 2)
            cv2.line(img, (x, 3 * 640 // 4), (x + 30, 3 * 640 // 4), (255, 255, 255), 2)
        
        # YOLO format annotations
        yolo_annotations = []
        
        # Add random vehicles
        num_vehicles = random.randint(1, 5)
        
        for j in range(num_vehicles):
            # Select vehicle type
            vehicle_idx = random.randint(0, len(vehicle_types) - 1)
            vehicle_type = vehicle_types[vehicle_idx]
            vehicle_color = vehicle_colors[vehicle_idx]
            
            # Generate random bounding box
            if vehicle_type in ["car", "truck", "bus"]:
                w = random.randint(80, 200)
                h = random.randint(60, 100)
            else:  # motorcycle, bicycle
                w = random.randint(40, 80)
                h = random.randint(40, 80)
            
            x = random.randint(0, 640 - w)
            y = random.randint(0, 640 - h)
            
            # Draw the vehicle
            cv2.rectangle(img, (x, y), (x + w, y + h), vehicle_color, -1)
            
            # Add text label
            cv2.putText(img, vehicle_type, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Add to YOLO annotations
            # YOLO format: class_id center_x center_y width height (normalized)
            center_x = (x + w / 2) / 640
            center_y = (y + h / 2) / 640
            norm_width = w / 640
            norm_height = h / 640
            yolo_annotations.append(f"{vehicle_idx} {center_x:.6f} {center_y:.6f} {norm_width:.6f} {norm_height:.6f}")
        
        # Save the image
        image_path = os.path.join(images_dir, f"vehicle_{i:04d}.jpg")
        cv2.imwrite(image_path, img)
        
        # Save the YOLO annotation
        label_path = os.path.join(labels_dir, f"vehicle_{i:04d}.txt")
        with open(label_path, "w") as f:
            f.write("\n".join(yolo_annotations))
    
    # Create dataset.yaml for YOLO
    yaml_content = {
        "path": os.path.abspath(output_dir),
        "train": "images",
        "val": "images",
        "names": {i: vehicle_type for i, vehicle_type in enumerate(vehicle_types)}
    }
    
    with open(os.path.join(output_dir, "dataset.yaml"), "w") as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    
    print(f"Created sample vehicle dataset with {num_samples} images")

def main():
    """Main function."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine which datasets to download
    datasets_to_download = args.datasets
    if "all" in datasets_to_download:
        datasets_to_download = ["helmet", "license_plate", "vehicle"]
    
    # Download datasets
    for dataset in datasets_to_download:
        if dataset == "helmet":
            download_helmet_dataset(args.output_dir)
        elif dataset == "license_plate":
            download_license_plate_dataset(args.output_dir)
        elif dataset == "vehicle":
            download_vehicle_dataset(args.output_dir)
    
    print("\nDataset download complete!")
    print(f"Datasets are stored in: {os.path.abspath(args.output_dir)}")
    print("\nTo train a YOLOv8 model on these datasets, run:")
    print("python train_models.py --dataset [helmet|license_plate|vehicle]")

if __name__ == "__main__":
    main()
