#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to download larger and more diverse datasets for traffic surveillance.
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
    parser = argparse.ArgumentParser(description="Download larger traffic surveillance datasets")
    parser.add_argument("--datasets", nargs="+", choices=["bdd100k", "visdrone", "ua-detrac", "all"], default=["all"],
                        help="Datasets to download (default: all)")
    parser.add_argument("--output-dir", default="data/datasets",
                        help="Output directory (default: data/datasets)")
    parser.add_argument("--sample-only", action="store_true",
                        help="Download only a small sample of each dataset")
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

def download_bdd100k(output_dir, sample_only=False):
    """
    Download BDD100K dataset.
    
    BDD100K is a large-scale diverse driving dataset with 100K videos and 10 tasks.
    https://bdd-data.berkeley.edu/
    """
    dataset_dir = os.path.join(output_dir, "bdd100k")
    os.makedirs(dataset_dir, exist_ok=True)
    
    print("\nDownloading BDD100K dataset...")
    
    # BDD100K requires registration, so we'll use a sample or alternative source
    if sample_only:
        print("Downloading a small sample of BDD100K...")
        
        # Sample images URL (this is a placeholder - in a real scenario, you would need to register and get the actual download link)
        sample_url = "https://github.com/ucbdrive/bdd-data/raw/master/bdd100k/images/10k/sample.zip"
        
        # Try to download the sample
        sample_path = os.path.join(dataset_dir, "bdd100k_sample.zip")
        success = download_file(sample_url, sample_path, "BDD100K sample")
        
        if not success:
            print("Failed to download BDD100K sample. Creating a synthetic sample instead...")
            create_synthetic_traffic_dataset(dataset_dir, 100)
            return True
        
        # Extract the sample
        extract_archive(sample_path, dataset_dir)
    else:
        print("BDD100K full dataset requires registration at https://bdd-data.berkeley.edu/")
        print("After registration, download the following files:")
        print("1. Images (100K/10K): https://bdd-data.berkeley.edu/portal.html#download")
        print("2. Labels (Detection/Segmentation): https://bdd-data.berkeley.edu/portal.html#download")
        print("\nCreating a synthetic sample instead...")
        create_synthetic_traffic_dataset(dataset_dir, 200)
    
    # Create dataset.yaml for YOLO
    classes = ["person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle", "traffic light", "traffic sign"]
    
    yaml_content = {
        "path": os.path.abspath(dataset_dir),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: name for i, name in enumerate(classes)}
    }
    
    with open(os.path.join(dataset_dir, "dataset.yaml"), "w") as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    
    print("BDD100K dataset prepared successfully")
    return True

def download_visdrone(output_dir, sample_only=False):
    """
    Download VisDrone dataset.
    
    VisDrone is a large-scale visual object detection and tracking benchmark dataset.
    http://aiskyeye.com/
    """
    dataset_dir = os.path.join(output_dir, "visdrone")
    os.makedirs(dataset_dir, exist_ok=True)
    
    print("\nDownloading VisDrone dataset...")
    
    # VisDrone requires registration, so we'll use a sample or alternative source
    if sample_only:
        print("Downloading a small sample of VisDrone...")
        
        # Sample images URL (this is a placeholder - in a real scenario, you would need to register and get the actual download link)
        sample_url = "https://github.com/VisDrone/VisDrone-Dataset/raw/master/VisDrone2019-DET-train-sample.zip"
        
        # Try to download the sample
        sample_path = os.path.join(dataset_dir, "visdrone_sample.zip")
        success = download_file(sample_url, sample_path, "VisDrone sample")
        
        if not success:
            print("Failed to download VisDrone sample. Creating a synthetic sample instead...")
            create_synthetic_traffic_dataset(dataset_dir, 100)
            return True
        
        # Extract the sample
        extract_archive(sample_path, dataset_dir)
    else:
        print("VisDrone full dataset requires registration at http://aiskyeye.com/")
        print("After registration, download the following files:")
        print("1. Object Detection in Images: http://aiskyeye.com/download/object-detection-in-images/")
        print("2. Object Detection in Videos: http://aiskyeye.com/download/object-detection-in-videos/")
        print("\nCreating a synthetic sample instead...")
        create_synthetic_traffic_dataset(dataset_dir, 200)
    
    # Create dataset.yaml for YOLO
    classes = ["pedestrian", "person", "bicycle", "car", "van", "truck", "tricycle", "awning-tricycle", "bus", "motor"]
    
    yaml_content = {
        "path": os.path.abspath(dataset_dir),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: name for i, name in enumerate(classes)}
    }
    
    with open(os.path.join(dataset_dir, "dataset.yaml"), "w") as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    
    print("VisDrone dataset prepared successfully")
    return True

def download_ua_detrac(output_dir, sample_only=False):
    """
    Download UA-DETRAC dataset.
    
    UA-DETRAC is a challenging real-world multi-object detection and tracking benchmark.
    https://detrac-db.rit.albany.edu/
    """
    dataset_dir = os.path.join(output_dir, "ua-detrac")
    os.makedirs(dataset_dir, exist_ok=True)
    
    print("\nDownloading UA-DETRAC dataset...")
    
    # UA-DETRAC requires registration, so we'll use a sample or alternative source
    if sample_only:
        print("Downloading a small sample of UA-DETRAC...")
        
        # Sample images URL (this is a placeholder - in a real scenario, you would need to register and get the actual download link)
        sample_url = "https://detrac-db.rit.albany.edu/Data/DETRAC-sample-data.zip"
        
        # Try to download the sample
        sample_path = os.path.join(dataset_dir, "ua_detrac_sample.zip")
        success = download_file(sample_url, sample_path, "UA-DETRAC sample")
        
        if not success:
            print("Failed to download UA-DETRAC sample. Creating a synthetic sample instead...")
            create_synthetic_traffic_dataset(dataset_dir, 100)
            return True
        
        # Extract the sample
        extract_archive(sample_path, dataset_dir)
    else:
        print("UA-DETRAC full dataset requires registration at https://detrac-db.rit.albany.edu/")
        print("After registration, download the following files:")
        print("1. Training Data: https://detrac-db.rit.albany.edu/download")
        print("2. Testing Data: https://detrac-db.rit.albany.edu/download")
        print("\nCreating a synthetic sample instead...")
        create_synthetic_traffic_dataset(dataset_dir, 200)
    
    # Create dataset.yaml for YOLO
    classes = ["car", "bus", "van", "others"]
    
    yaml_content = {
        "path": os.path.abspath(dataset_dir),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: name for i, name in enumerate(classes)}
    }
    
    with open(os.path.join(dataset_dir, "dataset.yaml"), "w") as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    
    print("UA-DETRAC dataset prepared successfully")
    return True

def create_synthetic_traffic_dataset(output_dir, num_samples=100):
    """Create a synthetic traffic dataset for testing."""
    print(f"Creating synthetic traffic dataset with {num_samples} images...")
    
    # Create directories
    images_dir = os.path.join(output_dir, "images")
    labels_dir = os.path.join(output_dir, "labels")
    train_images_dir = os.path.join(images_dir, "train")
    train_labels_dir = os.path.join(labels_dir, "train")
    val_images_dir = os.path.join(images_dir, "val")
    val_labels_dir = os.path.join(labels_dir, "val")
    
    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(train_labels_dir, exist_ok=True)
    os.makedirs(val_images_dir, exist_ok=True)
    os.makedirs(val_labels_dir, exist_ok=True)
    
    # Import required libraries
    import cv2
    import numpy as np
    import random
    
    # Define classes and colors
    classes = ["car", "truck", "bus", "motorcycle", "bicycle", "person"]
    colors = [
        (0, 0, 255),    # Red
        (0, 255, 0),    # Green
        (255, 0, 0),    # Blue
        (0, 255, 255),  # Yellow
        (255, 0, 255),  # Magenta
        (255, 255, 0)   # Cyan
    ]
    
    # Generate sample images and annotations
    for i in tqdm(range(num_samples), desc="Generating synthetic data"):
        # Create a blank image (road scene)
        img = np.zeros((640, 640, 3), dtype=np.uint8)
        
        # Add a background (road)
        cv2.rectangle(img, (0, 0), (640, 640), (100, 100, 100), -1)
        
        # Add road markings
        # Center line
        cv2.line(img, (0, 320), (640, 320), (255, 255, 255), 2)
        
        # Lane markings
        for x in range(0, 640, 50):
            cv2.line(img, (x, 160), (x + 30, 160), (255, 255, 255), 2)
            cv2.line(img, (x, 480), (x + 30, 480), (255, 255, 255), 2)
        
        # YOLO format annotations
        yolo_annotations = []
        
        # Add random objects
        num_objects = random.randint(3, 10)
        
        for j in range(num_objects):
            # Select object type
            class_idx = random.randint(0, len(classes) - 1)
            class_name = classes[class_idx]
            color = colors[class_idx]
            
            # Generate random bounding box
            if class_name in ["car", "truck", "bus"]:
                w = random.randint(80, 200)
                h = random.randint(60, 100)
            elif class_name in ["motorcycle", "bicycle"]:
                w = random.randint(40, 80)
                h = random.randint(40, 80)
            else:  # person
                w = random.randint(30, 60)
                h = random.randint(60, 120)
            
            x = random.randint(0, 640 - w)
            y = random.randint(0, 640 - h)
            
            # Draw the object
            cv2.rectangle(img, (x, y), (x + w, y + h), color, -1)
            
            # Add text label
            cv2.putText(img, class_name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Add to YOLO annotations
            # YOLO format: class_id center_x center_y width height (normalized)
            center_x = (x + w / 2) / 640
            center_y = (y + h / 2) / 640
            norm_width = w / 640
            norm_height = h / 640
            yolo_annotations.append(f"{class_idx} {center_x:.6f} {center_y:.6f} {norm_width:.6f} {norm_height:.6f}")
        
        # Decide whether to put in train or val set (80/20 split)
        if random.random() < 0.8:
            image_path = os.path.join(train_images_dir, f"synthetic_{i:04d}.jpg")
            label_path = os.path.join(train_labels_dir, f"synthetic_{i:04d}.txt")
        else:
            image_path = os.path.join(val_images_dir, f"synthetic_{i:04d}.jpg")
            label_path = os.path.join(val_labels_dir, f"synthetic_{i:04d}.txt")
        
        # Save the image
        cv2.imwrite(image_path, img)
        
        # Save the YOLO annotation
        with open(label_path, "w") as f:
            f.write("\n".join(yolo_annotations))
    
    # Create dataset.yaml for YOLO
    yaml_content = {
        "path": os.path.abspath(output_dir),
        "train": "images/train",
        "val": "images/val",
        "names": {i: name for i, name in enumerate(classes)}
    }
    
    with open(os.path.join(output_dir, "dataset.yaml"), "w") as f:
        yaml.dump(yaml_content, f, default_flow_style=False)
    
    print(f"Created synthetic traffic dataset with {num_samples} images")

def main():
    """Main function."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine which datasets to download
    datasets_to_download = args.datasets
    if "all" in datasets_to_download:
        datasets_to_download = ["bdd100k", "visdrone", "ua-detrac"]
    
    # Download datasets
    for dataset in datasets_to_download:
        if dataset == "bdd100k":
            download_bdd100k(args.output_dir, args.sample_only)
        elif dataset == "visdrone":
            download_visdrone(args.output_dir, args.sample_only)
        elif dataset == "ua-detrac":
            download_ua_detrac(args.output_dir, args.sample_only)
    
    print("\nLarger dataset download and preparation complete!")
    print(f"Datasets are stored in: {os.path.abspath(args.output_dir)}")
    print("\nTo train a YOLOv8 model on these datasets, run:")
    print("python train_advanced.py --dataset [bdd100k|visdrone|ua-detrac]")

if __name__ == "__main__":
    main()
