#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to download and prepare datasets for traffic surveillance.
Also sets up DVC for dataset versioning.
"""

import os
import sys
import argparse
import subprocess
import zipfile
import tarfile
import shutil
import requests
import gdown
from tqdm import tqdm
import yaml

# Dataset URLs
DATASETS = {
    "helmet": {
        "name": "Helmet Detection Dataset",
        "url": "https://www.kaggle.com/datasets/andrewmvd/helmet-detection/download",
        "kaggle_dataset": "andrewmvd/helmet-detection",
        "gdrive_id": "1PQFNqgpKDxTNDKXIzaAYQrbjWOCDPVXS",  # Alternative Google Drive ID
        "type": "zip",
        "size_mb": 57,
        "description": "Dataset with annotations for helmet detection on motorcycle riders"
    },
    "license_plate": {
        "name": "CCPD (Chinese City Parking Dataset)",
        "url": "https://github.com/detectRecog/CCPD",
        "kaggle_dataset": "tolgadincer/ccpd",
        "gdrive_id": "1rdEsCUcIUaYAO58U2Jn-hZQzC0U8Iwgj",  # Alternative Google Drive ID
        "type": "zip",
        "size_mb": 4200,
        "description": "Large dataset with annotated license plates"
    },
    "vehicle": {
        "name": "UA-DETRAC (Urban Traffic Dataset)",
        "url": "https://detrac-db.rit.albany.edu/download",
        "kaggle_dataset": "solesensei/solesensei_bstld",  # Alternative Kaggle dataset
        "gdrive_id": "1AiNIgVQlZYLBQjkpvZhDrNDaUMbIZ1Xn",  # Alternative Google Drive ID
        "type": "zip",
        "size_mb": 1800,
        "description": "Dataset for vehicle detection and tracking with annotations"
    }
}

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Download and prepare datasets for traffic surveillance")
    parser.add_argument("--datasets", nargs="+", choices=list(DATASETS.keys()) + ["all"], default=["all"],
                        help="Datasets to download (default: all)")
    parser.add_argument("--method", choices=["gdrive", "kaggle", "direct"], default="gdrive",
                        help="Download method (default: gdrive)")
    parser.add_argument("--setup-dvc", action="store_true", help="Set up DVC for dataset versioning")
    parser.add_argument("--dvc-remote", default="gdrive", help="DVC remote storage type (default: gdrive)")
    parser.add_argument("--no-convert", action="store_true", help="Skip YOLO format conversion")
    parser.add_argument("--output-dir", default="data/datasets", help="Output directory (default: data/datasets)")
    return parser.parse_args()

def download_file(url, destination, description=None):
    """Download a file with progress bar."""
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

def download_from_kaggle(dataset, destination_dir, description=None):
    """Download a dataset from Kaggle."""
    try:
        desc = description or dataset
        print(f"Downloading {desc} from Kaggle...")
        
        # Check if kaggle CLI is available
        try:
            subprocess.run(["kaggle", "--version"], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except (subprocess.SubprocessError, FileNotFoundError):
            print("Kaggle CLI not found. Please install it with 'pip install kaggle' and set up your API credentials.")
            return False
        
        # Download the dataset
        os.makedirs(destination_dir, exist_ok=True)
        subprocess.run(["kaggle", "datasets", "download", "-d", dataset, "-p", destination_dir, "--unzip"], check=True)
        return True
    
    except Exception as e:
        print(f"Error downloading from Kaggle: {e}")
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

def setup_dvc(datasets_dir, remote_type="gdrive"):
    """Set up DVC for dataset versioning."""
    try:
        print("Setting up DVC...")
        
        # Initialize DVC if not already initialized
        if not os.path.exists(".dvc"):
            subprocess.run(["dvc", "init"], check=True)
            print("DVC initialized")
        
        # Add datasets to DVC
        for dataset_type in os.listdir(datasets_dir):
            dataset_path = os.path.join(datasets_dir, dataset_type)
            if os.path.isdir(dataset_path):
                dvc_file = f"{dataset_path}.dvc"
                if not os.path.exists(dvc_file):
                    subprocess.run(["dvc", "add", dataset_path], check=True)
                    print(f"Added {dataset_path} to DVC")
        
        # Set up remote storage
        if remote_type == "gdrive":
            # Create a unique folder name for this project
            gdrive_folder = "traffic_surveillance_datasets"
            
            # Add Google Drive remote
            subprocess.run(["dvc", "remote", "add", "--default", "gdrive", f"gdrive://{gdrive_folder}"], check=True)
            subprocess.run(["dvc", "remote", "modify", "gdrive", "gdrive_acknowledge_abuse", "true"], check=True)
            print(f"Added Google Drive remote storage: {gdrive_folder}")
            
            # Provide instructions for pushing data
            print("\nTo push your datasets to Google Drive, run:")
            print("dvc push")
            print("\nThis will open a browser window for Google authentication.")
        
        elif remote_type == "local":
            # Add local remote
            remote_dir = os.path.abspath("dvc_remote")
            os.makedirs(remote_dir, exist_ok=True)
            subprocess.run(["dvc", "remote", "add", "--default", "local-remote", f"local://{remote_dir}"], check=True)
            print(f"Added local remote storage: {remote_dir}")
            
            # Provide instructions for pushing data
            print("\nTo push your datasets to the local remote, run:")
            print("dvc push")
        
        else:
            print(f"Unsupported remote type: {remote_type}")
            return False
        
        # Create .gitignore for the datasets directory
        with open(os.path.join(datasets_dir, ".gitignore"), "w") as f:
            f.write("*\n!.gitignore\n")
        
        print("\nDVC setup complete!")
        return True
    
    except Exception as e:
        print(f"Error setting up DVC: {e}")
        return False

def convert_to_yolo_format(dataset_dir, dataset_type):
    """Convert dataset annotations to YOLO format."""
    try:
        print(f"Converting {dataset_type} dataset to YOLO format...")
        
        # Create output directory for YOLO format
        yolo_dir = os.path.join(dataset_dir, "yolo")
        os.makedirs(os.path.join(yolo_dir, "images"), exist_ok=True)
        os.makedirs(os.path.join(yolo_dir, "labels"), exist_ok=True)
        
        # Implement conversion logic based on dataset type
        if dataset_type == "helmet":
            # Helmet dataset conversion (VOC XML to YOLO)
            from xml.etree import ElementTree as ET
            
            # Class mapping
            classes = {"helmet": 0, "head": 1, "person": 2}
            
            # Process annotations
            annotations_dir = os.path.join(dataset_dir, "annotations")
            images_dir = os.path.join(dataset_dir, "images")
            
            if os.path.exists(annotations_dir) and os.path.exists(images_dir):
                for xml_file in os.listdir(annotations_dir):
                    if not xml_file.endswith('.xml'):
                        continue
                    
                    # Parse XML
                    xml_path = os.path.join(annotations_dir, xml_file)
                    tree = ET.parse(xml_path)
                    root = tree.getroot()
                    
                    # Get image dimensions
                    size = root.find('size')
                    width = int(size.find('width').text)
                    height = int(size.find('height').text)
                    
                    # Create YOLO label file
                    image_id = os.path.splitext(xml_file)[0]
                    label_path = os.path.join(yolo_dir, "labels", f"{image_id}.txt")
                    
                    with open(label_path, 'w') as f:
                        for obj in root.findall('object'):
                            class_name = obj.find('name').text
                            if class_name in classes:
                                class_id = classes[class_name]
                                
                                # Get bounding box coordinates
                                bbox = obj.find('bndbox')
                                xmin = float(bbox.find('xmin').text)
                                ymin = float(bbox.find('ymin').text)
                                xmax = float(bbox.find('xmax').text)
                                ymax = float(bbox.find('ymax').text)
                                
                                # Convert to YOLO format (center_x, center_y, width, height)
                                x_center = (xmin + xmax) / 2.0 / width
                                y_center = (ymin + ymax) / 2.0 / height
                                bbox_width = (xmax - xmin) / width
                                bbox_height = (ymax - ymin) / height
                                
                                # Write to file
                                f.write(f"{class_id} {x_center} {y_center} {bbox_width} {bbox_height}\n")
                    
                    # Copy image to YOLO directory
                    image_path = os.path.join(images_dir, f"{image_id}.jpg")
                    if os.path.exists(image_path):
                        shutil.copy(image_path, os.path.join(yolo_dir, "images", f"{image_id}.jpg"))
                
                # Create dataset.yaml
                dataset_yaml = {
                    "path": os.path.abspath(yolo_dir),
                    "train": "images",
                    "val": "images",
                    "names": {v: k for k, v in classes.items()}
                }
                
                with open(os.path.join(yolo_dir, "dataset.yaml"), 'w') as f:
                    yaml.dump(dataset_yaml, f, default_flow_style=False)
                
                print(f"Converted {dataset_type} dataset to YOLO format")
                return True
            else:
                print(f"Required directories not found for {dataset_type} dataset")
                return False
        
        elif dataset_type == "license_plate":
            # CCPD dataset conversion
            # This is a placeholder - actual implementation would depend on the CCPD format
            print("CCPD dataset conversion not implemented yet")
            return False
        
        elif dataset_type == "vehicle":
            # UA-DETRAC dataset conversion
            # This is a placeholder - actual implementation would depend on the UA-DETRAC format
            print("UA-DETRAC dataset conversion not implemented yet")
            return False
        
        else:
            print(f"Unsupported dataset type: {dataset_type}")
            return False
    
    except Exception as e:
        print(f"Error converting to YOLO format: {e}")
        return False

def download_dataset(dataset_type, method, output_dir):
    """Download and prepare a dataset."""
    if dataset_type not in DATASETS:
        print(f"Unknown dataset type: {dataset_type}")
        return False
    
    dataset_info = DATASETS[dataset_type]
    dataset_dir = os.path.join(output_dir, dataset_type)
    os.makedirs(dataset_dir, exist_ok=True)
    
    print(f"\nDownloading {dataset_info['name']}...")
    print(f"Description: {dataset_info['description']}")
    print(f"Size: ~{dataset_info['size_mb']} MB")
    
    # Download the dataset
    success = False
    
    if method == "gdrive" and "gdrive_id" in dataset_info:
        # Download from Google Drive
        archive_path = os.path.join(dataset_dir, f"{dataset_type}.{dataset_info['type']}")
        success = download_from_gdrive(
            dataset_info["gdrive_id"], 
            archive_path, 
            f"{dataset_info['name']} ({dataset_info['size_mb']} MB)"
        )
    
    elif method == "kaggle" and "kaggle_dataset" in dataset_info:
        # Download from Kaggle
        success = download_from_kaggle(
            dataset_info["kaggle_dataset"], 
            dataset_dir, 
            dataset_info['name']
        )
    
    elif method == "direct" and "url" in dataset_info:
        # Direct download
        archive_path = os.path.join(dataset_dir, f"{dataset_type}.{dataset_info['type']}")
        success = download_file(
            dataset_info["url"], 
            archive_path, 
            f"{dataset_info['name']} ({dataset_info['size_mb']} MB)"
        )
    
    else:
        print(f"Download method {method} not supported for {dataset_type} dataset")
        return False
    
    if not success:
        print(f"Failed to download {dataset_type} dataset")
        return False
    
    # Extract the archive if needed
    if method != "kaggle" and "type" in dataset_info:
        archive_path = os.path.join(dataset_dir, f"{dataset_type}.{dataset_info['type']}")
        if os.path.exists(archive_path):
            success = extract_archive(archive_path, dataset_dir, dataset_info["type"])
            if not success:
                print(f"Failed to extract {dataset_type} dataset")
                return False
    
    print(f"Successfully downloaded and prepared {dataset_type} dataset")
    return True

def main():
    """Main function."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Determine which datasets to download
    datasets_to_download = args.datasets
    if "all" in datasets_to_download:
        datasets_to_download = list(DATASETS.keys())
    
    # Download datasets
    for dataset_type in datasets_to_download:
        success = download_dataset(dataset_type, args.method, args.output_dir)
        if not success:
            print(f"Failed to download {dataset_type} dataset")
    
    # Convert to YOLO format if requested
    if not args.no_convert:
        for dataset_type in datasets_to_download:
            dataset_dir = os.path.join(args.output_dir, dataset_type)
            if os.path.exists(dataset_dir):
                convert_to_yolo_format(dataset_dir, dataset_type)
    
    # Set up DVC if requested
    if args.setup_dvc:
        setup_dvc(args.output_dir, args.dvc_remote)
    
    print("\nDataset download and preparation complete!")

if __name__ == "__main__":
    main()
