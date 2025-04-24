#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to create sample data for testing the traffic surveillance system.
This is useful when you don't have access to the full datasets.
"""

import os
import argparse
import cv2
import numpy as np
import random
import shutil
import yaml
from tqdm import tqdm

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Create sample data for testing")
    parser.add_argument("--output-dir", default="data/datasets",
                        help="Output directory (default: data/datasets)")
    parser.add_argument("--num-samples", type=int, default=100,
                        help="Number of samples to generate (default: 100)")
    parser.add_argument("--image-size", type=int, default=640,
                        help="Image size (default: 640)")
    parser.add_argument("--setup-dvc", action="store_true",
                        help="Set up DVC for the sample data")
    return parser.parse_args()

def create_sample_helmet_dataset(output_dir, num_samples, image_size):
    """Create a sample helmet detection dataset."""
    print(f"Creating sample helmet detection dataset with {num_samples} images...")
    
    # Create directories
    dataset_dir = os.path.join(output_dir, "helmet")
    images_dir = os.path.join(dataset_dir, "images")
    annotations_dir = os.path.join(dataset_dir, "annotations")
    yolo_dir = os.path.join(dataset_dir, "yolo")
    yolo_images_dir = os.path.join(yolo_dir, "images")
    yolo_labels_dir = os.path.join(yolo_dir, "labels")
    
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(annotations_dir, exist_ok=True)
    os.makedirs(yolo_images_dir, exist_ok=True)
    os.makedirs(yolo_labels_dir, exist_ok=True)
    
    # Define colors for visualization
    colors = {
        "helmet": (0, 255, 0),  # Green
        "head": (0, 0, 255),    # Red
        "person": (255, 0, 0)   # Blue
    }
    
    # Generate sample images and annotations
    for i in tqdm(range(num_samples), desc="Generating helmet samples"):
        # Create a blank image
        img = np.zeros((image_size, image_size, 3), dtype=np.uint8)
        
        # Add a background (road scene)
        cv2.rectangle(img, (0, 0), (image_size, image_size), (100, 100, 100), -1)
        
        # Add some random lines for road markings
        for _ in range(5):
            x1 = random.randint(0, image_size)
            y1 = random.randint(0, image_size)
            x2 = random.randint(0, image_size)
            y2 = random.randint(0, image_size)
            cv2.line(img, (x1, y1), (x2, y2), (200, 200, 200), 2)
        
        # Create XML annotation
        xml_content = f"""<annotation>
    <folder>images</folder>
    <filename>helmet_{i:04d}.jpg</filename>
    <size>
        <width>{image_size}</width>
        <height>{image_size}</height>
        <depth>3</depth>
    </size>
"""
        
        # Add random objects (person, head, helmet)
        num_objects = random.randint(1, 3)
        
        # YOLO format annotations
        yolo_annotations = []
        
        for j in range(num_objects):
            # Decide object type
            if random.random() < 0.7:  # 70% chance of having a helmet
                obj_type = "helmet"
                class_id = 0
            else:
                if random.random() < 0.5:
                    obj_type = "head"
                    class_id = 1
                else:
                    obj_type = "person"
                    class_id = 2
            
            # Generate random bounding box
            w = random.randint(50, 150)
            h = random.randint(50, 150)
            x = random.randint(0, image_size - w)
            y = random.randint(0, image_size - h)
            
            # Draw the object
            cv2.rectangle(img, (x, y), (x + w, y + h), colors[obj_type], -1)
            
            # Add text label
            cv2.putText(img, obj_type, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Add to XML annotation
            xml_content += f"""    <object>
        <name>{obj_type}</name>
        <bndbox>
            <xmin>{x}</xmin>
            <ymin>{y}</ymin>
            <xmax>{x + w}</xmax>
            <ymax>{y + h}</ymax>
        </bndbox>
    </object>
"""
            
            # Add to YOLO annotations
            # YOLO format: class_id center_x center_y width height (normalized)
            center_x = (x + w / 2) / image_size
            center_y = (y + h / 2) / image_size
            norm_width = w / image_size
            norm_height = h / image_size
            yolo_annotations.append(f"{class_id} {center_x:.6f} {center_y:.6f} {norm_width:.6f} {norm_height:.6f}")
        
        # Close XML annotation
        xml_content += "</annotation>"
        
        # Save the image
        image_path = os.path.join(images_dir, f"helmet_{i:04d}.jpg")
        cv2.imwrite(image_path, img)
        
        # Save the XML annotation
        xml_path = os.path.join(annotations_dir, f"helmet_{i:04d}.xml")
        with open(xml_path, "w") as f:
            f.write(xml_content)
        
        # Save YOLO format
        # Copy the image
        shutil.copy(image_path, os.path.join(yolo_images_dir, f"helmet_{i:04d}.jpg"))
        
        # Save the YOLO annotation
        yolo_path = os.path.join(yolo_labels_dir, f"helmet_{i:04d}.txt")
        with open(yolo_path, "w") as f:
            f.write("\n".join(yolo_annotations))
    
    # Create dataset.yaml for YOLO
    dataset_yaml = {
        "path": os.path.abspath(yolo_dir),
        "train": "images",
        "val": "images",
        "names": {
            0: "helmet",
            1: "head",
            2: "person"
        }
    }
    
    with open(os.path.join(yolo_dir, "dataset.yaml"), "w") as f:
        yaml.dump(dataset_yaml, f, default_flow_style=False)
    
    print(f"Created sample helmet detection dataset with {num_samples} images")
    return dataset_dir

def create_sample_license_plate_dataset(output_dir, num_samples, image_size):
    """Create a sample license plate dataset."""
    print(f"Creating sample license plate dataset with {num_samples} images...")
    
    # Create directories
    dataset_dir = os.path.join(output_dir, "license_plate")
    images_dir = os.path.join(dataset_dir, "images")
    annotations_dir = os.path.join(dataset_dir, "annotations")
    yolo_dir = os.path.join(dataset_dir, "yolo")
    yolo_images_dir = os.path.join(yolo_dir, "images")
    yolo_labels_dir = os.path.join(yolo_dir, "labels")
    ocr_dir = os.path.join(dataset_dir, "ocr")
    ocr_train_dir = os.path.join(ocr_dir, "train")
    ocr_val_dir = os.path.join(ocr_dir, "val")
    
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(annotations_dir, exist_ok=True)
    os.makedirs(yolo_images_dir, exist_ok=True)
    os.makedirs(yolo_labels_dir, exist_ok=True)
    os.makedirs(ocr_train_dir, exist_ok=True)
    os.makedirs(ocr_val_dir, exist_ok=True)
    
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
    for i in tqdm(range(num_samples), desc="Generating license plate samples"):
        # Create a blank image
        img = np.zeros((image_size, image_size, 3), dtype=np.uint8)
        
        # Add a background (road scene)
        cv2.rectangle(img, (0, 0), (image_size, image_size), (100, 100, 100), -1)
        
        # Generate a random license plate
        plate_text = generate_license_plate()
        
        # Create a license plate image
        plate_width = random.randint(100, 200)
        plate_height = int(plate_width * 0.2)  # License plates are typically 5:1 ratio
        plate_x = random.randint(0, image_size - plate_width)
        plate_y = random.randint(0, image_size - plate_height)
        
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
        
        # Save annotation (simple text file with plate text and coordinates)
        annotation_path = os.path.join(annotations_dir, f"plate_{i:04d}.txt")
        with open(annotation_path, "w") as f:
            f.write(f"{plate_text},{plate_x},{plate_y},{plate_x+plate_width},{plate_y+plate_height}")
        
        # Save YOLO format
        # Copy the image
        shutil.copy(image_path, os.path.join(yolo_images_dir, f"plate_{i:04d}.jpg"))
        
        # YOLO format: class_id center_x center_y width height (normalized)
        center_x = (plate_x + plate_width / 2) / image_size
        center_y = (plate_y + plate_height / 2) / image_size
        norm_width = plate_width / image_size
        norm_height = plate_height / image_size
        
        # Save the YOLO annotation
        yolo_path = os.path.join(yolo_labels_dir, f"plate_{i:04d}.txt")
        with open(yolo_path, "w") as f:
            f.write(f"0 {center_x:.6f} {center_y:.6f} {norm_width:.6f} {norm_height:.6f}")
        
        # Extract the license plate for OCR training
        plate_img = img[plate_y:plate_y+plate_height, plate_x:plate_x+plate_width]
        
        # Decide whether to put in train or val set (80/20 split)
        if random.random() < 0.8:
            ocr_path = os.path.join(ocr_train_dir, f"{plate_text}_{i:04d}.jpg")
        else:
            ocr_path = os.path.join(ocr_val_dir, f"{plate_text}_{i:04d}.jpg")
        
        cv2.imwrite(ocr_path, plate_img)
    
    # Create dataset.yaml for YOLO
    dataset_yaml = {
        "path": os.path.abspath(yolo_dir),
        "train": "images",
        "val": "images",
        "names": {
            0: "license_plate"
        }
    }
    
    with open(os.path.join(yolo_dir, "dataset.yaml"), "w") as f:
        yaml.dump(dataset_yaml, f, default_flow_style=False)
    
    # Create metadata.txt for OCR
    with open(os.path.join(ocr_dir, "metadata.txt"), "w") as f:
        f.write(f"Train size: {int(num_samples * 0.8)}\n")
        f.write(f"Validation size: {int(num_samples * 0.2)}\n")
        f.write(f"Characters: A,B,C,D,E,F,G,H,J,K,L,M,N,P,Q,R,S,T,U,V,W,X,Y,Z,0,1,2,3,4,5,6,7,8,9,-\n")
    
    print(f"Created sample license plate dataset with {num_samples} images")
    return dataset_dir

def create_sample_vehicle_dataset(output_dir, num_samples, image_size):
    """Create a sample vehicle dataset."""
    print(f"Creating sample vehicle dataset with {num_samples} images...")
    
    # Create directories
    dataset_dir = os.path.join(output_dir, "vehicle")
    images_dir = os.path.join(dataset_dir, "images")
    annotations_dir = os.path.join(dataset_dir, "annotations")
    yolo_dir = os.path.join(dataset_dir, "yolo")
    yolo_images_dir = os.path.join(yolo_dir, "images")
    yolo_labels_dir = os.path.join(yolo_dir, "labels")
    
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(annotations_dir, exist_ok=True)
    os.makedirs(yolo_images_dir, exist_ok=True)
    os.makedirs(yolo_labels_dir, exist_ok=True)
    
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
    for i in tqdm(range(num_samples), desc="Generating vehicle samples"):
        # Create a blank image (road scene)
        img = np.zeros((image_size, image_size, 3), dtype=np.uint8)
        
        # Add a background (road)
        cv2.rectangle(img, (0, 0), (image_size, image_size), (100, 100, 100), -1)
        
        # Add road markings
        # Center line
        cv2.line(img, (0, image_size // 2), (image_size, image_size // 2), (255, 255, 255), 2)
        
        # Lane markings
        for x in range(0, image_size, 50):
            cv2.line(img, (x, image_size // 4), (x + 30, image_size // 4), (255, 255, 255), 2)
            cv2.line(img, (x, 3 * image_size // 4), (x + 30, 3 * image_size // 4), (255, 255, 255), 2)
        
        # Create annotation file
        annotation_content = ""
        
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
            
            x = random.randint(0, image_size - w)
            y = random.randint(0, image_size - h)
            
            # Draw the vehicle
            cv2.rectangle(img, (x, y), (x + w, y + h), vehicle_color, -1)
            
            # Add text label
            cv2.putText(img, vehicle_type, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
            # Add to annotation
            annotation_content += f"{vehicle_type},{x},{y},{x+w},{y+h}\n"
            
            # Add to YOLO annotations
            # YOLO format: class_id center_x center_y width height (normalized)
            center_x = (x + w / 2) / image_size
            center_y = (y + h / 2) / image_size
            norm_width = w / image_size
            norm_height = h / image_size
            yolo_annotations.append(f"{vehicle_idx} {center_x:.6f} {center_y:.6f} {norm_width:.6f} {norm_height:.6f}")
        
        # Save the image
        image_path = os.path.join(images_dir, f"vehicle_{i:04d}.jpg")
        cv2.imwrite(image_path, img)
        
        # Save the annotation
        annotation_path = os.path.join(annotations_dir, f"vehicle_{i:04d}.txt")
        with open(annotation_path, "w") as f:
            f.write(annotation_content)
        
        # Save YOLO format
        # Copy the image
        shutil.copy(image_path, os.path.join(yolo_images_dir, f"vehicle_{i:04d}.jpg"))
        
        # Save the YOLO annotation
        yolo_path = os.path.join(yolo_labels_dir, f"vehicle_{i:04d}.txt")
        with open(yolo_path, "w") as f:
            f.write("\n".join(yolo_annotations))
    
    # Create dataset.yaml for YOLO
    dataset_yaml = {
        "path": os.path.abspath(yolo_dir),
        "train": "images",
        "val": "images",
        "names": {i: vehicle_type for i, vehicle_type in enumerate(vehicle_types)}
    }
    
    with open(os.path.join(yolo_dir, "dataset.yaml"), "w") as f:
        yaml.dump(dataset_yaml, f, default_flow_style=False)
    
    print(f"Created sample vehicle dataset with {num_samples} images")
    return dataset_dir

def create_combined_dataset(output_dir, datasets):
    """Create a combined dataset from individual datasets."""
    print("Creating combined dataset...")
    
    # Create output directories
    combined_dir = os.path.join(output_dir, "combined")
    yolo_dir = os.path.join(combined_dir, "yolo")
    yolo_images_dir = os.path.join(yolo_dir, "images")
    yolo_labels_dir = os.path.join(yolo_dir, "labels")
    
    os.makedirs(yolo_images_dir, exist_ok=True)
    os.makedirs(yolo_labels_dir, exist_ok=True)
    
    # Track classes from all datasets
    all_classes = {}
    next_class_id = 0
    
    # Process each dataset
    for dataset_name, dataset_dir in datasets.items():
        # Find the dataset YAML
        yaml_path = os.path.join(dataset_dir, "yolo", "dataset.yaml")
        if not os.path.exists(yaml_path):
            print(f"Warning: Could not find YAML file for {dataset_name} dataset")
            continue
        
        # Load the dataset YAML
        with open(yaml_path, "r") as f:
            dataset_yaml = yaml.safe_load(f)
        
        # Get the dataset path and class names
        dataset_path = dataset_yaml.get("path", os.path.dirname(yaml_path))
        class_names = dataset_yaml.get("names", {})
        
        # Map class IDs to new combined IDs
        class_map = {}
        for class_id, class_name in class_names.items():
            if class_name not in all_classes:
                all_classes[class_name] = next_class_id
                next_class_id += 1
            class_map[class_id] = all_classes[class_name]
        
        # Copy images and convert labels
        images_dir = os.path.join(dataset_path, dataset_yaml.get("train", "images"))
        labels_dir = os.path.join(dataset_path, "labels")
        
        if not os.path.exists(images_dir):
            print(f"Warning: Images directory not found for {dataset_name} dataset")
            continue
        
        if not os.path.exists(labels_dir):
            print(f"Warning: Labels directory not found for {dataset_name} dataset")
            continue
        
        # Process each image and label
        for image_file in os.listdir(images_dir):
            if not image_file.endswith((".jpg", ".jpeg", ".png")):
                continue
            
            # Get the image ID
            image_id = os.path.splitext(image_file)[0]
            
            # Copy the image
            src_image = os.path.join(images_dir, image_file)
            dst_image = os.path.join(yolo_images_dir, f"{dataset_name}_{image_file}")
            shutil.copy(src_image, dst_image)
            
            # Convert the label
            label_file = f"{image_id}.txt"
            src_label = os.path.join(labels_dir, label_file)
            dst_label = os.path.join(yolo_labels_dir, f"{dataset_name}_{label_file}")
            
            if os.path.exists(src_label):
                with open(src_label, "r") as f_in, open(dst_label, "w") as f_out:
                    for line in f_in:
                        parts = line.strip().split()
                        if len(parts) >= 5:
                            class_id = int(parts[0])
                            new_class_id = class_map.get(class_id, class_id)
                            f_out.write(f"{new_class_id} {' '.join(parts[1:])}\n")
    
    # Create the combined dataset YAML
    combined_yaml = {
        "path": os.path.abspath(yolo_dir),
        "train": "images",
        "val": "images",
        "names": {v: k for k, v in all_classes.items()}
    }
    
    with open(os.path.join(yolo_dir, "dataset.yaml"), "w") as f:
        yaml.dump(combined_yaml, f, default_flow_style=False)
    
    print(f"Combined dataset created with {len(all_classes)} classes")
    return combined_dir

def setup_dvc(datasets_dir):
    """Set up DVC for the sample data."""
    print("Setting up DVC for sample data...")
    
    try:
        # Initialize DVC if not already initialized
        if not os.path.exists(".dvc"):
            os.system("dvc init")
            print("DVC initialized")
        
        # Add datasets to DVC
        for dataset_type in os.listdir(datasets_dir):
            dataset_path = os.path.join(datasets_dir, dataset_type)
            if os.path.isdir(dataset_path):
                os.system(f"dvc add {dataset_path}")
                print(f"Added {dataset_path} to DVC")
        
        # Set up remote storage
        os.system("dvc remote add --default gdrive gdrive://traffic_surveillance_data")
        os.system("dvc remote modify gdrive gdrive_acknowledge_abuse true")
        print("Added Google Drive remote storage")
        
        print("DVC setup complete!")
        print("\nTo push data to remote storage, run:")
        print("dvc push")
        print("\nThis will open a browser window for Google authentication.")
        
        return True
    
    except Exception as e:
        print(f"Error setting up DVC: {e}")
        return False

def main():
    """Main function."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create sample datasets
    datasets = {}
    
    # Helmet dataset
    helmet_dir = create_sample_helmet_dataset(args.output_dir, args.num_samples, args.image_size)
    datasets["helmet"] = helmet_dir
    
    # License plate dataset
    license_plate_dir = create_sample_license_plate_dataset(args.output_dir, args.num_samples, args.image_size)
    datasets["license_plate"] = license_plate_dir
    
    # Vehicle dataset
    vehicle_dir = create_sample_vehicle_dataset(args.output_dir, args.num_samples, args.image_size)
    datasets["vehicle"] = vehicle_dir
    
    # Create combined dataset
    combined_dir = create_combined_dataset(args.output_dir, datasets)
    datasets["combined"] = combined_dir
    
    # Set up DVC if requested
    if args.setup_dvc:
        setup_dvc(args.output_dir)
    
    print("\nSample data creation complete!")

if __name__ == "__main__":
    main()
