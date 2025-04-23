#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to train custom YOLOv8 models for traffic surveillance.
"""

import os
import argparse
import yaml
import torch
from ultralytics import YOLO
import shutil
from datetime import datetime

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train custom YOLOv8 models for traffic surveillance")
    parser.add_argument("--task", choices=["detect", "segment"], default="detect",
                        help="Training task (default: detect)")
    parser.add_argument("--model-size", choices=["n", "s", "m", "l", "x"], default="m",
                        help="YOLOv8 model size (default: m)")
    parser.add_argument("--dataset", choices=["helmet", "license_plate", "vehicle", "combined"], default="combined",
                        help="Dataset to train on (default: combined)")
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs (default: 100)")
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size (default: 16)")
    parser.add_argument("--img-size", type=int, default=640,
                        help="Image size (default: 640)")
    parser.add_argument("--data-yaml", type=str, default=None,
                        help="Path to data YAML file (default: auto-detect)")
    parser.add_argument("--device", type=str, default="",
                        help="Device to train on (default: auto-detect)")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of worker threads (default: 4)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from last checkpoint")
    parser.add_argument("--pretrained", action="store_true", default=True,
                        help="Use pretrained weights (default: True)")
    parser.add_argument("--output-dir", default="data/models",
                        help="Output directory for trained models (default: data/models)")
    return parser.parse_args()

def find_data_yaml(dataset_name):
    """Find the data YAML file for a dataset."""
    # Check in the dataset's yolo directory
    dataset_dir = os.path.join("data/datasets", dataset_name)
    yaml_path = os.path.join(dataset_dir, "yolo", "dataset.yaml")
    
    if os.path.exists(yaml_path):
        return yaml_path
    
    # Check in the dataset directory
    yaml_path = os.path.join(dataset_dir, "dataset.yaml")
    if os.path.exists(yaml_path):
        return yaml_path
    
    # Check for common names
    for name in ["data.yaml", f"{dataset_name}.yaml"]:
        yaml_path = os.path.join(dataset_dir, name)
        if os.path.exists(yaml_path):
            return yaml_path
    
    return None

def create_combined_dataset(output_dir="data/datasets/combined"):
    """Create a combined dataset from individual datasets."""
    print("Creating combined dataset...")
    
    # Create output directories
    os.makedirs(os.path.join(output_dir, "yolo", "images"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "yolo", "labels"), exist_ok=True)
    
    # Track classes from all datasets
    all_classes = {}
    next_class_id = 0
    
    # Process each dataset
    for dataset_name in ["helmet", "license_plate", "vehicle"]:
        # Find the dataset YAML
        yaml_path = find_data_yaml(dataset_name)
        if not yaml_path:
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
            dst_image = os.path.join(output_dir, "yolo", "images", f"{dataset_name}_{image_file}")
            shutil.copy(src_image, dst_image)
            
            # Convert the label
            label_file = f"{image_id}.txt"
            src_label = os.path.join(labels_dir, label_file)
            dst_label = os.path.join(output_dir, "yolo", "labels", f"{dataset_name}_{label_file}")
            
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
        "path": os.path.abspath(os.path.join(output_dir, "yolo")),
        "train": "images",
        "val": "images",
        "names": {v: k for k, v in all_classes.items()}
    }
    
    yaml_path = os.path.join(output_dir, "yolo", "dataset.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(combined_yaml, f, default_flow_style=False)
    
    print(f"Combined dataset created with {len(all_classes)} classes")
    return yaml_path

def train_yolo_model(args):
    """Train a YOLOv8 model."""
    # Find or create the data YAML file
    if args.data_yaml:
        data_yaml = args.data_yaml
    elif args.dataset == "combined":
        data_yaml = create_combined_dataset()
    else:
        data_yaml = find_data_yaml(args.dataset)
    
    if not data_yaml or not os.path.exists(data_yaml):
        print(f"Error: Could not find data YAML file for {args.dataset} dataset")
        return False
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = f"yolov8{args.model_size}_{args.dataset}_{timestamp}"
    output_dir = os.path.join(args.output_dir, args.dataset)
    os.makedirs(output_dir, exist_ok=True)
    
    # Determine device
    device = args.device
    if not device:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # Load the model
    if args.pretrained:
        model = YOLO(f"yolov8{args.model_size}.pt")
    else:
        model = YOLO(f"yolov8{args.model_size}.yaml")
    
    print(f"Training YOLOv8{args.model_size} model on {args.dataset} dataset")
    print(f"Data YAML: {data_yaml}")
    print(f"Device: {device}")
    print(f"Output directory: {output_dir}")
    
    # Train the model
    results = model.train(
        data=data_yaml,
        epochs=args.epochs,
        batch=args.batch_size,
        imgsz=args.img_size,
        device=device,
        workers=args.workers,
        project=output_dir,
        name=model_name,
        resume=args.resume
    )
    
    # Copy the best model to a more accessible location
    best_model_path = os.path.join(output_dir, model_name, "weights", "best.pt")
    if os.path.exists(best_model_path):
        final_model_path = os.path.join(output_dir, f"yolov8{args.model_size}_{args.dataset}_best.pt")
        shutil.copy(best_model_path, final_model_path)
        print(f"Best model saved to {final_model_path}")
    
    print("Training complete!")
    return True

def main():
    """Main function."""
    args = parse_args()
    
    # Check if CUDA is available
    if torch.cuda.is_available():
        print(f"CUDA is available. Found {torch.cuda.device_count()} device(s).")
        for i in range(torch.cuda.device_count()):
            print(f"  Device {i}: {torch.cuda.get_device_name(i)}")
    else:
        print("CUDA is not available. Training will use CPU, which may be slow.")
    
    # Train the model
    train_yolo_model(args)

if __name__ == "__main__":
    main()
