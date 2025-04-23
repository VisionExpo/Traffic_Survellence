#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Advanced training script for YOLOv8 models with data augmentation and hyperparameter optimization.
"""

import os
import argparse
import yaml
import torch
import numpy as np
import random
import shutil
from datetime import datetime
from ultralytics import YOLO
from ultralytics.data.augment import Albumentations
import albumentations as A
import cv2
import matplotlib.pyplot as plt
from pathlib import Path

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Advanced training for YOLOv8 models")
    parser.add_argument("--dataset", choices=["bdd100k", "visdrone", "ua-detrac", "combined"], default="combined",
                        help="Dataset to train on (default: combined)")
    parser.add_argument("--model-size", choices=["n", "s", "m", "l", "x"], default="m",
                        help="YOLOv8 model size (default: m)")
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
    parser.add_argument("--augment", action="store_true", default=True,
                        help="Use data augmentation (default: True)")
    parser.add_argument("--hyp-tune", action="store_true",
                        help="Perform hyperparameter tuning")
    parser.add_argument("--early-stop", type=int, default=10,
                        help="Early stopping patience (default: 10)")
    parser.add_argument("--save-period", type=int, default=10,
                        help="Save checkpoint every x epochs (default: 10)")
    return parser.parse_args()

def find_data_yaml(dataset_name):
    """Find the data YAML file for a dataset."""
    # Check in the dataset's yolo directory
    dataset_dir = os.path.join("data/datasets", dataset_name)
    yaml_path = os.path.join(dataset_dir, "dataset.yaml")
    
    if os.path.exists(yaml_path):
        return yaml_path
    
    # Check in the dataset directory
    yaml_path = os.path.join(dataset_dir, "yolo", "dataset.yaml")
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
    os.makedirs(os.path.join(output_dir, "images", "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "images", "val"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels", "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "labels", "val"), exist_ok=True)
    
    # Track classes from all datasets
    all_classes = {}
    next_class_id = 0
    
    # Process each dataset
    for dataset_name in ["bdd100k", "visdrone", "ua-detrac"]:
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
        for split in ["train", "val"]:
            images_dir = os.path.join(dataset_path, dataset_yaml.get(split, f"images/{split}"))
            labels_dir = os.path.join(dataset_path, f"labels/{split}")
            
            if not os.path.exists(images_dir):
                print(f"Warning: Images directory not found for {dataset_name} dataset ({split} split)")
                continue
            
            if not os.path.exists(labels_dir):
                print(f"Warning: Labels directory not found for {dataset_name} dataset ({split} split)")
                continue
            
            # Process each image and label
            for image_file in os.listdir(images_dir):
                if not image_file.endswith((".jpg", ".jpeg", ".png")):
                    continue
                
                # Get the image ID
                image_id = os.path.splitext(image_file)[0]
                
                # Copy the image
                src_image = os.path.join(images_dir, image_file)
                dst_image = os.path.join(output_dir, "images", split, f"{dataset_name}_{image_file}")
                shutil.copy(src_image, dst_image)
                
                # Convert the label
                label_file = f"{image_id}.txt"
                src_label = os.path.join(labels_dir, label_file)
                dst_label = os.path.join(output_dir, "labels", split, f"{dataset_name}_{label_file}")
                
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
        "path": os.path.abspath(output_dir),
        "train": "images/train",
        "val": "images/val",
        "names": {v: k for k, v in all_classes.items()}
    }
    
    yaml_path = os.path.join(output_dir, "dataset.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(combined_yaml, f, default_flow_style=False)
    
    print(f"Combined dataset created with {len(all_classes)} classes")
    return yaml_path

def create_custom_augmentations():
    """Create custom augmentation pipeline using Albumentations."""
    transform = A.Compose([
        # Geometric transformations
        A.HorizontalFlip(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=15, p=0.5),
        A.RandomResizedCrop(height=640, width=640, scale=(0.8, 1.0), ratio=(0.8, 1.2), p=0.5),
        
        # Color transformations
        A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        A.HueSaturationValue(hue_shift_limit=10, sat_shift_limit=30, val_shift_limit=20, p=0.5),
        A.GaussianBlur(blur_limit=(3, 7), p=0.3),
        
        # Weather simulations
        A.RandomRain(p=0.1),
        A.RandomFog(p=0.1),
        A.RandomSunFlare(p=0.1),
        
        # Noise and quality
        A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
        A.ImageCompression(quality_lower=80, quality_upper=100, p=0.3),
        
        # Time of day simulation
        A.RandomBrightnessContrast(brightness_limit=(-0.3, 0.1), contrast_limit=0.1, p=0.2),  # Night
        A.RandomBrightnessContrast(brightness_limit=(0.1, 0.3), contrast_limit=0.1, p=0.2),   # Day
    ], bbox_params=A.BboxParams(format='yolo', label_fields=['class_labels']))
    
    return transform

def visualize_augmentations(dataset_path, output_dir, num_samples=5):
    """Visualize augmentations on sample images."""
    print("Visualizing augmentations...")
    
    # Load dataset YAML
    with open(dataset_path, "r") as f:
        dataset_yaml = yaml.safe_load(f)
    
    # Get dataset path and class names
    dataset_dir = dataset_yaml.get("path", os.path.dirname(dataset_path))
    class_names = dataset_yaml.get("names", {})
    
    # Get train images directory
    train_images_dir = os.path.join(dataset_dir, dataset_yaml.get("train", "images/train"))
    train_labels_dir = os.path.join(dataset_dir, "labels/train")
    
    if not os.path.exists(train_images_dir):
        print(f"Warning: Train images directory not found: {train_images_dir}")
        return
    
    if not os.path.exists(train_labels_dir):
        print(f"Warning: Train labels directory not found: {train_labels_dir}")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Get list of image files
    image_files = [f for f in os.listdir(train_images_dir) if f.endswith((".jpg", ".jpeg", ".png"))]
    
    if not image_files:
        print(f"Warning: No images found in {train_images_dir}")
        return
    
    # Randomly select sample images
    sample_images = random.sample(image_files, min(num_samples, len(image_files)))
    
    # Create augmentation pipeline
    transform = create_custom_augmentations()
    
    # Process each sample image
    for i, image_file in enumerate(sample_images):
        # Get image ID
        image_id = os.path.splitext(image_file)[0]
        
        # Load image
        image_path = os.path.join(train_images_dir, image_file)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Load labels
        label_path = os.path.join(train_labels_dir, f"{image_id}.txt")
        bboxes = []
        class_labels = []
        
        if os.path.exists(label_path):
            with open(label_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        class_id = int(parts[0])
                        x_center, y_center, width, height = map(float, parts[1:5])
                        bboxes.append([x_center, y_center, width, height])
                        class_labels.append(class_id)
        
        # Create figure
        fig, axs = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle(f"Augmentation Examples for {image_file}", fontsize=16)
        
        # Plot original image
        axs[0, 0].imshow(image)
        axs[0, 0].set_title("Original")
        axs[0, 0].axis("off")
        
        # Draw bounding boxes on original image
        image_with_boxes = image.copy()
        h, w = image.shape[:2]
        for bbox, class_id in zip(bboxes, class_labels):
            x_center, y_center, width, height = bbox
            x1 = int((x_center - width / 2) * w)
            y1 = int((y_center - height / 2) * h)
            x2 = int((x_center + width / 2) * w)
            y2 = int((y_center + height / 2) * h)
            
            class_name = class_names.get(class_id, f"Class {class_id}")
            color = plt.cm.hsv(class_id / len(class_names))
            color = (color[0] * 255, color[1] * 255, color[2] * 255)
            
            cv2.rectangle(image_with_boxes, (x1, y1), (x2, y2), color, 2)
            cv2.putText(image_with_boxes, class_name, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        axs[0, 1].imshow(image_with_boxes)
        axs[0, 1].set_title("Original with Boxes")
        axs[0, 1].axis("off")
        
        # Generate and plot augmentations
        for j in range(4):
            row, col = (j + 2) // 3, (j + 2) % 3
            
            # Apply augmentation
            augmented = transform(image=image, bboxes=bboxes, class_labels=class_labels)
            aug_image = augmented["image"]
            aug_bboxes = augmented["bboxes"]
            aug_class_labels = augmented["class_labels"]
            
            # Draw bounding boxes on augmented image
            aug_image_with_boxes = aug_image.copy()
            h, w = aug_image.shape[:2]
            for bbox, class_id in zip(aug_bboxes, aug_class_labels):
                x_center, y_center, width, height = bbox
                x1 = int((x_center - width / 2) * w)
                y1 = int((y_center - height / 2) * h)
                x2 = int((x_center + width / 2) * w)
                y2 = int((y_center + height / 2) * h)
                
                class_name = class_names.get(class_id, f"Class {class_id}")
                color = plt.cm.hsv(class_id / len(class_names))
                color = (color[0] * 255, color[1] * 255, color[2] * 255)
                
                cv2.rectangle(aug_image_with_boxes, (x1, y1), (x2, y2), color, 2)
                cv2.putText(aug_image_with_boxes, class_name, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            axs[row, col].imshow(aug_image_with_boxes)
            axs[row, col].set_title(f"Augmentation {j+1}")
            axs[row, col].axis("off")
        
        # Save figure
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"augmentation_example_{i+1}.jpg"))
        plt.close(fig)
    
    print(f"Augmentation examples saved to {output_dir}")

def train_yolo_model(args):
    """Train a YOLOv8 model with advanced settings."""
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
    model_name = f"yolov8{args.model_size}_{args.dataset}_advanced_{timestamp}"
    output_dir = os.path.join(args.output_dir, args.dataset)
    os.makedirs(output_dir, exist_ok=True)
    
    # Visualize augmentations
    if args.augment:
        visualize_augmentations(data_yaml, os.path.join(output_dir, "augmentation_examples"))
    
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
    print(f"Augmentation: {'Enabled' if args.augment else 'Disabled'}")
    
    # Set up hyperparameters
    hyp = {
        "lr0": 0.01,              # Initial learning rate
        "lrf": 0.01,              # Final learning rate factor
        "momentum": 0.937,        # SGD momentum
        "weight_decay": 0.0005,   # Optimizer weight decay
        "warmup_epochs": 3.0,     # Warmup epochs
        "warmup_momentum": 0.8,   # Warmup momentum
        "warmup_bias_lr": 0.1,    # Warmup bias learning rate
        "box": 7.5,               # Box loss gain
        "cls": 0.5,               # Cls loss gain
        "dfl": 1.5,               # DFL loss gain
        "fl_gamma": 0.0,          # Focal loss gamma
        "hsv_h": 0.015,           # Image HSV-Hue augmentation
        "hsv_s": 0.7,             # Image HSV-Saturation augmentation
        "hsv_v": 0.4,             # Image HSV-Value augmentation
        "degrees": 0.0,           # Image rotation (+/- deg)
        "translate": 0.1,         # Image translation (+/- fraction)
        "scale": 0.5,             # Image scale (+/- gain)
        "shear": 0.0,             # Image shear (+/- deg)
        "perspective": 0.0,       # Image perspective (+/- fraction)
        "flipud": 0.0,            # Image flip up-down (probability)
        "fliplr": 0.5,            # Image flip left-right (probability)
        "mosaic": 1.0,            # Image mosaic (probability)
        "mixup": 0.0,             # Image mixup (probability)
        "copy_paste": 0.0         # Segment copy-paste (probability)
    }
    
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
        resume=args.resume,
        pretrained=args.pretrained,
        optimizer="SGD",  # SGD, Adam, AdamW
        lr0=hyp["lr0"],
        lrf=hyp["lrf"],
        momentum=hyp["momentum"],
        weight_decay=hyp["weight_decay"],
        warmup_epochs=hyp["warmup_epochs"],
        warmup_momentum=hyp["warmup_momentum"],
        warmup_bias_lr=hyp["warmup_bias_lr"],
        box=hyp["box"],
        cls=hyp["cls"],
        dfl=hyp["dfl"],
        fl_gamma=hyp["fl_gamma"],
        hsv_h=hyp["hsv_h"],
        hsv_s=hyp["hsv_s"],
        hsv_v=hyp["hsv_v"],
        degrees=hyp["degrees"],
        translate=hyp["translate"],
        scale=hyp["scale"],
        shear=hyp["shear"],
        perspective=hyp["perspective"],
        flipud=hyp["flipud"],
        fliplr=hyp["fliplr"],
        mosaic=hyp["mosaic"],
        mixup=hyp["mixup"],
        copy_paste=hyp["copy_paste"],
        patience=args.early_stop,
        save_period=args.save_period,
        verbose=True
    )
    
    # Copy the best model to a more accessible location
    best_model_path = os.path.join(output_dir, model_name, "weights", "best.pt")
    if os.path.exists(best_model_path):
        final_model_path = os.path.join(output_dir, f"yolov8{args.model_size}_{args.dataset}_advanced_best.pt")
        shutil.copy(best_model_path, final_model_path)
        print(f"Best model saved to {final_model_path}")
    
    # Validate the model
    print("Validating the model...")
    metrics = model.val()
    
    print("Validation results:")
    print(f"mAP50-95: {metrics.box.map}")
    print(f"mAP50: {metrics.box.map50}")
    print(f"mAP75: {metrics.box.map75}")
    
    # Export the model to ONNX format for deployment
    print("Exporting model to ONNX format...")
    model.export(format="onnx")
    
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
    
    # Set random seed for reproducibility
    seed = 42
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    # Train the model
    train_yolo_model(args)

if __name__ == "__main__":
    main()
