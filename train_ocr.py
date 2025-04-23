#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to fine-tune OCR models for license plate recognition.
"""

import os
import argparse
import torch
import numpy as np
import cv2
from tqdm import tqdm
import random
import shutil
from datetime import datetime

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Fine-tune OCR models for license plate recognition")
    parser.add_argument("--dataset", default="license_plate",
                        help="Dataset to train on (default: license_plate)")
    parser.add_argument("--model-type", choices=["easyocr", "tesseract"], default="easyocr",
                        help="OCR model type (default: easyocr)")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs (default: 10)")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size (default: 32)")
    parser.add_argument("--learning-rate", type=float, default=0.001,
                        help="Learning rate (default: 0.001)")
    parser.add_argument("--device", type=str, default="",
                        help="Device to train on (default: auto-detect)")
    parser.add_argument("--output-dir", default="data/models/ocr",
                        help="Output directory for trained models (default: data/models/ocr)")
    return parser.parse_args()

def prepare_license_plate_dataset(dataset_dir, output_dir, split_ratio=0.8):
    """Prepare license plate dataset for OCR training."""
    print("Preparing license plate dataset for OCR training...")
    
    # Create output directories
    os.makedirs(os.path.join(output_dir, "train"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "val"), exist_ok=True)
    
    # Find license plate images and annotations
    images = []
    annotations = {}
    
    # This is a placeholder - actual implementation would depend on the dataset format
    # For CCPD dataset, the license plate text is encoded in the filename
    
    # Example for CCPD dataset
    provinces = ["皖", "沪", "津", "渝", "冀", "晋", "蒙", "辽", "吉", "黑", "苏", "浙", "京", "闽", "赣", "鲁", "豫", "鄂", "湘", "粤", "桂", "琼", "川", "贵", "云", "藏", "陕", "甘", "青", "宁", "新"]
    alphabets = ["A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
    ads = ["A", "B", "C", "D", "E", "F", "G", "H", "J", "K", "L", "M", "N", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    
    # Search for CCPD images
    for root, _, files in os.walk(dataset_dir):
        for file in files:
            if file.endswith((".jpg", ".jpeg", ".png")) and file.startswith("CCPD"):
                image_path = os.path.join(root, file)
                
                # Extract license plate text from filename
                # Format: CCPD2019/base/[province]-[alphabet]-[ads]-[ads]-[ads]-[ads]-[ads].jpg
                try:
                    parts = file.split("-")
                    if len(parts) >= 7:
                        province_idx = int(parts[0])
                        alphabet_idx = int(parts[1])
                        ad_indices = [int(parts[2]), int(parts[3]), int(parts[4]), int(parts[5]), int(parts[6].split(".")[0])]
                        
                        license_text = provinces[province_idx] + alphabets[alphabet_idx]
                        for ad_idx in ad_indices:
                            license_text += ads[ad_idx]
                        
                        images.append(image_path)
                        annotations[image_path] = license_text
                except:
                    # Skip files that don't match the expected format
                    continue
    
    print(f"Found {len(images)} license plate images with annotations")
    
    # Split into train and validation sets
    random.shuffle(images)
    split_idx = int(len(images) * split_ratio)
    train_images = images[:split_idx]
    val_images = images[split_idx:]
    
    # Copy images to output directories
    for image_set, output_subdir in [(train_images, "train"), (val_images, "val")]:
        for image_path in tqdm(image_set, desc=f"Copying {output_subdir} images"):
            # Read the image
            img = cv2.imread(image_path)
            if img is None:
                continue
            
            # Get the license plate text
            license_text = annotations.get(image_path, "")
            if not license_text:
                continue
            
            # Save the image with the license text as the filename
            output_path = os.path.join(output_dir, output_subdir, f"{license_text}_{os.path.basename(image_path)}")
            cv2.imwrite(output_path, img)
    
    print(f"Prepared {len(train_images)} training images and {len(val_images)} validation images")
    
    # Create a metadata file
    metadata = {
        "train_size": len(train_images),
        "val_size": len(val_images),
        "characters": sorted(set("".join(annotations.values())))
    }
    
    with open(os.path.join(output_dir, "metadata.txt"), "w", encoding="utf-8") as f:
        f.write(f"Train size: {metadata['train_size']}\n")
        f.write(f"Validation size: {metadata['val_size']}\n")
        f.write(f"Characters: {','.join(metadata['characters'])}\n")
    
    return metadata

def fine_tune_easyocr(train_dir, val_dir, characters, args):
    """Fine-tune EasyOCR model."""
    try:
        import easyocr
        from easyocr.trainer import Trainer
    except ImportError:
        print("Error: EasyOCR not installed or trainer module not available")
        print("Please install EasyOCR with: pip install easyocr")
        return False
    
    print("Fine-tuning EasyOCR model...")
    
    # Determine device
    device = args.device
    if not device:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(args.output_dir, f"easyocr_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # Create trainer
    trainer = Trainer(
        characters=characters,
        train_data_path=train_dir,
        val_data_path=val_dir,
        output_path=output_dir,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        num_epochs=args.epochs,
        device=device
    )
    
    # Train the model
    trainer.train()
    
    # Copy the best model to a more accessible location
    best_model_path = os.path.join(output_dir, "best_model.pth")
    if os.path.exists(best_model_path):
        final_model_path = os.path.join(args.output_dir, "easyocr_license_plate_best.pth")
        shutil.copy(best_model_path, final_model_path)
        print(f"Best model saved to {final_model_path}")
    
    print("Fine-tuning complete!")
    return True

def fine_tune_tesseract(train_dir, val_dir, args):
    """Fine-tune Tesseract OCR model."""
    print("Fine-tuning Tesseract OCR model...")
    print("Note: Tesseract fine-tuning requires tesstrain tools and is more complex")
    print("This is a placeholder for Tesseract fine-tuning")
    
    # Check if tesseract is installed
    try:
        import pytesseract
        tesseract_version = pytesseract.get_tesseract_version()
        print(f"Tesseract version: {tesseract_version}")
    except:
        print("Error: pytesseract not installed or Tesseract not found")
        print("Please install Tesseract and pytesseract")
        return False
    
    # Actual Tesseract fine-tuning would involve:
    # 1. Converting images to box files
    # 2. Creating training data
    # 3. Running tesstrain
    # 4. Generating a new traineddata file
    
    print("Tesseract fine-tuning not implemented in this script")
    print("Please refer to Tesseract documentation for fine-tuning instructions")
    
    return False

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
    
    # Prepare dataset
    dataset_dir = os.path.join("data/datasets", args.dataset)
    ocr_dataset_dir = os.path.join("data/datasets", args.dataset, "ocr")
    os.makedirs(ocr_dataset_dir, exist_ok=True)
    
    metadata = prepare_license_plate_dataset(dataset_dir, ocr_dataset_dir)
    
    # Fine-tune OCR model
    if args.model_type == "easyocr":
        fine_tune_easyocr(
            os.path.join(ocr_dataset_dir, "train"),
            os.path.join(ocr_dataset_dir, "val"),
            metadata["characters"],
            args
        )
    elif args.model_type == "tesseract":
        fine_tune_tesseract(
            os.path.join(ocr_dataset_dir, "train"),
            os.path.join(ocr_dataset_dir, "val"),
            args
        )
    else:
        print(f"Unsupported OCR model type: {args.model_type}")

if __name__ == "__main__":
    main()
