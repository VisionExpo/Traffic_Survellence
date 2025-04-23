#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simple script to train YOLOv8 models on our sample datasets.
"""

import os
import argparse
from ultralytics import YOLO

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train YOLOv8 models on sample datasets")
    parser.add_argument("--dataset", choices=["helmet", "license_plate", "vehicle", "combined"], default="combined",
                        help="Dataset to train on (default: combined)")
    parser.add_argument("--model-size", choices=["n", "s", "m"], default="n",
                        help="YOLOv8 model size (default: n)")
    parser.add_argument("--epochs", type=int, default=10,
                        help="Number of training epochs (default: 10)")
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Batch size (default: 8)")
    parser.add_argument("--img-size", type=int, default=640,
                        help="Image size (default: 640)")
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()

    # Find the dataset YAML file
    dataset_dir = os.path.join("data", "datasets", args.dataset)
    yaml_path = os.path.join(dataset_dir, "yolo", "dataset.yaml")

    if not os.path.exists(yaml_path):
        yaml_path = os.path.join(dataset_dir, "dataset.yaml")
        if not os.path.exists(yaml_path):
            print(f"Error: Could not find dataset YAML file for {args.dataset} dataset")
            return

    # Create output directory
    output_dir = os.path.join("data", "models", args.dataset)
    os.makedirs(output_dir, exist_ok=True)

    # Load the model
    model = YOLO(f"yolov8{args.model_size}.pt")

    print(f"Training YOLOv8{args.model_size} model on {args.dataset} dataset")
    print(f"Dataset YAML: {yaml_path}")
    print(f"Output directory: {output_dir}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Image size: {args.img_size}")

    # Train the model
    model.train(
        data=yaml_path,
        epochs=args.epochs,
        batch=args.batch_size,
        imgsz=args.img_size,
        project=output_dir,
        name=f"yolov8{args.model_size}_{args.dataset}"
    )

    print("Training complete!")
    print(f"Model saved to {output_dir}/yolov8{args.model_size}_{args.dataset}")

    # Validate the model
    print("Validating the model...")
    metrics = model.val()

    print("Validation complete!")
    print(f"mAP50-95: {metrics.box.map}")
    print(f"mAP50: {metrics.box.map50}")
    print(f"mAP75: {metrics.box.map75}")

    # Export the model to ONNX format for deployment
    print("Exporting model to ONNX format...")
    model.export(format="onnx")

    print("Export complete!")
    print(f"ONNX model saved to {output_dir}/yolov8{args.model_size}_{args.dataset}/weights/best.onnx")

if __name__ == "__main__":
    main()
