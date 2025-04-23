#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to test a trained YOLOv8 model on an image or video.
"""

import os
import argparse
import cv2
import numpy as np
from ultralytics import YOLO
import matplotlib.pyplot as plt

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test a trained YOLOv8 model")
    parser.add_argument("--model", required=True,
                        help="Path to the trained model")
    parser.add_argument("--source", required=True,
                        help="Path to the image or video file")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold (default: 0.25)")
    parser.add_argument("--output-dir", default="data/output",
                        help="Output directory (default: data/output)")
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()
    
    # Check if the model file exists
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        return
    
    # Check if the source file exists
    if not os.path.exists(args.source):
        print(f"Error: Source file not found: {args.source}")
        return
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load the model
    model = YOLO(args.model)
    
    print(f"Testing model: {args.model}")
    print(f"Source: {args.source}")
    print(f"Confidence threshold: {args.conf}")
    
    # Run inference
    results = model(args.source, conf=args.conf)
    
    # Process results
    for i, result in enumerate(results):
        # Get the original image
        orig_img = result.orig_img
        
        # Get the annotated image
        annotated_img = result.plot()
        
        # Save the annotated image
        output_path = os.path.join(args.output_dir, f"result_{i}.jpg")
        cv2.imwrite(output_path, annotated_img)
        
        # Display the results
        plt.figure(figsize=(12, 6))
        
        plt.subplot(1, 2, 1)
        plt.imshow(cv2.cvtColor(orig_img, cv2.COLOR_BGR2RGB))
        plt.title("Original Image")
        plt.axis("off")
        
        plt.subplot(1, 2, 2)
        plt.imshow(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB))
        plt.title("Detected Objects")
        plt.axis("off")
        
        plt.tight_layout()
        plt.savefig(os.path.join(args.output_dir, f"comparison_{i}.jpg"))
        plt.show()
        
        # Print detection results
        print(f"\nDetections in image {i}:")
        for j, (box, conf, cls) in enumerate(zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls)):
            class_name = result.names[int(cls)]
            print(f"  {j+1}. {class_name}: {conf:.2f}")
    
    print(f"\nResults saved to {args.output_dir}")

if __name__ == "__main__":
    main()
