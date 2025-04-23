#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simple traffic surveillance script using a pre-trained YOLOv8 model.
"""

import os
import argparse
import cv2
import numpy as np
import time
from datetime import datetime
from ultralytics import YOLO

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Simple traffic surveillance")
    parser.add_argument("--source", default="0",
                        help="Path to video file or camera index (default: 0)")
    parser.add_argument("--conf", type=float, default=0.25,
                        help="Confidence threshold (default: 0.25)")
    parser.add_argument("--output-dir", default="data/output",
                        help="Output directory (default: data/output)")
    parser.add_argument("--save-video", action="store_true",
                        help="Save output video")
    parser.add_argument("--display", action="store_true", default=True,
                        help="Display output video")
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load YOLOv8 model
    print("Loading YOLOv8 model...")
    model = YOLO("yolov8n.pt")
    print("Model loaded")
    
    # Open video source
    print(f"Opening video source: {args.source}")
    if args.source.isdigit():
        cap = cv2.VideoCapture(int(args.source))
    else:
        cap = cv2.VideoCapture(args.source)
    
    if not cap.isOpened():
        print(f"Error: Could not open video source {args.source}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        fps = 30
    
    # Create video writer if needed
    if args.save_video:
        output_path = os.path.join(args.output_dir, f"output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        print(f"Saving output video to {output_path}")
    
    # Initialize variables for FPS calculation
    frame_count = 0
    start_time = time.time()
    fps_display = 0
    
    # Define classes of interest
    vehicle_classes = [2, 3, 5, 7]  # car, motorcycle, bus, truck
    person_class = 0  # person
    
    print("Starting traffic surveillance...")
    print("Press 'q' to quit")
    
    # Main loop
    while True:
        # Read a frame
        ret, frame = cap.read()
        
        if not ret:
            print("End of video stream")
            break
        
        # Increment frame count
        frame_count += 1
        
        # Calculate FPS
        if frame_count % 10 == 0:
            elapsed_time = time.time() - start_time
            fps_display = frame_count / elapsed_time
        
        # Create a copy of the frame for display
        display_frame = frame.copy()
        
        # Run YOLOv8 inference
        results = model(frame, conf=args.conf)
        
        # Process results
        vehicles_count = 0
        people_count = 0
        
        for result in results:
            boxes = result.boxes
            
            for box in boxes:
                # Get box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                
                # Get class and confidence
                cls = int(box.cls[0].item())
                conf = float(box.conf[0].item())
                
                # Get class name
                class_name = result.names[cls]
                
                # Check if it's a vehicle or person
                if cls in vehicle_classes:
                    color = (0, 0, 255)  # Red for vehicles
                    vehicles_count += 1
                elif cls == person_class:
                    color = (0, 255, 0)  # Green for people
                    people_count += 1
                else:
                    continue  # Skip other classes
                
                # Draw bounding box
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw label
                label = f"{class_name}: {conf:.2f}"
                cv2.putText(display_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Add information panel
        panel_height = 80
        panel = np.zeros((panel_height, width, 3), dtype=np.uint8)
        
        # Add current date and time
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(panel, current_time, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add detection counts
        cv2.putText(panel, f"Vehicles: {vehicles_count}", (width // 4, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(panel, f"People: {people_count}", (width // 4, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add FPS counter
        cv2.putText(panel, f"FPS: {fps_display:.2f}", (width // 2 + 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add "LIVE" indicator with blinking effect
        if int(time.time() * 2) % 2 == 0:  # Blink every 0.5 seconds
            cv2.putText(panel, "LIVE", (width - 100, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Combine the frame and panel
        display_frame_with_panel = np.vstack([display_frame, panel])
        
        # Display the frame
        if args.display:
            cv2.imshow("Traffic Surveillance", display_frame_with_panel)
        
        # Save the frame
        if args.save_video:
            out.write(display_frame)
        
        # Check for user input to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("User requested exit")
            break
    
    # Clean up resources
    cap.release()
    if args.save_video:
        out.release()
    cv2.destroyAllWindows()
    
    # Print statistics
    elapsed_time = time.time() - start_time
    print(f"Processed {frame_count} frames in {elapsed_time:.2f} seconds ({frame_count / elapsed_time:.2f} FPS)")
    print("Traffic surveillance stopped")

if __name__ == "__main__":
    main()
