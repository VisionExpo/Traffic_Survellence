#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to run the traffic surveillance system with our trained models.
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
    parser = argparse.ArgumentParser(description="Run traffic surveillance system")
    parser.add_argument("--source", default="0",
                        help="Path to video file or camera index (default: 0)")
    parser.add_argument("--helmet-model", default="data/models/helmet/yolov8n_helmet3/weights/best.pt",
                        help="Path to helmet detection model")
    parser.add_argument("--vehicle-model", default=None,
                        help="Path to vehicle detection model (default: use YOLOv8n)")
    parser.add_argument("--license-plate-model", default=None,
                        help="Path to license plate detection model (default: use YOLOv8n)")
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
    
    # Load models
    print("Loading models...")
    
    # Load helmet detection model
    if args.helmet_model and os.path.exists(args.helmet_model):
        helmet_model = YOLO(args.helmet_model)
        print(f"Loaded helmet detection model: {args.helmet_model}")
    else:
        helmet_model = YOLO("yolov8n.pt")
        print("Using default YOLOv8n model for helmet detection")
    
    # Load vehicle detection model
    if args.vehicle_model and os.path.exists(args.vehicle_model):
        vehicle_model = YOLO(args.vehicle_model)
        print(f"Loaded vehicle detection model: {args.vehicle_model}")
    else:
        vehicle_model = YOLO("yolov8n.pt")
        print("Using default YOLOv8n model for vehicle detection")
    
    # Load license plate detection model
    if args.license_plate_model and os.path.exists(args.license_plate_model):
        license_plate_model = YOLO(args.license_plate_model)
        print(f"Loaded license plate detection model: {args.license_plate_model}")
    else:
        license_plate_model = None
        print("License plate detection disabled")
    
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
        
        # Detect vehicles
        vehicle_results = vehicle_model(frame, conf=args.conf, classes=[2, 3, 5, 7])  # car, motorcycle, bus, truck
        
        # Process vehicle detections
        vehicles = []
        for result in vehicle_results:
            for i, (box, conf, cls) in enumerate(zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls)):
                x1, y1, x2, y2 = box.cpu().numpy().astype(int)
                class_name = result.names[int(cls)]
                
                # Add to vehicles list
                vehicles.append({
                    "bbox": [x1, y1, x2, y2],
                    "conf": float(conf),
                    "class": class_name
                })
                
                # Draw bounding box
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                
                # Draw label
                label = f"{class_name}: {conf:.2f}"
                cv2.putText(display_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        
        # Detect helmets and people
        helmet_results = helmet_model(frame, conf=args.conf)
        
        # Process helmet detections
        helmets = []
        people = []
        for result in helmet_results:
            for i, (box, conf, cls) in enumerate(zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls)):
                x1, y1, x2, y2 = box.cpu().numpy().astype(int)
                class_name = result.names[int(cls)]
                
                # Add to appropriate list
                if class_name == "helmet":
                    helmets.append({
                        "bbox": [x1, y1, x2, y2],
                        "conf": float(conf)
                    })
                    color = (0, 255, 0)  # Green for helmet
                elif class_name == "person" or class_name == "head":
                    people.append({
                        "bbox": [x1, y1, x2, y2],
                        "conf": float(conf),
                        "has_helmet": False
                    })
                    color = (0, 0, 255)  # Red for person without helmet
                else:
                    continue
                
                # Draw bounding box
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw label
                label = f"{class_name}: {conf:.2f}"
                cv2.putText(display_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Check if people are wearing helmets
        for person in people:
            px1, py1, px2, py2 = person["bbox"]
            person_area = (px2 - px1) * (py2 - py1)
            
            for helmet in helmets:
                hx1, hy1, hx2, hy2 = helmet["bbox"]
                
                # Check if helmet is above the person
                if (hx1 >= px1 and hx2 <= px2 and hy1 >= py1 and hy2 <= py2) or \
                   (hx1 <= px1 and hx2 >= px2 and hy1 <= py1 and hy2 >= py2) or \
                   (hx1 <= px2 and hx2 >= px1 and hy1 <= py2 and hy2 >= py1):
                    # Calculate intersection area
                    ix1 = max(px1, hx1)
                    iy1 = max(py1, hy1)
                    ix2 = min(px2, hx2)
                    iy2 = min(py2, hy2)
                    
                    if ix2 > ix1 and iy2 > iy1:
                        intersection_area = (ix2 - ix1) * (iy2 - iy1)
                        
                        # If intersection is significant, mark person as having helmet
                        if intersection_area / person_area > 0.1:
                            person["has_helmet"] = True
                            break
        
        # Detect license plates if model is available
        if license_plate_model:
            license_plate_results = license_plate_model(frame, conf=args.conf)
            
            # Process license plate detections
            for result in license_plate_results:
                for i, (box, conf, cls) in enumerate(zip(result.boxes.xyxy, result.boxes.conf, result.boxes.cls)):
                    x1, y1, x2, y2 = box.cpu().numpy().astype(int)
                    
                    # Draw bounding box
                    cv2.rectangle(display_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    
                    # Draw label
                    label = f"License Plate: {conf:.2f}"
                    cv2.putText(display_frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
        
        # Add information panel
        panel_height = 80
        panel = np.zeros((panel_height, width, 3), dtype=np.uint8)
        
        # Add current date and time
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(panel, current_time, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add detection counts
        cv2.putText(panel, f"Vehicles: {len(vehicles)}", (width // 4, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(panel, f"People: {len(people)}", (width // 4, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
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
