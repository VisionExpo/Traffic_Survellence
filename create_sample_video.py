#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to create a sample video for testing the traffic surveillance system.
"""

import os
import cv2
import numpy as np
import argparse

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Create a sample video for testing")
    parser.add_argument("--output", default="data/output/sample_video.mp4",
                        help="Output video file (default: data/output/sample_video.mp4)")
    parser.add_argument("--duration", type=int, default=10,
                        help="Duration of the video in seconds (default: 10)")
    parser.add_argument("--fps", type=int, default=30,
                        help="Frames per second (default: 30)")
    parser.add_argument("--width", type=int, default=640,
                        help="Width of the video (default: 640)")
    parser.add_argument("--height", type=int, default=480,
                        help="Height of the video (default: 480)")
    return parser.parse_args()

def main():
    """Main function."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.output, fourcc, args.fps, (args.width, args.height))
    
    # Calculate total frames
    total_frames = args.duration * args.fps
    
    print(f"Creating sample video with {total_frames} frames...")
    
    # Create frames
    for i in range(total_frames):
        # Create a blank frame
        frame = np.zeros((args.height, args.width, 3), dtype=np.uint8)
        
        # Add a background (road)
        cv2.rectangle(frame, (0, 0), (args.width, args.height), (120, 120, 120), -1)
        
        # Add road markings
        # Center line
        cv2.line(frame, (0, args.height // 2), (args.width, args.height // 2), (255, 255, 255), 2)
        
        # Lane markings
        for x in range(0, args.width, 50):
            offset = (i % 60) - 30
            cv2.line(frame, (x + offset, args.height // 4), (x + offset + 30, args.height // 4), (255, 255, 255), 2)
            cv2.line(frame, (x + offset, 3 * args.height // 4), (x + offset + 30, 3 * args.height // 4), (255, 255, 255), 2)
        
        # Add moving vehicles
        # Car 1 (moving left to right)
        car1_x = (i * 5) % (args.width + 200) - 100
        cv2.rectangle(frame, (car1_x, args.height // 4 - 30), (car1_x + 80, args.height // 4 + 10), (0, 0, 255), -1)
        cv2.putText(frame, "Car", (car1_x, args.height // 4 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Car 2 (moving right to left)
        car2_x = args.width - ((i * 3) % (args.width + 200) - 100)
        cv2.rectangle(frame, (car2_x, 3 * args.height // 4 - 10), (car2_x + 80, 3 * args.height // 4 + 30), (255, 0, 0), -1)
        cv2.putText(frame, "Car", (car2_x, 3 * args.height // 4 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Motorcycle with rider (moving left to right)
        if i % 90 < 60:
            moto_x = ((i % 90) * 8) % (args.width + 200) - 100
            
            # Draw motorcycle
            cv2.rectangle(frame, (moto_x, args.height // 2 - 15), (moto_x + 60, args.height // 2 + 15), (0, 0, 200), -1)
            cv2.circle(frame, (moto_x + 10, args.height // 2 + 15), 10, (0, 0, 0), -1)
            cv2.circle(frame, (moto_x + 50, args.height // 2 + 15), 10, (0, 0, 0), -1)
            
            # Draw rider
            cv2.rectangle(frame, (moto_x + 20, args.height // 2 - 45), (moto_x + 40, args.height // 2 - 15), (0, 0, 255), -1)
            
            # Draw helmet or head
            if i % 180 < 90:  # With helmet
                cv2.circle(frame, (moto_x + 30, args.height // 2 - 55), 12, (0, 255, 0), -1)
                cv2.putText(frame, "Helmet", (moto_x + 45, args.height // 2 - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            else:  # Without helmet
                cv2.circle(frame, (moto_x + 30, args.height // 2 - 55), 10, (0, 200, 255), -1)
                cv2.putText(frame, "No Helmet", (moto_x + 45, args.height // 2 - 55), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            cv2.putText(frame, "Motorcycle", (moto_x, args.height // 2 + 30), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Add frame number
        cv2.putText(frame, f"Frame: {i}/{total_frames}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Write the frame
        out.write(frame)
        
        # Display progress
        if i % 30 == 0:
            print(f"Progress: {i}/{total_frames} frames ({i/total_frames*100:.1f}%)")
    
    # Release resources
    out.release()
    
    print(f"Sample video created: {args.output}")

if __name__ == "__main__":
    main()
