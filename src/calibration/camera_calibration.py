#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script for camera calibration to improve speed estimation accuracy.
"""

import os
import argparse
import cv2
import numpy as np
import yaml
import matplotlib.pyplot as plt
from datetime import datetime

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Camera calibration for traffic surveillance")
    parser.add_argument("--input", required=True,
                        help="Input video file or camera index")
    parser.add_argument("--chessboard-size", default="9x6",
                        help="Chessboard size (width x height) (default: 9x6)")
    parser.add_argument("--square-size", type=float, default=0.025,
                        help="Chessboard square size in meters (default: 0.025)")
    parser.add_argument("--frames", type=int, default=20,
                        help="Number of frames to use for calibration (default: 20)")
    parser.add_argument("--output-dir", default="data/calibration",
                        help="Output directory for calibration results (default: data/calibration)")
    parser.add_argument("--reference-distance", type=float, default=10.0,
                        help="Reference distance in meters for speed calibration (default: 10.0)")
    parser.add_argument("--reference-points", action="store_true",
                        help="Select reference points for perspective transformation")
    return parser.parse_args()

def capture_chessboard_frames(input_source, chessboard_size, num_frames):
    """Capture frames with a chessboard pattern for calibration."""
    print(f"Capturing {num_frames} frames with chessboard pattern...")
    
    # Parse chessboard size
    width, height = map(int, chessboard_size.split("x"))
    
    # Open video source
    if input_source.isdigit():
        cap = cv2.VideoCapture(int(input_source))
    else:
        cap = cv2.VideoCapture(input_source)
    
    if not cap.isOpened():
        print(f"Error: Could not open video source {input_source}")
        return None, None
    
    # Prepare object points (3D points in real world space)
    objp = np.zeros((height * width, 3), np.float32)
    objp[:, :2] = np.mgrid[0:width, 0:height].T.reshape(-1, 2)
    
    # Arrays to store object points and image points
    objpoints = []  # 3D points in real world space
    imgpoints = []  # 2D points in image plane
    
    # Capture frames
    frames_captured = 0
    while frames_captured < num_frames:
        ret, frame = cap.read()
        if not ret:
            break
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Find chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (width, height), None)
        
        # If found, add object points and image points
        if ret:
            objpoints.append(objp)
            imgpoints.append(corners)
            
            # Draw and display the corners
            cv2.drawChessboardCorners(frame, (width, height), corners, ret)
            cv2.putText(frame, f"Frames: {frames_captured+1}/{num_frames}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            frames_captured += 1
        else:
            cv2.putText(frame, "Chessboard not found", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        # Display the frame
        cv2.imshow("Chessboard Detection", frame)
        
        # Wait for key press
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    
    if frames_captured < num_frames:
        print(f"Warning: Only captured {frames_captured} frames out of {num_frames}")
    
    return objpoints, imgpoints, gray.shape[::-1]

def calibrate_camera(objpoints, imgpoints, image_size, square_size):
    """Calibrate the camera using chessboard images."""
    print("Calibrating camera...")
    
    # Scale object points by square size
    for objp in objpoints:
        objp *= square_size
    
    # Calibrate camera
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, image_size, None, None)
    
    # Calculate reprojection error
    mean_error = 0
    for i in range(len(objpoints)):
        imgpoints2, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], mtx, dist)
        error = cv2.norm(imgpoints[i], imgpoints2, cv2.NORM_L2) / len(imgpoints2)
        mean_error += error
    
    mean_error /= len(objpoints)
    print(f"Calibration complete. Mean reprojection error: {mean_error}")
    
    return mtx, dist, mean_error

def select_reference_points(input_source, output_dir):
    """Select reference points for perspective transformation."""
    print("Select 4 points on the ground plane for perspective transformation")
    print("The points should form a rectangle in the real world")
    print("Press 'r' to reset, 'c' to confirm, 'q' to quit")
    
    # Open video source
    if input_source.isdigit():
        cap = cv2.VideoCapture(int(input_source))
    else:
        cap = cv2.VideoCapture(input_source)
    
    if not cap.isOpened():
        print(f"Error: Could not open video source {input_source}")
        return None
    
    # Read a frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from video source")
        cap.release()
        return None
    
    # Create a copy of the frame
    img = frame.copy()
    
    # List to store selected points
    points = []
    
    # Mouse callback function
    def mouse_callback(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if len(points) < 4:
                points.append((x, y))
                # Draw the point
                cv2.circle(img, (x, y), 5, (0, 255, 0), -1)
                # Draw the point number
                cv2.putText(img, str(len(points)), (x+10, y+10),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                # Draw lines between points
                if len(points) > 1:
                    cv2.line(img, points[-2], points[-1], (0, 255, 0), 2)
                    if len(points) == 4:
                        cv2.line(img, points[0], points[3], (0, 255, 0), 2)
                cv2.imshow("Select Reference Points", img)
    
    # Create window and set mouse callback
    cv2.namedWindow("Select Reference Points")
    cv2.setMouseCallback("Select Reference Points", mouse_callback)
    
    # Display the frame
    cv2.imshow("Select Reference Points", img)
    
    # Wait for key press
    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord("r"):
            # Reset points
            points = []
            img = frame.copy()
            cv2.imshow("Select Reference Points", img)
        elif key == ord("c") and len(points) == 4:
            # Confirm points
            break
        elif key == ord("q"):
            # Quit
            points = []
            break
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    
    if len(points) != 4:
        print("Error: Did not select 4 points")
        return None
    
    # Save the reference points
    reference_points = np.array(points, dtype=np.float32)
    
    # Save the reference image with points
    for i, point in enumerate(points):
        cv2.circle(frame, point, 5, (0, 255, 0), -1)
        cv2.putText(frame, str(i+1), (point[0]+10, point[1]+10),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    # Draw lines between points
    for i in range(4):
        cv2.line(frame, points[i], points[(i+1)%4], (0, 255, 0), 2)
    
    # Save the image
    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(os.path.join(output_dir, "reference_points.jpg"), frame)
    
    print("Reference points saved")
    return reference_points

def calculate_perspective_transform(reference_points, reference_distance):
    """Calculate perspective transformation matrix."""
    print("Calculating perspective transformation matrix...")
    
    # Define the destination points (rectangle in bird's eye view)
    # We'll use a square with side length equal to the reference distance
    dst_points = np.array([
        [0, 0],
        [reference_distance, 0],
        [reference_distance, reference_distance],
        [0, reference_distance]
    ], dtype=np.float32)
    
    # Calculate the perspective transformation matrix
    M = cv2.getPerspectiveTransform(reference_points, dst_points)
    
    print("Perspective transformation matrix calculated")
    return M

def test_calibration(input_source, camera_matrix, dist_coeffs, perspective_matrix, output_dir):
    """Test the calibration by undistorting and transforming a frame."""
    print("Testing calibration...")
    
    # Open video source
    if input_source.isdigit():
        cap = cv2.VideoCapture(int(input_source))
    else:
        cap = cv2.VideoCapture(input_source)
    
    if not cap.isOpened():
        print(f"Error: Could not open video source {input_source}")
        return
    
    # Read a frame
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame from video source")
        cap.release()
        return
    
    # Undistort the frame
    h, w = frame.shape[:2]
    newcameramtx, roi = cv2.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    undistorted = cv2.undistort(frame, camera_matrix, dist_coeffs, None, newcameramtx)
    
    # Apply perspective transformation
    warped = cv2.warpPerspective(undistorted, perspective_matrix, (w, h))
    
    # Save the results
    os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(os.path.join(output_dir, "original.jpg"), frame)
    cv2.imwrite(os.path.join(output_dir, "undistorted.jpg"), undistorted)
    cv2.imwrite(os.path.join(output_dir, "warped.jpg"), warped)
    
    # Create a side-by-side comparison
    comparison = np.hstack((frame, undistorted, warped))
    cv2.imwrite(os.path.join(output_dir, "comparison.jpg"), comparison)
    
    # Display the results
    cv2.imshow("Original", frame)
    cv2.imshow("Undistorted", undistorted)
    cv2.imshow("Warped (Bird's Eye View)", warped)
    cv2.waitKey(0)
    
    # Release resources
    cap.release()
    cv2.destroyAllWindows()
    
    print("Calibration test complete")

def save_calibration_results(camera_matrix, dist_coeffs, perspective_matrix, reference_distance, output_dir):
    """Save calibration results to a YAML file."""
    print("Saving calibration results...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create calibration data
    calibration_data = {
        "camera_matrix": camera_matrix.tolist(),
        "dist_coeffs": dist_coeffs.tolist(),
        "perspective_matrix": perspective_matrix.tolist(),
        "reference_distance": float(reference_distance),
        "calibration_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    }
    
    # Save to YAML file
    yaml_path = os.path.join(output_dir, "calibration.yaml")
    with open(yaml_path, "w") as f:
        yaml.dump(calibration_data, f, default_flow_style=False)
    
    print(f"Calibration results saved to {yaml_path}")

def main():
    """Main function."""
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Capture chessboard frames for camera calibration
    objpoints, imgpoints, image_size = capture_chessboard_frames(
        args.input, args.chessboard_size, args.frames)
    
    if objpoints is None or imgpoints is None:
        print("Error: Could not capture enough chessboard frames")
        return
    
    # Calibrate camera
    camera_matrix, dist_coeffs, _ = calibrate_camera(
        objpoints, imgpoints, image_size, args.square_size)
    
    # Select reference points for perspective transformation
    reference_points = None
    if args.reference_points:
        reference_points = select_reference_points(args.input, args.output_dir)
    
    # Calculate perspective transformation matrix
    perspective_matrix = None
    if reference_points is not None:
        perspective_matrix = calculate_perspective_transform(
            reference_points, args.reference_distance)
    
    # Test calibration
    if perspective_matrix is not None:
        test_calibration(args.input, camera_matrix, dist_coeffs, perspective_matrix, args.output_dir)
    
    # Save calibration results
    save_calibration_results(
        camera_matrix, dist_coeffs, perspective_matrix or np.eye(3),
        args.reference_distance, args.output_dir)
    
    print("Camera calibration complete!")

if __name__ == "__main__":
    main()
