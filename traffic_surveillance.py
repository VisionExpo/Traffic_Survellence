import cv2
import argparse
import time
import os
import numpy as np

from detector import ObjectDetector
from visualization import VisualizationEngine
from utils import get_video_source, limit_fps

def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Traffic Surveillance System")
    parser.add_argument("--source", "-s", type=str, default=None,
                        help="Path to video file (default: use webcam)")
    parser.add_argument("--confidence", "-c", type=float, default=0.5,
                        help="Minimum confidence threshold for detections (0-1)")
    parser.add_argument("--fps", "-f", type=int, default=30,
                        help="Target FPS for display (simulates real-time)")
    parser.add_argument("--width", "-W", type=int, default=1280,
                        help="Display window width")
    parser.add_argument("--height", "-H", type=int, default=720,
                        help="Display window height")
    
    return parser.parse_args()

def create_sample_video(output_path="sample_traffic.mp4", duration=10, fps=30):
    """
    Create a sample video with moving rectangles to simulate traffic.
    Useful for testing when no real traffic footage is available.
    
    Args:
        output_path (str): Path to save the sample video
        duration (int): Duration of the video in seconds
        fps (int): Frames per second
        
    Returns:
        str: Path to the created video file
    """
    # Define video parameters
    width, height = 640, 480
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Create frames with moving objects
    total_frames = duration * fps
    for i in range(total_frames):
        # Create a blank frame
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Draw background (road)
        cv2.rectangle(frame, (0, height//2-50), (width, height//2+50), (100, 100, 100), -1)
        
        # Draw lane markings
        for j in range(0, width, 30):
            offset = (i % 60) - 30
            cv2.rectangle(frame, (j + offset, height//2-2), (j + offset + 15, height//2+2), (255, 255, 255), -1)
        
        # Draw moving "vehicles"
        # Car 1
        car1_x = (i * 5) % (width + 100) - 50
        cv2.rectangle(frame, (car1_x, height//2-40), (car1_x + 60, height//2-10), (0, 0, 255), -1)
        
        # Car 2
        car2_x = (i * 3 + 200) % (width + 100) - 50
        cv2.rectangle(frame, (car2_x, height//2+10), (car2_x + 60, height//2+40), (255, 0, 0), -1)
        
        # Pedestrian
        if i % 90 < 45:
            ped_y = height//2 + 70 - (i % 45) * 3
            cv2.circle(frame, (width//4, ped_y), 10, (0, 255, 0), -1)
        
        # Write the frame to the video
        out.write(frame)
    
    # Release the video writer
    out.release()
    
    return output_path

def main():
    """
    Main function for the traffic surveillance system.
    """
    # Parse command line arguments
    args = parse_arguments()
    
    # Create a sample video if no source is provided
    if args.source is None:
        sample_path = "sample_traffic.mp4"
        if not os.path.exists(sample_path):
            print("Creating sample traffic video for demonstration...")
            create_sample_video(sample_path)
        args.source = sample_path
    
    # Initialize the video source
    cap = get_video_source(args.source)
    
    if not cap.isOpened():
        print("Error: Could not open video source.")
        return
    
    # Initialize the detector and visualization engine
    detector = ObjectDetector(confidence_threshold=args.confidence)
    visualizer = VisualizationEngine(
        window_name="Traffic Surveillance",
        width=args.width,
        height=args.height
    )
    
    # Set up the display window
    visualizer.setup_window()
    
    print("Starting traffic surveillance...")
    print("Press 'q' to quit")
    
    # Main processing loop
    while True:
        # Record the start time for FPS control
        start_time = time.time()
        
        # Read a frame from the video source
        ret, frame = cap.read()
        
        if not ret:
            print("End of video stream. Restarting...")
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # Restart the video
            continue
        
        # Detect objects in the frame
        processed_frame, detections = detector.detect(frame)
        
        # Add information panel
        display_frame = visualizer.add_info_panel(processed_frame, detections)
        
        # Display the frame
        visualizer.display_frame(display_frame)
        
        # Limit the FPS to simulate real-time processing
        start_time = limit_fps(start_time, args.fps)
        
        # Check for user input to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Clean up resources
    cap.release()
    visualizer.cleanup()
    print("Traffic surveillance stopped.")

if __name__ == "__main__":
    main()
