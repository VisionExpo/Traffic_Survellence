import cv2
import os
import time

def get_video_source(source_path=None):
    """
    Get a video source for processing.
    
    Args:
        source_path (str, optional): Path to a video file. If None, uses webcam.
        
    Returns:
        cv2.VideoCapture: Video capture object
    """
    if source_path and os.path.exists(source_path):
        return cv2.VideoCapture(source_path)
    else:
        # If no valid source path is provided, try to use the webcam
        print("No valid video source provided. Attempting to use webcam...")
        return cv2.VideoCapture(0)

def create_sample_video(output_path="sample_traffic.mp4", duration=10, fps=30):
    """
    Create a sample video with a moving rectangle to simulate traffic.
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

def limit_fps(start_time, target_fps=30):
    """
    Limit the processing rate to simulate real-time processing.
    
    Args:
        start_time (float): Start time of the frame processing
        target_fps (int): Target frames per second
        
    Returns:
        float: New start time for the next frame
    """
    elapsed = time.time() - start_time
    target_time = 1.0 / target_fps
    
    # If processing was faster than target, wait
    if elapsed < target_time:
        time.sleep(target_time - elapsed)
    
    return time.time()
