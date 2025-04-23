import cv2
import argparse
import time
import os
import numpy as np
import threading
from flask import Flask, Response, render_template
import base64
from io import BytesIO
from PIL import Image

from detector import ObjectDetector
from visualization import VisualizationEngine
from utils import get_video_source, limit_fps

# Global variables
output_frame = None
lock = threading.Lock()
detector = None
visualizer = None
cap = None
processing_active = False

# Initialize Flask app
app = Flask(__name__)

def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(description="Traffic Surveillance Web App")
    parser.add_argument("--source", "-s", type=str, default=None,
                        help="Path to video file (default: use webcam)")
    parser.add_argument("--confidence", "-c", type=float, default=0.5,
                        help="Minimum confidence threshold for detections (0-1)")
    parser.add_argument("--fps", "-f", type=int, default=15,
                        help="Target FPS for processing")
    parser.add_argument("--port", "-p", type=int, default=5000,
                        help="Port for the web server")
    
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

def process_frames():
    """
    Process video frames in a separate thread.
    """
    global output_frame, lock, detector, visualizer, cap, processing_active
    
    # Main processing loop
    while processing_active:
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
        
        # Update the output frame
        with lock:
            output_frame = display_frame.copy()
        
        # Limit the FPS to simulate real-time processing
        start_time = limit_fps(start_time, args.fps)

def generate():
    """
    Generate MJPEG stream from the processed frames.
    """
    global output_frame, lock
    
    while True:
        # Wait until a frame is available
        if output_frame is None:
            time.sleep(0.1)
            continue
        
        # Encode the frame as JPEG
        with lock:
            (flag, encoded_image) = cv2.imencode(".jpg", output_frame)
            
            if not flag:
                continue
        
        # Yield the frame in the MJPEG format
        yield(b'--frame\r\n' b'Content-Type: image/jpeg\r\n\r\n' + 
              bytearray(encoded_image) + b'\r\n')
        
        # Sleep to control the frame rate
        time.sleep(0.05)

@app.route("/")
def index():
    """
    Render the home page.
    """
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    """
    Return the video feed as a streaming response.
    """
    return Response(generate(),
                    mimetype="multipart/x-mixed-replace; boundary=frame")

def create_templates_directory():
    """
    Create the templates directory and index.html file.
    """
    os.makedirs("templates", exist_ok=True)
    
    index_html = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Traffic Surveillance System</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                margin: 0;
                padding: 0;
                background-color: #f0f0f0;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }
            h1 {
                color: #333;
                text-align: center;
            }
            .video-container {
                display: flex;
                justify-content: center;
                margin: 20px 0;
            }
            .video-feed {
                border: 5px solid #333;
                border-radius: 5px;
                max-width: 100%;
                height: auto;
            }
            .status {
                background-color: #333;
                color: white;
                padding: 10px;
                border-radius: 5px;
                margin-top: 20px;
                text-align: center;
            }
            .live-indicator {
                display: inline-block;
                width: 10px;
                height: 10px;
                background-color: red;
                border-radius: 50%;
                margin-right: 5px;
                animation: blink 1s infinite;
            }
            @keyframes blink {
                0% { opacity: 1; }
                50% { opacity: 0; }
                100% { opacity: 1; }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Traffic Surveillance System</h1>
            <div class="video-container">
                <img src="{{ url_for('video_feed') }}" class="video-feed">
            </div>
            <div class="status">
                <span class="live-indicator"></span> LIVE FEED
            </div>
        </div>
    </body>
    </html>
    """
    
    with open("templates/index.html", "w") as f:
        f.write(index_html)

if __name__ == "__main__":
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
        exit()
    
    # Initialize the detector and visualization engine
    detector = ObjectDetector(confidence_threshold=args.confidence)
    visualizer = VisualizationEngine(window_name="Traffic Surveillance")
    
    # Create the templates directory and index.html file
    create_templates_directory()
    
    # Start the frame processing thread
    processing_active = True
    t = threading.Thread(target=process_frames)
    t.daemon = True
    t.start()
    
    # Start the Flask app
    print(f"Starting web server on http://localhost:{args.port}")
    app.run(host="0.0.0.0", port=args.port, debug=False, threaded=True)
    
    # Clean up resources when the app is closed
    processing_active = False
    cap.release()
    print("Traffic surveillance stopped.")
