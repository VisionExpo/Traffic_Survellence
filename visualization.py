import cv2
import numpy as np
import time
from datetime import datetime

class VisualizationEngine:
    """
    Class for visualizing traffic surveillance results.
    """
    
    def __init__(self, window_name="Traffic Surveillance", width=1280, height=720):
        """
        Initialize the visualization engine.
        
        Args:
            window_name (str): Name of the display window
            width (int): Width of the display window
            height (int): Height of the display window
        """
        self.window_name = window_name
        self.width = width
        self.height = height
        self.vehicle_count = 0
        self.pedestrian_count = 0
        self.start_time = time.time()
        self.frame_count = 0
        
    def setup_window(self):
        """
        Set up the display window.
        """
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.width, self.height)
        
    def add_info_panel(self, frame, detections):
        """
        Add an information panel to the frame with detection statistics.
        
        Args:
            frame (numpy.ndarray): Input frame
            detections (dict): Dictionary with detection counts
            
        Returns:
            numpy.ndarray: Frame with information panel
        """
        # Update counts
        self.vehicle_count = detections["vehicles"]
        self.pedestrian_count = detections["pedestrians"]
        self.frame_count += 1
        
        # Calculate FPS
        elapsed_time = time.time() - self.start_time
        fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
        
        # Create a black panel at the bottom of the frame
        h, w = frame.shape[:2]
        panel_height = 80
        panel = np.zeros((panel_height, w, 3), dtype=np.uint8)
        
        # Add current date and time
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(panel, current_time, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add detection counts
        cv2.putText(panel, f"Vehicles: {self.vehicle_count}", (w // 4, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(panel, f"Pedestrians: {self.pedestrian_count}", (w // 4, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add FPS counter
        cv2.putText(panel, f"FPS: {fps:.2f}", (w // 2 + 100, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add "LIVE" indicator with blinking effect
        if int(time.time() * 2) % 2 == 0:  # Blink every 0.5 seconds
            cv2.putText(panel, "LIVE", (w - 100, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Combine the frame and panel
        result = np.vstack([frame, panel])
        
        return result
    
    def display_frame(self, frame):
        """
        Display a frame in the window.
        
        Args:
            frame (numpy.ndarray): Frame to display
        """
        cv2.imshow(self.window_name, frame)
    
    def cleanup(self):
        """
        Clean up resources.
        """
        cv2.destroyAllWindows()
