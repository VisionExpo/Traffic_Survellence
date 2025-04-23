#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Visualization module for traffic surveillance.
"""

import cv2
import numpy as np
import time
from datetime import datetime
import logging

logger = logging.getLogger(__name__)

class VisualizationEngine:
    """
    Class for visualizing traffic surveillance results.
    """
    
    def __init__(self, window_name="Traffic Surveillance", width=1280, height=720,
                 show_detections=True, show_tracks=True, show_speed=True,
                 show_license_plates=True, show_helmet_status=True):
        """
        Initialize the visualization engine.
        
        Args:
            window_name (str): Name of the display window
            width (int): Width of the display window
            height (int): Height of the display window
            show_detections (bool): Whether to show detection bounding boxes
            show_tracks (bool): Whether to show tracking information
            show_speed (bool): Whether to show speed information
            show_license_plates (bool): Whether to show license plate information
            show_helmet_status (bool): Whether to show helmet status
        """
        self.window_name = window_name
        self.width = width
        self.height = height
        self.show_detections = show_detections
        self.show_tracks = show_tracks
        self.show_speed = show_speed
        self.show_license_plates = show_license_plates
        self.show_helmet_status = show_helmet_status
        
        # Statistics
        self.vehicle_count = 0
        self.pedestrian_count = 0
        self.violation_count = 0
        self.start_time = time.time()
        self.frame_count = 0
        
        # Colors
        self.colors = {
            "vehicle": (0, 0, 255),  # Red
            "pedestrian": (0, 255, 0),  # Green
            "helmet": (0, 255, 255),  # Yellow
            "no_helmet": (0, 0, 255),  # Red
            "license_plate": (255, 0, 0),  # Blue
            "speeding": (0, 0, 255),  # Red
            "track": (255, 255, 255),  # White
            "text_bg": (0, 0, 0),  # Black
            "text_fg": (255, 255, 255)  # White
        }
        
        logger.info(f"Initializing visualization engine with window size {width}x{height}")
    
    def setup_window(self):
        """
        Set up the display window.
        """
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, self.width, self.height)
    
    def visualize(self, frame, detections=None, tracks=None, violations=None, 
                  timestamp=None, frame_count=None, fps=None):
        """
        Visualize detections, tracks, and violations on the frame.
        
        Args:
            frame (numpy.ndarray): Input frame
            detections (list): List of detection objects
            tracks (list): List of track objects
            violations (list): List of violation objects
            timestamp (float): Current timestamp
            frame_count (int): Current frame count
            fps (float): Current FPS
            
        Returns:
            numpy.ndarray: Visualized frame
        """
        # Create a copy of the frame to draw on
        output = frame.copy()
        
        # Update statistics
        if tracks:
            self.vehicle_count = sum(1 for t in tracks if t.class_name == "vehicle" and t.state == "confirmed")
            self.pedestrian_count = sum(1 for t in tracks if t.class_name == "pedestrian" and t.state == "confirmed")
        
        if violations:
            self.violation_count = len(violations)
        
        if frame_count is not None:
            self.frame_count = frame_count
        
        # Draw detections
        if self.show_detections and detections:
            output = self._draw_detections(output, detections)
        
        # Draw tracks
        if self.show_tracks and tracks:
            output = self._draw_tracks(output, tracks)
        
        # Draw violations
        if violations:
            output = self._draw_violations(output, violations)
        
        # Add information panel
        output = self._add_info_panel(output, fps)
        
        return output
    
    def _draw_detections(self, frame, detections):
        """
        Draw detection bounding boxes on the frame.
        
        Args:
            frame (numpy.ndarray): Input frame
            detections (list): List of detection objects
            
        Returns:
            numpy.ndarray: Frame with detection bounding boxes
        """
        for detection in detections:
            # Skip license plates and helmets if they're being handled by tracks
            if self.show_tracks and detection["class_name"] in ["license_plate", "helmet", "no_helmet"]:
                continue
            
            # Get bounding box coordinates
            x1, y1, x2, y2 = detection["bbox"]
            
            # Get color based on class
            color = self.colors.get(detection["class_name"], (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{detection['class_name']}: {detection['confidence']:.2f}"
            self._draw_label(frame, label, (x1, y1), color)
        
        return frame
    
    def _draw_tracks(self, frame, tracks):
        """
        Draw tracks on the frame.
        
        Args:
            frame (numpy.ndarray): Input frame
            tracks (list): List of track objects
            
        Returns:
            numpy.ndarray: Frame with tracks
        """
        for track in tracks:
            # Skip tentative tracks
            if track.state != "confirmed":
                continue
            
            # Get bounding box coordinates
            x1, y1, x2, y2 = track.bbox
            
            # Get color based on class
            color = self.colors.get(track.class_name, (255, 255, 255))
            
            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw track ID
            label = f"ID: {track.track_id}"
            self._draw_label(frame, label, (x1, y1), color)
            
            # Draw speed for vehicles
            if self.show_speed and track.class_name == "vehicle" and track.speed > 0:
                speed_label = f"{track.speed:.1f} km/h"
                self._draw_label(frame, speed_label, (x1, y2), color)
            
            # Draw license plate
            if self.show_license_plates and track.class_name == "vehicle" and track.license_plate:
                if "text" in track.license_plate and track.license_plate["text"]:
                    lp_label = f"LP: {track.license_plate['text']}"
                    self._draw_label(frame, lp_label, (x2 - 100, y2), self.colors["license_plate"])
            
            # Draw helmet status
            if self.show_helmet_status and track.class_name == "pedestrian" and track.helmet_status:
                helmet_label = f"Helmet: {track.helmet_status}"
                helmet_color = self.colors["helmet"] if track.helmet_status == "helmet" else self.colors["no_helmet"]
                self._draw_label(frame, helmet_label, (x1, y2), helmet_color)
            
            # Draw trajectory
            if len(track.positions) > 1:
                for i in range(1, len(track.positions)):
                    cv2.line(frame, track.positions[i-1], track.positions[i], color, 2)
        
        return frame
    
    def _draw_violations(self, frame, violations):
        """
        Draw violations on the frame.
        
        Args:
            frame (numpy.ndarray): Input frame
            violations (list): List of violation objects
            
        Returns:
            numpy.ndarray: Frame with violations
        """
        for violation in violations:
            # Get bounding box coordinates
            x1, y1, x2, y2 = violation["bbox"]
            
            # Get color based on violation type
            color = self.colors.get(violation["type"], (0, 0, 255))
            
            # Draw bounding box with thicker line
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
            
            # Draw violation type
            label = f"VIOLATION: {violation['type'].upper()}"
            self._draw_label(frame, label, (x1, y1 - 30), color, font_scale=0.7)
            
            # Draw additional information
            if violation["type"] == "speeding":
                speed_label = f"Speed: {violation['speed']:.1f} km/h"
                self._draw_label(frame, speed_label, (x1, y1 - 10), color, font_scale=0.6)
                
                if "license_plate" in violation and violation["license_plate"]:
                    lp_label = f"LP: {violation['license_plate']}"
                    self._draw_label(frame, lp_label, (x1, y2 + 20), color, font_scale=0.6)
        
        return frame
    
    def _draw_label(self, frame, text, position, color, font_scale=0.5, thickness=1):
        """
        Draw a label with background on the frame.
        
        Args:
            frame (numpy.ndarray): Input frame
            text (str): Text to draw
            position (tuple): Position (x, y)
            color (tuple): Color (B, G, R)
            font_scale (float): Font scale
            thickness (int): Line thickness
        """
        # Get text size
        (text_width, text_height), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        
        # Calculate background rectangle coordinates
        x, y = position
        bg_x1 = x
        bg_y1 = y - text_height - baseline
        bg_x2 = x + text_width
        bg_y2 = y
        
        # Draw background rectangle
        cv2.rectangle(frame, (bg_x1, bg_y1), (bg_x2, bg_y2), self.colors["text_bg"], -1)
        
        # Draw text
        cv2.putText(frame, text, (x, y - baseline), cv2.FONT_HERSHEY_SIMPLEX,
                    font_scale, color, thickness)
    
    def _add_info_panel(self, frame, fps=None):
        """
        Add an information panel to the frame with detection statistics.
        
        Args:
            frame (numpy.ndarray): Input frame
            fps (float): Current FPS
            
        Returns:
            numpy.ndarray: Frame with information panel
        """
        # Create a black panel at the bottom of the frame
        h, w = frame.shape[:2]
        panel_height = 80
        panel = np.zeros((panel_height, w, 3), dtype=np.uint8)
        
        # Add current date and time
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        cv2.putText(panel, current_time, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Add detection counts
        cv2.putText(panel, f"Vehicles: {self.vehicle_count}", (w // 5, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        cv2.putText(panel, f"Pedestrians: {self.pedestrian_count}", (w // 5, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Add violation count
        cv2.putText(panel, f"Violations: {self.violation_count}", (w // 2, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        # Add FPS counter
        if fps is None:
            elapsed_time = time.time() - self.start_time
            fps = self.frame_count / elapsed_time if elapsed_time > 0 else 0
        
        cv2.putText(panel, f"FPS: {fps:.2f}", (w // 2, 60),
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
