#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Video processing module for traffic surveillance.
"""

import os
import time
import logging
import cv2
import numpy as np
from datetime import datetime

logger = logging.getLogger(__name__)

class VideoProcessor:
    """
    Class for processing video streams for traffic surveillance.
    """
    
    def __init__(self, source=0, detector=None, tracker=None, lpr=None, visualizer=None, 
                 db_manager=None, speed_limit=60, calibration_factor=0.1, output_path=None):
        """
        Initialize the video processor.
        
        Args:
            source (str/int): Video source (file path or camera index)
            detector (ObjectDetector): Object detector instance
            tracker (ObjectTracker): Object tracker instance
            lpr (LicensePlateRecognizer): License plate recognizer instance
            visualizer (VisualizationEngine): Visualization engine instance
            db_manager (DatabaseManager): Database manager instance
            speed_limit (float): Speed limit in km/h
            calibration_factor (float): Pixels to meters conversion factor
            output_path (str): Path to save output video
        """
        self.source = source
        self.detector = detector
        self.tracker = tracker
        self.lpr = lpr
        self.visualizer = visualizer
        self.db_manager = db_manager
        self.speed_limit = speed_limit
        self.calibration_factor = calibration_factor
        self.output_path = output_path
        
        self.cap = None
        self.out = None
        self.fps = 30
        self.frame_count = 0
        self.start_time = time.time()
        self.processing = False
        
        logger.info(f"Initializing video processor with source: {source}")
    
    def _open_video_source(self):
        """
        Open the video source.
        
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            # If source is a string and exists as a file, open it
            if isinstance(self.source, str) and os.path.exists(self.source):
                self.cap = cv2.VideoCapture(self.source)
            else:
                # Try to open as camera index
                self.cap = cv2.VideoCapture(int(self.source) if str(self.source).isdigit() else self.source)
            
            if not self.cap.isOpened():
                logger.error(f"Could not open video source: {self.source}")
                return False
            
            # Get video properties
            self.fps = self.cap.get(cv2.CAP_PROP_FPS)
            if self.fps <= 0:
                self.fps = 30
                logger.warning(f"Invalid FPS value, using default: {self.fps}")
            
            # Update tracker FPS
            if self.tracker:
                self.tracker.set_fps(self.fps)
            
            # Initialize video writer if output path is provided
            if self.output_path:
                width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.out = cv2.VideoWriter(self.output_path, fourcc, self.fps, (width, height))
            
            logger.info(f"Video source opened successfully. FPS: {self.fps}")
            return True
        
        except Exception as e:
            logger.error(f"Error opening video source: {e}")
            return False
    
    def process_frame(self, frame):
        """
        Process a single frame.
        
        Args:
            frame (numpy.ndarray): Input frame
            
        Returns:
            tuple: (processed_frame, detections, tracks, violations)
        """
        # Increment frame count
        self.frame_count += 1
        
        # Calculate timestamp
        timestamp = self.frame_count / self.fps
        
        # Detect objects
        detections, annotated_frame = self.detector.detect(frame)
        
        # Separate detections by class
        vehicle_detections = [d for d in detections if d["class_name"] == "vehicle"]
        pedestrian_detections = [d for d in detections if d["class_name"] == "pedestrian"]
        
        # Detect license plates
        license_plate_detections = self.detector.detect_license_plates(frame, vehicle_detections)
        
        # Detect helmets
        helmet_detections = self.detector.detect_helmets(frame, pedestrian_detections)
        
        # Combine all detections
        all_detections = detections + license_plate_detections + helmet_detections
        
        # Update tracks
        tracks = self.tracker.update(all_detections, timestamp)
        
        # Calculate speeds
        self.tracker.calculate_speeds(self.calibration_factor)
        
        # Recognize license plates
        if self.lpr and license_plate_detections:
            lp_results = self.lpr.recognize(frame, license_plate_detections)
            
            # Update license plate text in tracks
            for lp_result in lp_results:
                if "vehicle_id" in lp_result:
                    for track in tracks:
                        if track.class_name == "vehicle" and id(track) == lp_result["vehicle_id"]:
                            if track.license_plate:
                                track.license_plate["text"] = lp_result["text"]
                                track.license_plate["confidence"] = lp_result["confidence"]
        
        # Detect violations
        violations = self._detect_violations(tracks)
        
        # Store data in database
        if self.db_manager:
            self._store_data(frame, tracks, violations, timestamp)
        
        # Visualize results
        if self.visualizer:
            processed_frame = self.visualizer.visualize(
                frame, 
                detections=all_detections, 
                tracks=tracks, 
                violations=violations,
                timestamp=timestamp,
                frame_count=self.frame_count,
                fps=self.calculate_fps()
            )
        else:
            processed_frame = annotated_frame
        
        return processed_frame, detections, tracks, violations
    
    def _detect_violations(self, tracks):
        """
        Detect traffic violations.
        
        Args:
            tracks (list): List of active tracks
            
        Returns:
            list: List of violation objects
        """
        violations = []
        
        for track in tracks:
            # Skip tentative tracks
            if track.state != "confirmed":
                continue
            
            # Check for speeding
            if track.class_name == "vehicle" and track.speed > self.speed_limit:
                violations.append({
                    "type": "speeding",
                    "track_id": track.track_id,
                    "speed": track.speed,
                    "license_plate": track.license_plate.get("text", "") if track.license_plate else "",
                    "confidence": track.license_plate.get("confidence", 0.0) if track.license_plate else 0.0,
                    "bbox": track.bbox
                })
            
            # Check for no helmet
            if track.class_name == "pedestrian" and track.helmet_status == "no_helmet":
                violations.append({
                    "type": "no_helmet",
                    "track_id": track.track_id,
                    "bbox": track.bbox
                })
        
        return violations
    
    def _store_data(self, frame, tracks, violations, timestamp):
        """
        Store data in the database.
        
        Args:
            frame (numpy.ndarray): Current frame
            tracks (list): List of active tracks
            violations (list): List of detected violations
            timestamp (float): Current timestamp
        """
        try:
            # Store violations
            for violation in violations:
                # Extract ROI for the violation
                x1, y1, x2, y2 = violation["bbox"]
                roi = frame[y1:y2, x1:x2]
                
                # Store in database
                self.db_manager.store_violation(
                    violation_type=violation["type"],
                    timestamp=timestamp,
                    frame_number=self.frame_count,
                    track_id=violation["track_id"],
                    license_plate=violation.get("license_plate", ""),
                    confidence=violation.get("confidence", 0.0),
                    speed=violation.get("speed", 0.0),
                    image=roi
                )
        
        except Exception as e:
            logger.error(f"Error storing data: {e}")
    
    def calculate_fps(self):
        """
        Calculate the actual processing FPS.
        
        Returns:
            float: Frames per second
        """
        elapsed_time = time.time() - self.start_time
        if elapsed_time > 0:
            return self.frame_count / elapsed_time
        return 0
    
    def process(self):
        """
        Process the video stream.
        """
        if not self._open_video_source():
            return
        
        self.processing = True
        self.start_time = time.time()
        self.frame_count = 0
        
        # Set up visualization window
        if self.visualizer:
            self.visualizer.setup_window()
        
        logger.info("Starting video processing")
        
        while self.processing:
            # Read a frame
            ret, frame = self.cap.read()
            
            if not ret:
                logger.info("End of video stream")
                break
            
            # Process the frame
            processed_frame, _, _, _ = self.process_frame(frame)
            
            # Write to output video
            if self.out:
                self.out.write(processed_frame)
            
            # Display the frame
            if self.visualizer:
                self.visualizer.display_frame(processed_frame)
            
            # Check for user input to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                logger.info("User requested exit")
                break
        
        # Clean up resources
        self._cleanup()
    
    def _cleanup(self):
        """
        Clean up resources.
        """
        self.processing = False
        
        if self.cap:
            self.cap.release()
        
        if self.out:
            self.out.release()
        
        if self.visualizer:
            self.visualizer.cleanup()
        
        logger.info("Video processing stopped")
    
    def get_frame(self):
        """
        Get the current processed frame for the web dashboard.
        
        Returns:
            numpy.ndarray: Processed frame
        """
        if not self.cap or not self.cap.isOpened():
            if not self._open_video_source():
                return None
        
        ret, frame = self.cap.read()
        
        if not ret:
            # Restart the video if it's a file
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = self.cap.read()
            if not ret:
                return None
        
        # Process the frame
        processed_frame, _, _, _ = self.process_frame(frame)
        
        return processed_frame
