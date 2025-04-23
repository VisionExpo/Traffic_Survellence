#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Object tracking module for traffic surveillance.
Supports tracking of vehicles, pedestrians, and other objects.
"""

import logging
import numpy as np
import cv2

logger = logging.getLogger(__name__)

class Track:
    """Class to represent a tracked object."""
    
    def __init__(self, track_id, bbox, class_name, confidence):
        """
        Initialize a track.
        
        Args:
            track_id (int): Unique ID for the track
            bbox (list): Bounding box coordinates [x1, y1, x2, y2]
            class_name (str): Class name of the object
            confidence (float): Detection confidence
        """
        self.track_id = track_id
        self.bbox = bbox
        self.class_name = class_name
        self.confidence = confidence
        self.age = 0
        self.hits = 1
        self.time_since_update = 0
        self.state = "tentative"  # tentative, confirmed, deleted
        
        # For speed estimation
        self.positions = [self._get_center()]
        self.timestamps = [0]  # Will be set by the tracker
        self.speed = 0.0
        
        # For license plate and helmet detection
        self.license_plate = None
        self.helmet_status = None
    
    def _get_center(self):
        """Get the center point of the bounding box."""
        x1, y1, x2, y2 = self.bbox
        return ((x1 + x2) // 2, (y1 + y2) // 2)
    
    def update(self, bbox, confidence, timestamp):
        """
        Update the track with a new detection.
        
        Args:
            bbox (list): New bounding box coordinates
            confidence (float): New detection confidence
            timestamp (float): Current timestamp
        """
        self.bbox = bbox
        self.confidence = confidence
        self.hits += 1
        self.time_since_update = 0
        
        # Update position history
        center = self._get_center()
        self.positions.append(center)
        self.timestamps.append(timestamp)
        
        # Limit history length
        if len(self.positions) > 30:
            self.positions.pop(0)
            self.timestamps.pop(0)
        
        # Update state
        if self.hits >= 3:
            self.state = "confirmed"
    
    def predict(self):
        """
        Predict the next position of the track.
        Simple linear motion model.
        """
        if len(self.positions) < 2:
            return self.bbox
        
        # Calculate velocity
        dx = self.positions[-1][0] - self.positions[-2][0]
        dy = self.positions[-1][1] - self.positions[-2][1]
        
        # Predict new center
        new_center_x = self.positions[-1][0] + dx
        new_center_y = self.positions[-1][1] + dy
        
        # Calculate new bbox
        width = self.bbox[2] - self.bbox[0]
        height = self.bbox[3] - self.bbox[1]
        
        x1 = int(new_center_x - width // 2)
        y1 = int(new_center_y - height // 2)
        x2 = int(new_center_x + width // 2)
        y2 = int(new_center_y + height // 2)
        
        self.bbox = [x1, y1, x2, y2]
        self.time_since_update += 1
        self.age += 1
        
        return self.bbox
    
    def calculate_speed(self, calibration_factor, fps):
        """
        Calculate the speed of the tracked object.
        
        Args:
            calibration_factor (float): Pixels to meters conversion factor
            fps (float): Frames per second
            
        Returns:
            float: Speed in km/h
        """
        if len(self.positions) < 2 or len(self.timestamps) < 2:
            return 0.0
        
        # Calculate distance traveled in pixels
        distances = []
        for i in range(1, len(self.positions)):
            x1, y1 = self.positions[i-1]
            x2, y2 = self.positions[i]
            distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
            distances.append(distance)
        
        # Calculate time elapsed in seconds
        time_elapsed = (self.timestamps[-1] - self.timestamps[0]) / fps
        if time_elapsed <= 0:
            return 0.0
        
        # Calculate average distance
        avg_distance = np.mean(distances)
        
        # Convert pixels to meters
        distance_meters = avg_distance * calibration_factor
        
        # Calculate speed in m/s
        speed_ms = distance_meters / time_elapsed
        
        # Convert to km/h
        speed_kmh = speed_ms * 3.6
        
        self.speed = speed_kmh
        return speed_kmh


class ObjectTracker:
    """
    Class for tracking objects in traffic surveillance footage.
    Implements a simple tracking algorithm based on IoU.
    """
    
    def __init__(self, algorithm="iou", max_age=30, min_hits=3, iou_threshold=0.3):
        """
        Initialize the tracker.
        
        Args:
            algorithm (str): Tracking algorithm to use
            max_age (int): Maximum number of frames to keep a track alive without matching
            min_hits (int): Minimum number of hits to confirm a track
            iou_threshold (float): IoU threshold for matching
        """
        self.algorithm = algorithm
        self.max_age = max_age
        self.min_hits = min_hits
        self.iou_threshold = iou_threshold
        self.tracks = []
        self.next_id = 0
        self.frame_count = 0
        self.fps = 30  # Default FPS, will be updated
        
        logger.info(f"Initializing {algorithm} tracker with max_age={max_age}, "
                   f"min_hits={min_hits}, iou_threshold={iou_threshold}")
    
    def _calculate_iou(self, bbox1, bbox2):
        """
        Calculate IoU between two bounding boxes.
        
        Args:
            bbox1 (list): First bounding box [x1, y1, x2, y2]
            bbox2 (list): Second bounding box [x1, y1, x2, y2]
            
        Returns:
            float: IoU value
        """
        # Determine the coordinates of the intersection rectangle
        x_left = max(bbox1[0], bbox2[0])
        y_top = max(bbox1[1], bbox2[1])
        x_right = min(bbox1[2], bbox2[2])
        y_bottom = min(bbox1[3], bbox2[3])
        
        # If there is no intersection, return 0
        if x_right < x_left or y_bottom < y_top:
            return 0.0
        
        # Calculate area of intersection
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate area of both bounding boxes
        bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        
        # Calculate IoU
        iou = intersection_area / float(bbox1_area + bbox2_area - intersection_area)
        
        return iou
    
    def update(self, detections, timestamp=None):
        """
        Update tracks with new detections.
        
        Args:
            detections (list): List of detection objects
            timestamp (float): Current timestamp
            
        Returns:
            list: List of active tracks
        """
        self.frame_count += 1
        
        # Set timestamp if not provided
        if timestamp is None:
            timestamp = self.frame_count / self.fps
        
        # If no tracks exist, create new tracks for all detections
        if len(self.tracks) == 0:
            for detection in detections:
                if detection["class_name"] in ["vehicle", "pedestrian"]:
                    self._create_new_track(detection, timestamp)
            return self.tracks
        
        # Predict new locations of existing tracks
        for track in self.tracks:
            track.predict()
        
        # Separate detections by class
        vehicle_detections = [d for d in detections if d["class_name"] == "vehicle"]
        pedestrian_detections = [d for d in detections if d["class_name"] == "pedestrian"]
        
        # Update vehicle tracks
        vehicle_tracks = [t for t in self.tracks if t.class_name == "vehicle"]
        self._update_class_tracks(vehicle_tracks, vehicle_detections, timestamp)
        
        # Update pedestrian tracks
        pedestrian_tracks = [t for t in self.tracks if t.class_name == "pedestrian"]
        self._update_class_tracks(pedestrian_tracks, pedestrian_detections, timestamp)
        
        # Remove old tracks
        self.tracks = [t for t in self.tracks if t.time_since_update <= self.max_age]
        
        # Update license plate and helmet information
        self._update_additional_info(detections)
        
        return self.tracks
    
    def _update_class_tracks(self, class_tracks, class_detections, timestamp):
        """
        Update tracks of a specific class.
        
        Args:
            class_tracks (list): List of tracks of a specific class
            class_detections (list): List of detections of a specific class
            timestamp (float): Current timestamp
        """
        # Calculate IoU matrix
        iou_matrix = np.zeros((len(class_tracks), len(class_detections)))
        for i, track in enumerate(class_tracks):
            for j, detection in enumerate(class_detections):
                iou_matrix[i, j] = self._calculate_iou(track.bbox, detection["bbox"])
        
        # Hungarian algorithm for optimal assignment
        if len(class_tracks) > 0 and len(class_detections) > 0:
            # Find matches
            matched_indices = []
            
            # Greedy matching for simplicity
            # In a real system, use the Hungarian algorithm
            for i in range(len(class_tracks)):
                max_iou = self.iou_threshold
                max_j = -1
                for j in range(len(class_detections)):
                    if iou_matrix[i, j] > max_iou:
                        max_iou = iou_matrix[i, j]
                        max_j = j
                
                if max_j != -1:
                    matched_indices.append((i, max_j))
            
            # Update matched tracks
            for i, j in matched_indices:
                track = class_tracks[i]
                detection = class_detections[j]
                track.update(detection["bbox"], detection["confidence"], timestamp)
            
            # Create new tracks for unmatched detections
            matched_det_indices = [j for _, j in matched_indices]
            unmatched_detections = [d for i, d in enumerate(class_detections) if i not in matched_det_indices]
            
            for detection in unmatched_detections:
                self._create_new_track(detection, timestamp)
    
    def _create_new_track(self, detection, timestamp):
        """
        Create a new track from a detection.
        
        Args:
            detection (dict): Detection object
            timestamp (float): Current timestamp
        """
        track = Track(
            track_id=self.next_id,
            bbox=detection["bbox"],
            class_name=detection["class_name"],
            confidence=detection["confidence"]
        )
        track.timestamps[0] = timestamp
        self.tracks.append(track)
        self.next_id += 1
    
    def _update_additional_info(self, detections):
        """
        Update license plate and helmet information for tracks.
        
        Args:
            detections (list): List of detection objects
        """
        # Update license plate information
        license_plate_detections = [d for d in detections if d["class_name"] == "license_plate"]
        for lp_detection in license_plate_detections:
            if "vehicle_id" in lp_detection:
                # Find the corresponding vehicle track
                for track in self.tracks:
                    if track.class_name == "vehicle" and id(track) == lp_detection["vehicle_id"]:
                        track.license_plate = lp_detection
                        break
        
        # Update helmet information
        helmet_detections = [d for d in detections if d["class_name"] in ["helmet", "no_helmet"]]
        for helmet_detection in helmet_detections:
            if "pedestrian_id" in helmet_detection:
                # Find the corresponding pedestrian track
                for track in self.tracks:
                    if track.class_name == "pedestrian" and id(track) == helmet_detection["pedestrian_id"]:
                        track.helmet_status = helmet_detection["class_name"]
                        break
    
    def calculate_speeds(self, calibration_factor):
        """
        Calculate speeds for all vehicle tracks.
        
        Args:
            calibration_factor (float): Pixels to meters conversion factor
        """
        for track in self.tracks:
            if track.class_name == "vehicle" and track.state == "confirmed":
                track.calculate_speed(calibration_factor, self.fps)
    
    def set_fps(self, fps):
        """
        Set the FPS value for speed calculation.
        
        Args:
            fps (float): Frames per second
        """
        self.fps = fps
