#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Object detection module for traffic surveillance.
Supports detection of vehicles, pedestrians, helmets, and license plates.
"""

import os
import logging
import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)

class ObjectDetector:
    """
    Class for detecting objects in traffic surveillance footage.
    Supports YOLOv8 and other detection models.
    """
    
    def __init__(self, model_type="yolov8", model_size="m", confidence_threshold=0.5, classes=None):
        """
        Initialize the detector with models for object detection.
        
        Args:
            model_type (str): Type of model to use (yolov8, yolov5, etc.)
            model_size (str): Size of the model (n, s, m, l, x)
            confidence_threshold (float): Minimum confidence for detection (0-1)
            classes (list): List of classes to detect
        """
        self.model_type = model_type
        self.model_size = model_size
        self.confidence_threshold = confidence_threshold
        self.classes = classes or ["vehicle", "pedestrian", "helmet", "no_helmet", "license_plate"]
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        logger.info(f"Initializing {model_type}{model_size} detector on {self.device}")
        logger.info(f"Classes to detect: {self.classes}")
        
        self.model = self._load_model()
        
    def _load_model(self):
        """
        Load the detection model.
        
        Returns:
            model: Loaded detection model
        """
        try:
            if self.model_type == "yolov8":
                from ultralytics import YOLO
                # Try to load a custom trained model first
                custom_model_path = f"data/models/detection/yolov8{self.model_size}_custom.pt"
                if os.path.exists(custom_model_path):
                    logger.info(f"Loading custom model from {custom_model_path}")
                    model = YOLO(custom_model_path)
                else:
                    # Load pre-trained model
                    logger.info(f"Loading pre-trained YOLOv8{self.model_size} model")
                    model = YOLO(f"yolov8{self.model_size}.pt")
                return model
            else:
                logger.error(f"Unsupported model type: {self.model_type}")
                raise ValueError(f"Unsupported model type: {self.model_type}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            logger.info("Falling back to OpenCV DNN models")
            return self._load_opencv_models()
    
    def _load_opencv_models(self):
        """
        Load OpenCV DNN models as fallback.
        
        Returns:
            dict: Dictionary of loaded models
        """
        models = {}
        
        # Load vehicle and person detection model (MobileNet SSD)
        try:
            prototxt = os.path.join(cv2.data.haarcascades, "MobileNetSSD_deploy.prototxt")
            model = os.path.join(cv2.data.haarcascades, "MobileNetSSD_deploy.caffemodel")
            if os.path.exists(prototxt) and os.path.exists(model):
                models["ssd"] = cv2.dnn.readNetFromCaffe(prototxt, model)
                models["ssd_classes"] = ["background", "aeroplane", "bicycle", "bird", "boat",
                                        "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                                        "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                                        "sofa", "train", "tvmonitor"]
        except Exception as e:
            logger.error(f"Error loading MobileNet SSD: {e}")
        
        # Load Haar cascades
        try:
            models["car_cascade"] = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, 'haarcascade_car.xml'))
            models["person_cascade"] = cv2.CascadeClassifier(os.path.join(cv2.data.haarcascades, 'haarcascade_fullbody.xml'))
        except Exception as e:
            logger.error(f"Error loading Haar cascades: {e}")
        
        # Initialize HOG descriptor for pedestrian detection
        try:
            models["hog"] = cv2.HOGDescriptor()
            models["hog"].setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        except Exception as e:
            logger.error(f"Error initializing HOG descriptor: {e}")
        
        return models
    
    def detect(self, frame):
        """
        Detect objects in a frame.
        
        Args:
            frame (numpy.ndarray): Input frame
            
        Returns:
            tuple: (detections, annotated_frame)
                - detections: List of detection objects
                - annotated_frame: Frame with annotations (if requested)
        """
        if self.model_type == "yolov8" and not isinstance(self.model, dict):
            return self._detect_yolov8(frame)
        else:
            return self._detect_opencv(frame)
    
    def _detect_yolov8(self, frame):
        """
        Detect objects using YOLOv8.
        
        Args:
            frame (numpy.ndarray): Input frame
            
        Returns:
            tuple: (detections, annotated_frame)
        """
        results = self.model(frame, conf=self.confidence_threshold)
        
        # Process results
        detections = []
        
        for result in results:
            boxes = result.boxes.cpu().numpy()
            
            for i, box in enumerate(boxes):
                x1, y1, x2, y2 = box.xyxy[0].astype(int)
                confidence = float(box.conf[0])
                class_id = int(box.cls[0])
                class_name = result.names[class_id]
                
                # Only include classes we're interested in
                if class_name in self.classes or class_id in self.classes:
                    detections.append({
                        "bbox": [x1, y1, x2, y2],
                        "confidence": confidence,
                        "class_id": class_id,
                        "class_name": class_name
                    })
        
        # Get annotated frame
        annotated_frame = results[0].plot()
        
        return detections, annotated_frame
    
    def _detect_opencv(self, frame):
        """
        Detect objects using OpenCV models.
        
        Args:
            frame (numpy.ndarray): Input frame
            
        Returns:
            tuple: (detections, annotated_frame)
        """
        detections = []
        annotated_frame = frame.copy()
        
        # Use MobileNet SSD if available
        if "ssd" in self.model:
            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
            self.model["ssd"].setInput(blob)
            ssd_detections = self.model["ssd"].forward()
            
            for i in range(ssd_detections.shape[2]):
                confidence = ssd_detections[0, 0, i, 2]
                
                if confidence > self.confidence_threshold:
                    idx = int(ssd_detections[0, 0, i, 1])
                    box = ssd_detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (x1, y1, x2, y2) = box.astype("int")
                    
                    class_name = self.model["ssd_classes"][idx]
                    
                    # Map SSD classes to our classes
                    if class_name in ["car", "bus", "motorbike", "truck", "bicycle"]:
                        mapped_class = "vehicle"
                    elif class_name == "person":
                        mapped_class = "pedestrian"
                    else:
                        continue
                    
                    detections.append({
                        "bbox": [x1, y1, x2, y2],
                        "confidence": float(confidence),
                        "class_id": idx,
                        "class_name": mapped_class
                    })
                    
                    # Draw bounding box
                    color = (0, 0, 255) if mapped_class == "vehicle" else (0, 255, 0)
                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(annotated_frame, f"{mapped_class}: {confidence:.2f}",
                                (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        # Use Haar cascades as fallback
        else:
            # Detect vehicles
            if "car_cascade" in self.model:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                vehicles = self.model["car_cascade"].detectMultiScale(gray, 1.1, 3)
                
                for (x, y, w, h) in vehicles:
                    detections.append({
                        "bbox": [x, y, x + w, y + h],
                        "confidence": 1.0,  # Haar cascades don't provide confidence
                        "class_id": 0,
                        "class_name": "vehicle"
                    })
                    
                    # Draw bounding box
                    cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(annotated_frame, "Vehicle", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
            
            # Detect pedestrians using HOG
            if "hog" in self.model:
                pedestrians, _ = self.model["hog"].detectMultiScale(
                    frame, winStride=(4, 4), padding=(8, 8), scale=1.05)
                
                for (x, y, w, h) in pedestrians:
                    detections.append({
                        "bbox": [x, y, x + w, y + h],
                        "confidence": 1.0,  # HOG doesn't provide confidence
                        "class_id": 1,
                        "class_name": "pedestrian"
                    })
                    
                    # Draw bounding box
                    cv2.rectangle(annotated_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(annotated_frame, "Pedestrian", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return detections, annotated_frame
    
    def detect_license_plates(self, frame, vehicle_detections):
        """
        Detect license plates within vehicle detections.
        
        Args:
            frame (numpy.ndarray): Input frame
            vehicle_detections (list): List of vehicle detection objects
            
        Returns:
            list: List of license plate detections
        """
        license_plate_detections = []
        
        # For each vehicle detection, try to find a license plate
        for vehicle in vehicle_detections:
            x1, y1, x2, y2 = vehicle["bbox"]
            
            # Extract vehicle ROI
            vehicle_roi = frame[y1:y2, x1:x2]
            if vehicle_roi.size == 0:
                continue
            
            # Convert to grayscale
            gray = cv2.cvtColor(vehicle_roi, cv2.COLOR_BGR2GRAY)
            
            # Apply Gaussian blur
            blur = cv2.GaussianBlur(gray, (5, 5), 0)
            
            # Apply Sobel edge detection
            sobel_x = cv2.Sobel(blur, cv2.CV_8U, 1, 0, ksize=3)
            
            # Apply threshold
            _, thresh = cv2.threshold(sobel_x, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Find contours
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            # Filter contours by aspect ratio and area
            for contour in contours:
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = float(w) / h
                area = w * h
                
                # License plates typically have an aspect ratio between 2 and 5
                # and occupy a reasonable portion of the vehicle
                if 2.0 < aspect_ratio < 5.0 and area > 500 and area < 15000:
                    # Convert to global coordinates
                    global_x1 = x1 + x
                    global_y1 = y1 + y
                    global_x2 = global_x1 + w
                    global_y2 = global_y1 + h
                    
                    license_plate_detections.append({
                        "bbox": [global_x1, global_y1, global_x2, global_y2],
                        "confidence": 0.8,  # Arbitrary confidence
                        "class_id": 4,
                        "class_name": "license_plate",
                        "vehicle_id": id(vehicle)  # Associate with vehicle
                    })
        
        return license_plate_detections
    
    def detect_helmets(self, frame, pedestrian_detections):
        """
        Detect helmets on pedestrians (motorcycle riders).
        
        Args:
            frame (numpy.ndarray): Input frame
            pedestrian_detections (list): List of pedestrian detection objects
            
        Returns:
            list: List of helmet/no_helmet detections
        """
        helmet_detections = []
        
        # For each pedestrian detection, try to determine if they're wearing a helmet
        for pedestrian in pedestrian_detections:
            x1, y1, x2, y2 = pedestrian["bbox"]
            
            # Extract head region (upper 1/3 of the pedestrian bounding box)
            head_y2 = y1 + (y2 - y1) // 3
            head_roi = frame[y1:head_y2, x1:x2]
            if head_roi.size == 0:
                continue
            
            # Simple color-based detection (for demonstration)
            # In a real system, you would use a trained helmet classifier
            hsv = cv2.cvtColor(head_roi, cv2.COLOR_BGR2HSV)
            
            # Define color ranges for common helmet colors
            # (this is a simplified approach and would need refinement)
            helmet_colors = [
                # Red helmets
                (np.array([0, 100, 100]), np.array([10, 255, 255])),
                (np.array([160, 100, 100]), np.array([180, 255, 255])),
                # Blue helmets
                (np.array([100, 100, 100]), np.array([140, 255, 255])),
                # Yellow helmets
                (np.array([20, 100, 100]), np.array([40, 255, 255])),
                # White helmets
                (np.array([0, 0, 200]), np.array([180, 30, 255]))
            ]
            
            # Check for helmet colors
            has_helmet = False
            for lower, upper in helmet_colors:
                mask = cv2.inRange(hsv, lower, upper)
                if cv2.countNonZero(mask) > (head_roi.shape[0] * head_roi.shape[1] * 0.2):
                    has_helmet = True
                    break
            
            # Add detection
            class_name = "helmet" if has_helmet else "no_helmet"
            helmet_detections.append({
                "bbox": [x1, y1, x2, head_y2],
                "confidence": 0.7,  # Arbitrary confidence
                "class_id": 2 if has_helmet else 3,
                "class_name": class_name,
                "pedestrian_id": id(pedestrian)  # Associate with pedestrian
            })
        
        return helmet_detections
