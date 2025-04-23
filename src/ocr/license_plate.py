#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
License plate recognition module for traffic surveillance.
"""

import os
import logging
import cv2
import numpy as np

logger = logging.getLogger(__name__)

class LicensePlateRecognizer:
    """
    Class for recognizing license plates in traffic surveillance footage.
    """
    
    def __init__(self, engine="easyocr", languages=None):
        """
        Initialize the license plate recognizer.
        
        Args:
            engine (str): OCR engine to use (easyocr, tesseract)
            languages (list): List of languages to recognize
        """
        self.engine = engine
        self.languages = languages or ["en"]
        self.ocr = None
        
        logger.info(f"Initializing license plate recognizer with {engine} engine")
        logger.info(f"Languages: {self.languages}")
        
        self._load_ocr_engine()
    
    def _load_ocr_engine(self):
        """
        Load the OCR engine.
        """
        try:
            if self.engine == "easyocr":
                import easyocr
                self.ocr = easyocr.Reader(self.languages, gpu=True)
                logger.info("EasyOCR engine loaded successfully")
            elif self.engine == "tesseract":
                import pytesseract
                self.ocr = pytesseract
                logger.info("Tesseract engine loaded successfully")
            else:
                logger.error(f"Unsupported OCR engine: {self.engine}")
                raise ValueError(f"Unsupported OCR engine: {self.engine}")
        except ImportError as e:
            logger.error(f"Error loading OCR engine: {e}")
            logger.info("Falling back to simple pattern matching")
            self.engine = "pattern_matching"
    
    def preprocess_plate(self, plate_img):
        """
        Preprocess the license plate image for better OCR.
        
        Args:
            plate_img (numpy.ndarray): License plate image
            
        Returns:
            numpy.ndarray: Preprocessed image
        """
        # Resize the image
        plate_img = cv2.resize(plate_img, (0, 0), fx=2, fy=2)
        
        # Convert to grayscale
        gray = cv2.cvtColor(plate_img, cv2.COLOR_BGR2GRAY)
        
        # Apply bilateral filter to remove noise while keeping edges sharp
        filtered = cv2.bilateralFilter(gray, 11, 17, 17)
        
        # Apply threshold to get black and white image
        _, thresh = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological operations to remove noise
        kernel = np.ones((3, 3), np.uint8)
        opening = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)
        
        return opening
    
    def recognize(self, frame, license_plate_detections):
        """
        Recognize license plates in the frame.
        
        Args:
            frame (numpy.ndarray): Input frame
            license_plate_detections (list): List of license plate detection objects
            
        Returns:
            list: List of license plate recognition results
        """
        results = []
        
        for detection in license_plate_detections:
            x1, y1, x2, y2 = detection["bbox"]
            
            # Extract license plate ROI
            plate_img = frame[y1:y2, x1:x2]
            if plate_img.size == 0:
                continue
            
            # Preprocess the plate image
            preprocessed = self.preprocess_plate(plate_img)
            
            # Recognize text
            text = ""
            confidence = 0.0
            
            try:
                if self.engine == "easyocr":
                    ocr_result = self.ocr.readtext(preprocessed)
                    if ocr_result:
                        # Combine all detected text
                        text_parts = []
                        conf_sum = 0.0
                        for res in ocr_result:
                            text_parts.append(res[1])
                            conf_sum += res[2]
                        
                        text = " ".join(text_parts)
                        confidence = conf_sum / len(ocr_result)
                
                elif self.engine == "tesseract":
                    # Configure tesseract for license plate recognition
                    config = "--psm 7 --oem 1 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
                    text = self.ocr.image_to_string(preprocessed, config=config).strip()
                    confidence = 0.8  # Tesseract doesn't provide confidence
                
                elif self.engine == "pattern_matching":
                    # Simple pattern matching (placeholder)
                    text = "ABC123"
                    confidence = 0.5
            
            except Exception as e:
                logger.error(f"Error recognizing license plate: {e}")
                text = ""
                confidence = 0.0
            
            # Filter out unlikely license plates
            if self._validate_license_plate(text):
                results.append({
                    "bbox": detection["bbox"],
                    "text": text,
                    "confidence": confidence,
                    "vehicle_id": detection.get("vehicle_id")
                })
        
        return results
    
    def _validate_license_plate(self, text):
        """
        Validate if the recognized text is a valid license plate.
        
        Args:
            text (str): Recognized text
            
        Returns:
            bool: True if valid, False otherwise
        """
        # Remove spaces and special characters
        text = "".join(c for c in text if c.isalnum())
        
        # Check if the text is empty
        if not text:
            return False
        
        # Check if the text is too short or too long
        if len(text) < 4 or len(text) > 10:
            return False
        
        # Check if the text contains at least one letter and one digit
        has_letter = any(c.isalpha() for c in text)
        has_digit = any(c.isdigit() for c in text)
        
        return has_letter and has_digit
