#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Script to create a more realistic test image for helmet detection.
"""

import os
import cv2
import numpy as np

def create_test_image(output_path="data/output/test_image.jpg"):
    """Create a test image with a person wearing a helmet."""
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Create a blank image (road scene)
    img = np.zeros((640, 640, 3), dtype=np.uint8)
    
    # Add a background (road)
    cv2.rectangle(img, (0, 0), (640, 640), (120, 120, 120), -1)
    
    # Add road markings
    # Center line
    cv2.line(img, (0, 320), (640, 320), (255, 255, 255), 2)
    
    # Lane markings
    for x in range(0, 640, 50):
        cv2.line(img, (x, 160), (x + 30, 160), (255, 255, 255), 2)
        cv2.line(img, (x, 480), (x + 30, 480), (255, 255, 255), 2)
    
    # Draw a motorcycle
    # Body
    cv2.rectangle(img, (200, 300), (400, 350), (50, 50, 200), -1)
    # Wheels
    cv2.circle(img, (220, 350), 30, (0, 0, 0), -1)
    cv2.circle(img, (380, 350), 30, (0, 0, 0), -1)
    cv2.circle(img, (220, 350), 20, (200, 200, 200), -1)
    cv2.circle(img, (380, 350), 20, (200, 200, 200), -1)
    # Handlebars
    cv2.rectangle(img, (200, 280), (250, 300), (50, 50, 200), -1)
    
    # Draw a person
    # Body
    cv2.rectangle(img, (280, 200), (320, 300), (0, 0, 255), -1)
    # Head
    cv2.circle(img, (300, 180), 20, (255, 200, 150), -1)
    
    # Draw a helmet
    cv2.ellipse(img, (300, 170), (25, 30), 0, 0, 360, (0, 255, 0), -1)
    
    # Add text labels
    cv2.putText(img, "Motorcycle", (200, 400), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(img, "Person", (330, 250), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    cv2.putText(img, "Helmet", (330, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    # Save the image
    cv2.imwrite(output_path, img)
    print(f"Test image saved to {output_path}")
    
    return output_path

if __name__ == "__main__":
    create_test_image()
