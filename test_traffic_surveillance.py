import unittest
import os
import cv2
import numpy as np
from detector import ObjectDetector
from visualization import VisualizationEngine

class TestTrafficSurveillance(unittest.TestCase):
    """
    Test cases for the Traffic Surveillance System.
    """
    
    def setUp(self):
        """
        Set up test environment.
        """
        self.detector = ObjectDetector(confidence_threshold=0.5)
        self.visualizer = VisualizationEngine()
        
        # Create a simple test frame with a rectangle (simulating a vehicle)
        self.test_frame = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.rectangle(self.test_frame, (100, 200), (200, 300), (0, 0, 255), -1)
        
    def test_detector_initialization(self):
        """
        Test that the detector initializes correctly.
        """
        self.assertIsNotNone(self.detector)
        self.assertEqual(self.detector.confidence_threshold, 0.5)
        
    def test_detector_detect(self):
        """
        Test that the detector can process a frame.
        """
        processed_frame, detections = self.detector.detect(self.test_frame)
        
        # Check that the processed frame is not None
        self.assertIsNotNone(processed_frame)
        
        # Check that detections is a dictionary with the expected keys
        self.assertIsInstance(detections, dict)
        self.assertIn("vehicles", detections)
        self.assertIn("pedestrians", detections)
        
    def test_visualizer_initialization(self):
        """
        Test that the visualizer initializes correctly.
        """
        self.assertIsNotNone(self.visualizer)
        self.assertEqual(self.visualizer.window_name, "Traffic Surveillance")
        
    def test_visualizer_add_info_panel(self):
        """
        Test that the visualizer can add an information panel to a frame.
        """
        detections = {"vehicles": 2, "pedestrians": 1}
        frame_with_panel = self.visualizer.add_info_panel(self.test_frame, detections)
        
        # Check that the frame with panel is not None
        self.assertIsNotNone(frame_with_panel)
        
        # Check that the frame with panel is larger than the original frame
        # (because it adds a panel at the bottom)
        self.assertGreater(frame_with_panel.shape[0], self.test_frame.shape[0])
        
    def test_create_sample_video(self):
        """
        Test the sample video creation function.
        """
        from traffic_surveillance import create_sample_video
        
        # Create a sample video
        sample_path = "test_sample.mp4"
        create_sample_video(sample_path, duration=1, fps=10)
        
        # Check that the file exists
        self.assertTrue(os.path.exists(sample_path))
        
        # Check that the file is a valid video
        cap = cv2.VideoCapture(sample_path)
        self.assertTrue(cap.isOpened())
        
        # Clean up
        cap.release()
        if os.path.exists(sample_path):
            os.remove(sample_path)

if __name__ == "__main__":
    unittest.main()
