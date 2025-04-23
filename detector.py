import cv2
import numpy as np
import os

class ObjectDetector:
    """
    Class for detecting vehicles and pedestrians in traffic surveillance footage
    using pre-trained models.
    """
    
    def __init__(self, confidence_threshold=0.5):
        """
        Initialize the detector with models for vehicle and pedestrian detection.
        
        Args:
            confidence_threshold (float): Minimum confidence for detection (0-1)
        """
        self.confidence_threshold = confidence_threshold
        self.classes = []
        self.load_models()
        
    def load_models(self):
        """
        Load pre-trained models for object detection.
        Uses OpenCV's DNN module with MobileNet SSD.
        """
        # Path to the pre-trained model files
        model_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
        os.makedirs(model_dir, exist_ok=True)
        
        # Check if model files exist, otherwise use OpenCV's built-in HOG detector
        self.use_hog = True
        
        if self.use_hog:
            # Initialize HOG descriptor for pedestrian detection
            self.hog = cv2.HOGDescriptor()
            self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            
            # Initialize Haar cascade for vehicle detection
            cascade_file = os.path.join(cv2.data.haarcascades, 'haarcascade_car.xml')
            if os.path.exists(cascade_file):
                self.car_cascade = cv2.CascadeClassifier(cascade_file)
            else:
                print(f"Warning: Car cascade file not found at {cascade_file}")
                self.car_cascade = None
        else:
            # Load MobileNet SSD model
            print("Loading MobileNet SSD model...")
            prototxt = os.path.join(model_dir, "MobileNetSSD_deploy.prototxt")
            model = os.path.join(model_dir, "MobileNetSSD_deploy.caffemodel")
            
            if os.path.exists(prototxt) and os.path.exists(model):
                self.net = cv2.dnn.readNetFromCaffe(prototxt, model)
                self.classes = ["background", "aeroplane", "bicycle", "bird", "boat",
                               "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                               "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                               "sofa", "train", "tvmonitor"]
            else:
                print(f"Warning: MobileNet SSD model files not found. Using HOG detector instead.")
                self.use_hog = True
                self.hog = cv2.HOGDescriptor()
                self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
    
    def detect(self, frame):
        """
        Detect vehicles and pedestrians in a frame.
        
        Args:
            frame (numpy.ndarray): Input frame
            
        Returns:
            tuple: (processed_frame, detections)
                - processed_frame: Frame with annotations
                - detections: Dictionary with detection counts
        """
        # Create a copy of the frame to draw on
        output = frame.copy()
        
        # Initialize detection counts
        detections = {
            "vehicles": 0,
            "pedestrians": 0
        }
        
        if self.use_hog:
            # Detect pedestrians using HOG
            pedestrians, _ = self.hog.detectMultiScale(
                frame, 
                winStride=(4, 4),
                padding=(8, 8), 
                scale=1.05
            )
            
            # Draw bounding boxes for pedestrians
            for (x, y, w, h) in pedestrians:
                cv2.rectangle(output, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(output, "Pedestrian", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                detections["pedestrians"] += 1
            
            # Detect vehicles using Haar cascade if available
            if self.car_cascade is not None:
                vehicles = self.car_cascade.detectMultiScale(
                    cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY),
                    scaleFactor=1.1, 
                    minNeighbors=5,
                    minSize=(30, 30)
                )
                
                # Draw bounding boxes for vehicles
                for (x, y, w, h) in vehicles:
                    cv2.rectangle(output, (x, y), (x + w, y + h), (0, 0, 255), 2)
                    cv2.putText(output, "Vehicle", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    detections["vehicles"] += 1
        else:
            # Use MobileNet SSD for detection
            (h, w) = frame.shape[:2]
            blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
            self.net.setInput(blob)
            detections_ssd = self.net.forward()
            
            # Process each detection
            for i in range(detections_ssd.shape[2]):
                confidence = detections_ssd[0, 0, i, 2]
                
                if confidence > self.confidence_threshold:
                    idx = int(detections_ssd[0, 0, i, 1])
                    box = detections_ssd[0, 0, i, 3:7] * np.array([w, h, w, h])
                    (startX, startY, endX, endY) = box.astype("int")
                    
                    # Get the class label
                    label = self.classes[idx]
                    
                    # Count vehicles (car, bus, motorbike) and pedestrians (person)
                    if label in ["car", "bus", "motorbike", "truck"]:
                        color = (0, 0, 255)  # Red for vehicles
                        detections["vehicles"] += 1
                        display_label = "Vehicle"
                    elif label == "person":
                        color = (0, 255, 0)  # Green for pedestrians
                        detections["pedestrians"] += 1
                        display_label = "Pedestrian"
                    else:
                        continue  # Skip other classes
                    
                    # Draw the bounding box and label
                    cv2.rectangle(output, (startX, startY), (endX, endY), color, 2)
                    y = startY - 15 if startY - 15 > 15 else startY + 15
                    cv2.putText(output, f"{display_label}: {confidence:.2f}",
                                (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return output, detections
