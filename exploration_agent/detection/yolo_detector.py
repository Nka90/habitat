"""
YOLO object detection module
"""
import os
import cv2
import numpy as np
from ultralytics import YOLO
from typing import Optional, Dict, Any, List


class YOLODetector:
    def __init__(self, model_path: str, device: str = 'cpu'):
        self.model_path = model_path
        self.device = device
        self.model = None
        self.load_model()
    
    def load_model(self) -> bool:
        #Load the YOLO model
        try:
            if not os.path.exists(self.model_path):
                print(f"Model file {self.model_path} not found")
                return False    
            self.model = YOLO(self.model_path)
            self.model.to(self.device)
            return True
            
        except Exception as e:
            print(f"ERROR loading YOLO model: {e}")
            self.model = None
            return False
    
    def detect(self, frame_rgb: np.ndarray, verbose: bool = False) -> Optional[Any]:
        #Run object detection on the input frame
        if self.model is None:
            return None
        try:
            results = self.model(frame_rgb, verbose=verbose) #results: bboxes, confidence, coordinates of boxes, class id, ect.
            return results
        except Exception as e:
            print(f"Error during object detection: {e}")
            return None

    def draw_detections(self, frame_rgb: np.ndarray, 
                       detections: List[Dict]) -> np.ndarray:
        #Draw bounding boxes on frame
        frame_with_boxes = frame_rgb.copy() #copy the input frame to draw on
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class_name']
            
            cv2.rectangle(frame_with_boxes, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = f"{class_name} {confidence:.2f}"
            cv2.putText(frame_with_boxes, label, (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        
        return frame_with_boxes