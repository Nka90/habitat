"""
Detection module for YOLO object detection.
"""
from .yolo_detector import YOLODetector
from .detection_processor import DetectionProcessor

__all__ = [
    'YOLODetector',
    'DetectionProcessor'
]