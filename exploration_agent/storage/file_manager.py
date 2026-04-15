"""
File and directory management
"""
import os
import cv2
import numpy as np 
from typing import List

class FileManager:
    def __init__(self, config):
        self.config = config
        self._create_directories()
    
    def _create_directories(self):
        #Create all necessary directories
        directories = [
            self.config.frame_dir,
            self.config.detection_dir,
            self.config.results_dir
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def save_frame(self, filename: str, frame: np.ndarray) -> str:
        #Save a frame to the directory
        filepath = os.path.join(self.config.frame_dir, filename)
        cv2.imwrite(filepath, frame)
        return filepath
    
    def save_detection_frame(self, filename: str, frame: np.ndarray) -> str:
        #Save a detection frame to the detection directory
        filepath = os.path.join(self.config.detection_dir, filename)
        cv2.imwrite(filepath, frame)
        return filepath
    
    def get_frame_filename(self, 
                          frame_num: int,
                          position: List[float],
                          direction: float,
                          view_type: str = "view") -> str:
        #Generate a filename for a frame based on its metadata
        return (f"frame_{frame_num:04d}_x{position[0]:.1f}_"
                f"z{position[2]:.1f}_dir{np.degrees(direction):.0f}_{view_type}.png")