"""
Perception cycle: the connection happens through callbacks. The PerceptionCycle doesn't directly call YOLO - it triggers callbacks 
that the DetectionProcessor registered PerceptionCycle.step() → calls capture_frame() → triggers callbacks → DetectionProcessor._on_frame_captured() → YOLO detection
"""
from typing import Callable, List
import cv2
import numpy as np
from environment.path_utils import PathUtils


class PerceptionCycle:
    def __init__(self, sim, yolo_stop_interval: int = 5):
        self.sim = sim
        self.yolo_stop_interval = yolo_stop_interval
        self.steps_since_last_yolo = 0
        self.stop_points: List[List[float]] = []
        self.frame_callbacks = [] #callbacks need to decouple the perception cycle from detection logic
        self.current_direction = 0.0
        self.frame_count = 0
    
    def register_frame_callback(self, callback: Callable):
        #Register callback for when a frame is captured
        self.frame_callbacks.append(callback)
    
    def set_direction(self, direction: float): 
        #records direction of agent for metadata
        self.current_direction = direction
    
    def capture_frame(self, 
                     movement_controller,
                     position: np.ndarray,
                     step_number: int,
                     view_type: str) -> bool:
        if position is None:
            position = movement_controller.get_state().position
        
        # Capture frame from sensors
        frame_rgb, frame_bgr, depth = self.make_frame()
        
        # Create metadata
        metadata = {
            'x': float(position[0]), 
            'y': float(position[1]), 
            'z': float(position[2]),
            'direction': self.current_direction,
            'view_type': view_type,
            'step': step_number,
            'depth': depth,
            'timestamp': step_number
        }
        
        #observer: trigger callbacks with the captured frame and metadata - this is where the detection processor will get the frame to run YOLO on if it's a stop frame
        for callback in self.frame_callbacks:
            callback(frame_rgb, metadata)
        
        return True
    
    def make_frame(self):
        #Capture color and depth frames
        obs = self.sim.get_sensor_observations() # Get the latest sensor observations
    
        #different components expect different color formats: yolo expect rgb, opencv expect bgr, habitat provides rgba, depth helps to calculate distance for coordinates
        frame = obs["color_sensor"]
        if frame.shape[2] == 4: 
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_RGBA2RGB)
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        else:
            frame_rgb = frame
            frame_bgr = frame
        
        # Depth frame
        depth = obs["depth_sensor"]
        
        self.frame_count += 1
        return frame_rgb, frame_bgr, depth
        
    def step(self, movement_controller, step_number: int) -> bool:
        #check if it's time for YOLO detection
        self.steps_since_last_yolo += 1
        position = movement_controller.get_state().position
        
        # if it is time
        if self.steps_since_last_yolo >= self.yolo_stop_interval:
            # Record stop point
            pos_list = PathUtils.get_position_as_list(position)
            self.stop_points.append(pos_list)
            
            # Capture frame that should trigger detection
            self.capture_frame(
                movement_controller=movement_controller,
                position=position,
                step_number=step_number,
                view_type="stop"  #triggers YOLO
            )
            
            self.steps_since_last_yolo = 0
            return True
        else:
            # Regular movement frame - just capture, no YOLO
            self.capture_frame(
                movement_controller=movement_controller,
                position=position,
                step_number=step_number,
                view_type="after_movement"  # Regular movement frame
            )
            return False