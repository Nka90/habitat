"""
Configuration settings for the exploration agent
"""
import os
from dataclasses import dataclass
import shutil
from typing import Tuple, List

@dataclass
class RoomBounds:
    #Room boundaries configuration for maps
    x_min: float = -4.5
    x_max: float = 4.5
    z_min: float = -4.5
    z_max: float = 4.5
    
    def get_boundary_points(self) -> List[Tuple[float, float]]:
        #Return boundary points for visualization
        return [
            (self.x_min, self.z_min),
            (self.x_max, self.z_min),
            (self.x_max, self.z_max),
            (self.x_min, self.z_max),
            (self.x_min, self.z_min)
        ]

@dataclass
class AgentConfig:
    #Agent configuration
    agent_height: float = 0.0  # Agent stands on floor
    yolo_stop_interval: int = 5  # Stop every n steps for detection
    max_total_steps: int = 300 #max steps of discovering the environment
    agent_radius: float = 0.5  # Radius of the agent for collision checks

@dataclass
class PathConfig:
    max_frontiers: int = 8 #in bigger environment can be increased

@dataclass
class PathsConfig:
    #File and directory paths
    base_dir: str = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    scene_path: str = "data/scene_datasets/environment.glb"
    yolo_model_path: str = "data/yolo_model/best.pt"
    frame_dir: str = "exploration_frames"
    detection_dir: str = "detection_results"
    results_dir: str = "results"
    
    def create_directories(self):
        #Create all necessary directories
        for dir_path in [self.frame_dir, self.detection_dir, self.results_dir]:
            os.makedirs(dir_path, exist_ok=True)
    
    def initialDeleteDirectories(self):
        #Delete existing directories before starting a new run
        for dir_path in [self.frame_dir, self.detection_dir, self.results_dir]:
            if os.path.exists(dir_path):
                shutil.rmtree(dir_path)
            os.makedirs(dir_path, exist_ok=True)

@dataclass
class CameraConfig:
    fov: float = 90.0 #field of view in degrees
    width: int = 640 #frame width in pixels
    height: int = 640 #frame height in pixels
    sensor_height: float = 1.5 #height of camera from floor