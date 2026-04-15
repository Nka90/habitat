"""
Helping geometry methods
"""
import math
import numpy as np
from typing import List, Tuple

class PathUtils:
    @staticmethod
    def get_position_as_list(position) -> List[float]:
        #Convert any position format to list [x, y, z]
        if hasattr(position, 'x'):
            return [position.x, position.y, position.z]
        elif isinstance(position, (list, tuple, np.ndarray)) and len(position) >= 3:
            return [float(position[0]), float(position[1]), float(position[2])]
        else:
            try:
                return [float(position[0]), float(position[1]), float(position[2])]
            except:
                return [0.0, 0.0, 0.0]
    
    @staticmethod
    def get_position_key(position) -> Tuple[float, float, float]:
        #Create a unique identifier for a position
        pos_list = PathUtils.get_position_as_list(position)
        return (round(pos_list[0], 1), round(pos_list[1], 1), round(pos_list[2], 1))
    
    @staticmethod
    def calculate_distance(pos1, pos2) -> float:
        #Calculate Euclidean distance (ignore height)
        pos1_list = PathUtils.get_position_as_list(pos1)
        pos2_list = PathUtils.get_position_as_list(pos2)
        return math.sqrt((pos1_list[0] - pos2_list[0])**2 + 
                        (pos1_list[2] - pos2_list[2])**2)
    
    @staticmethod
    def calculate_direction_to_point(from_point, to_point) -> float:
        #calculate direction to frontier (need to rotate agent to face it)
        from_list = PathUtils.get_position_as_list(from_point)
        to_list = PathUtils.get_position_as_list(to_point)
        
        dx = to_list[0] - from_list[0]
        dz = to_list[2] - from_list[2]
        
        #if the points are very close, return 0 
        if abs(dx) < 0.001 and abs(dz) < 0.001:
            return 0.0
    
        return math.atan2(dx, dz) #returns angle in radians, 0 is facing forward (positive z), and positive angles rotate to the right (positive x)
        
    @staticmethod
    def create_rotation_quaternion(direction_radians: float) -> np.ndarray:
        #Create a quaternion for rotation around Y axis in simulator (expects quaternion in [w, x, y, z] order)
        half_angle = direction_radians / 2.0
        
        # For rotation around Y axis:
        return np.array([
            math.cos(half_angle),  # w
            0.0,                    # x
            -math.sin(half_angle),   # y (rotation around Y axis)
            0.0                     # z
        ], dtype=np.float32)