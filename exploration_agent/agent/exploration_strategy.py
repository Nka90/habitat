"""
Frontier-based exploration strategy
"""
import math
import random
import numpy as np
import habitat_sim
from typing import List, Tuple, Set, Optional
from config.settings import PathConfig
from environment.path_utils import PathUtils
from agent.movement_controller import MovementController


class FrontierExplorer:
    def __init__(self, 
                 pathfinder,
                 movement_controller: MovementController,
                 path_config: PathConfig):
        
        self.pathfinder = pathfinder
        self.movement = movement_controller
        self.config = path_config
        self.visited_positions: Set[Tuple] = set()
        self.position_history: List[List[float]] = []
    
    def add_visited_position(self, position):
        #Add a position to visited set and history
        pos_list = PathUtils.get_position_as_list(position)
        self.visited_positions.add(PathUtils.get_position_key(pos_list))
        self.position_history.append(pos_list)
        
    def get_exploration_bonus(self, candidate, current_pos) -> float:
        #Calculate bonus for exploring new directions
        if len(self.position_history) < 10:
            return 2.0
        
        # Calculate direction to candidate
        dx = candidate[0] - current_pos[0]
        dz = candidate[2] - current_pos[2]
        magnitude = math.sqrt(dx*dx + dz*dz)
        
        if magnitude < 0.1:
            return 0.0
        
        # Normalize direction
        direction = np.array([dx, dz]) / magnitude
        
        # Check recent movement directions
        recent_positions = self.position_history[-15:]
        similar_directions = 0
        
        for pos in recent_positions:
            pdx = pos[0] - current_pos[0]
            pdz = pos[2] - current_pos[2]
            pmag = math.sqrt(pdx*pdx + pdz*pdz)
            
            if pmag > 0.2:
                prev_dir = np.array([pdx, pdz]) / pmag
                similarity = np.dot(direction, prev_dir)
                if similarity > 0.7:
                    similar_directions += 1
        
        # Higher bonus for less explored directions
        return max(1.0, 5.0 - similar_directions)
    
    
    def discover_frontiers(self, current_position: List[float]) -> List[List[float]]:
        #discover frontiers around current position
        frontiers = []
        current_list = PathUtils.get_position_as_list(current_position)
        
        # Sample points in different directions
        for angle in np.linspace(0, 2 * math.pi, 24):
            # Add slight randomness to angles
            varied_angle = angle + random.uniform(-0.1, 0.1)
            
            for distance in [1.5, 2.5, 3.5, 4.5]:
                # Add randomness to distances
                varied_distance = distance + random.uniform(-0.2, 0.2)
                
                candidate = [
                    current_list[0] + varied_distance * math.cos(varied_angle), #x coord
                    self.movement.agent_config.agent_height,
                    current_list[2] + varied_distance * math.sin(varied_angle) #z coord
                ]
                
                candidate_vec = np.array(candidate, dtype=np.float32)
                
                #check if candidate is navigable and not visited
                if self.pathfinder.is_navigable(candidate_vec):
                    candidate_key = PathUtils.get_position_key(candidate)
                    
                    #calculate shortest path to give bonus based on distance
                    if candidate_key not in self.visited_positions:
                        path = habitat_sim.ShortestPath()
                        path.requested_start = np.array(current_list, dtype=np.float32)
                        path.requested_end = candidate_vec
                        
                        if self.pathfinder.find_path(path):
                            # Calculate score with randomness
                            distance_score = varied_distance / 5.0
                            novelty_bonus = self.get_exploration_bonus(candidate, current_list)
                            #randomness = random.uniform(-1.0, 1.0) 
                            
                            #total_score = distance_score + novelty_bonus + randomness
                            total_score = distance_score + novelty_bonus
                            frontiers.append((candidate, total_score))
        
        frontiers.sort(key=lambda x: x[1], reverse=True)
        
        # Select unique frontiers
        unique_frontiers = []
        seen_keys = set()
        
        #unique key for each frontier to avoid duplicates
        for f in frontiers:
            key = PathUtils.get_position_key(f[0])
            if key not in seen_keys:
                seen_keys.add(key)
                unique_frontiers.append(f[0])
                if len(unique_frontiers) >= self.config.max_frontiers:
                    break
        
        return unique_frontiers

    def select_next_target(self, frontiers) -> Optional[List[float]]:
        #Select next target frontier to move to, prioritizing with better scores
        if not frontiers:
            return None
        return frontiers[0]

    
    def find_path_to_target(self, start_pos: List[float], target_pos: List[float]) -> Optional[List[List[float]]]:
        try:
            # Convert to numpy
            start_vec = np.array(start_pos, dtype=np.float32)
            target_vec = np.array(target_pos, dtype=np.float32)
                
            if not self.pathfinder.is_navigable(start_vec):
                print("Start point is not navigable on NavMesh")
                return None
            else:
                start_vec = self.pathfinder.snap_point(start_vec)
            
            if not self.pathfinder.is_navigable(target_vec):
                print("Target point is not navigable on NavMesh")
                return None
            else:
                target_vec = self.pathfinder.snap_point(target_vec)

            # Build shortest path
            path = habitat_sim.ShortestPath()
            path.requested_start = start_vec
            path.requested_end = target_vec

            if self.pathfinder.find_path(path) and len(path.points) > 0:
                waypoints = []
                for p in path.points:
                    p = p.tolist() if hasattr(p, "tolist") else list(p)
                    waypoints.append(p[:3])
                return waypoints

            print("No path found between points")
            return None

        except Exception as e:
            print(f"Error in pathfinding: {e}")
            return None
        
    def calculate_final_path(self, agent_pos_at_detection: List[float], obj_pos: List[float]) -> Optional[List[List[float]]]:
        #calculate final path between agent and detected object
        try:
            path = self.find_path_to_target(agent_pos_at_detection, obj_pos)           
            if path and len(path) > 1:
                # Calculate path distance
                total_distance = 0.0
                for i in range(len(path)-1):
                    p1 = np.array(path[i][:3])
                    p2 = np.array(path[i+1][:3])
                    total_distance += float(np.linalg.norm(p2 - p1))          
                    object_path = {
                        'start': agent_pos_at_detection[:3],
                        'target': obj_pos[:3],
                        'waypoints': path,
                        'length': len(path),
                        'distance': total_distance
                    }
                    print(f"Path found {object_path['distance']:.2f}m long")
            else:
                print(f"No navigable path found from detection point to object!")
                object_path = None
        except Exception as e:
            print(f"  Error calculating path: {e}")
            object_path = None
        return object_path