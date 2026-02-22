"""
The agent moves in a custom model
Movement logic: Hierarchical strukture (Frontier-Based Exploration (where to go next) -> A* Pathfinding (how to get there) -> Obstacle/Wall Avoidance (what to do when stuck))
The agent can look around (At start position, After reaching each frontier, After recovering from being stuck, At regular stop intervals)
Creation of a 2D/3D map 
Creation of frames from the agent's sensor
"""
import os
import habitat_sim
import magnum as mn
import numpy as np
import cv2
import math
from datetime import datetime
import matplotlib.pyplot as plt

# Setup environment for CPU rendering
os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'
os.environ['GALLIUM_DRIVER'] = 'llvmpipe'
os.environ['CUDA_VISIBLE_DEVICES'] = ''

class SimpleExplorationAgent:
    """
    Discovers environment using Frontier-Based Exploration + A* pathfinding.
    Agent stands on floor (y=0), camera at eye level (y=1.5)
    Can look around (pan/tilt) to observe surroundings
    """ 
    def __init__(self, sim):
        self.sim = sim
        self.agent = sim.get_agent(0)
        self.pathfinder = sim.pathfinder
        self.agent_height = 0.0  # Agent stands on floor
        self.sensor_height = 1.5  # Camera at eye level
        self.room_bounds = { # Room boundaries (7x7 room)
            'x_min': -3.5,
            'x_max': 3.5,
            'z_min': -3.5,
            'z_max': 3.5
        }
        #Possible improvement: manual setting of room dimensions in the application, but for now I just use these fixed bounds for the test environment
        self.safe_zone = { # Safe zone (stay away from walls); avoiding the problem of falling into walls/objects
            'x_min': -2.5,
            'x_max': 2.5,
            'z_min': -2.5,
            'z_max': 2.5
        }
        # Exploration state
        self.visited_positions = set()
        self.position_history = []  # List of [x, y, z] positions
        self.stop_points = []  # Points where agent stops for YOLO
        self.frontier_points = []  # Discovered frontiers
        
        # YOLO integration (disabled for now)
        self.yolo_stop_interval = 10
        self.steps_since_last_yolo = 0
        
        # Look-around settings
        self.look_around_enabled = True
        self.look_angles = [0, 45, 90, 135, 180, 225, 270, 315]  # Pan angles - look in 8 directions
        self.look_tilts = [-15, 0, 15]  # Tilt angles - look up/down
        
        # Store original agent rotation
        self.original_rotation = None
        
        # Coverage tracking
        self.target_coverage = 30.0 # Target percentage of safe zone to cover
        self.max_total_steps = 200 
        
        # Wall avoidance tracking
        self.near_wall_count = 0
        self.max_wall_steps = 5  # If near wall for this many steps, force move to center
        
        # Frame capture settings
        self.frame_dir = "exploration_frames_lookaround"
        self.results_dir = "results"
        os.makedirs(self.frame_dir, exist_ok=True)
        os.makedirs(self.results_dir, exist_ok=True)
        self.frame_count = 0
        
        # Set initial sensor position to eye level
        self.update_sensor_position()
        print(f"Agent initialized: on floor (y=0), camera at {self.sensor_height}m")
        print(f"Look-around: {len(self.look_angles)} pan angles * {len(self.look_tilts)} tilts = {len(self.look_angles) * len(self.look_tilts)} views per stop")
        
    def update_sensor_position(self):
        """Set sensor at eye level above agent"""
        agent_config = self.agent.agent_config
        for spec in agent_config.sensor_specifications:
            if spec.uuid == "color_sensor":
                spec.position = mn.Vector3(0, self.sensor_height, 0) # Sensor position relative to agent
                spec.orientation = mn.Vector3(0, 0, 0) # Keep sensor orientation fixed (straight ahead)
                break
    
    def set_agent_orientation(self, pan_degrees):
        """
        Rotate the agent to look in different directions
        This is more reliable than trying to rotate just the sensor
        """
        pan_rad = math.radians(pan_degrees) # Convert degrees to radians
        
        # Rotation around Y axis (up axis)
        w = math.cos(pan_rad / 2)
        y = math.sin(pan_rad / 2)
        
        # Agent uses quaternions in (w, x, y, z) order
        rotation = np.array([w, 0.0, y, 0.0], dtype=np.float32)
        
        # Apply to agent
        agent_state = self.agent.get_state()
        agent_state.rotation = rotation
        self.agent.set_state(agent_state)
    
    def tilt_camera(self, tilt_degrees):
        """
        Tilt the camera up/down by changing sensor orientation
        """
        tilt_rad = math.radians(tilt_degrees) # Convert degrees to radians
        orientation = mn.Vector3(tilt_rad, 0.0, 0.0) # Rotation around X axis for tilting up/down
        
        agent_config = self.agent.agent_config # Apply to sensor
        for spec in agent_config.sensor_specifications:
            if spec.uuid == "color_sensor":
                spec.orientation = orientation
                break
    
    def reset_agent_orientation(self):
        """Reset agent to face forward"""
        self.set_agent_orientation(0)
    
    def reset_camera_tilt(self):
        """Reset camera to level"""
        self.tilt_camera(0)
    
    def look_around_at_position(self, position_name=""):
        """
        Perform a full look-around scan at current position
        Rotates the agent for pan, tilts the camera for up/down
        """
        if not self.look_around_enabled:
            return
        
        current_pos = self.get_position_as_list(self.agent.get_state().position)
        print(f"Looking around at ({current_pos[0]:.1f}, {current_pos[2]:.1f})")
        
        original_state = self.agent.get_state() # Store original state to restore after look-around
        
        # Perform look-around
        scan_number = 0
        for tilt in self.look_tilts:
            self.tilt_camera(tilt) # Set camera tilt
            
            for pan in self.look_angles:
                self.set_agent_orientation(pan) # Rotate agent to this pan angle
                
                # Capture frame
                scan_number += 1
                filename = f"{self.frame_dir}/look_{position_name}_pos{current_pos[0]:.1f}_{current_pos[2]:.1f}_pan{pan}_tilt{tilt}.png"
                
                observations = self.sim.get_sensor_observations() # Get current frame from sensor
                frame = observations["color_sensor"]
                
                # Convert RGBA to BGR for saving; OpenCV uses BGR, so I convert to BGR for saving and visualization
                if frame.shape[2] == 4: 
                    frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
                else:
                    frame_bgr = frame
                # Later I add converting to RGB for Yolo
                
                # Add viewing direction info to image for debugging
                cv2.putText(frame_bgr, f"Position: ({current_pos[0]:.1f}, {current_pos[1]:.1f}, {current_pos[2]:.1f})", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame_bgr, f"Looking: pan={pan}, tilt={tilt}", 
                           (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                cv2.imwrite(filename, frame_bgr)
        
        # Restore original orientation
        self.agent.set_state(original_state)
        self.reset_camera_tilt()
        print(f"Completed {scan_number} look-around views")
    
    def get_position_key(self, position):
        """Create a unique identifier for each position"""
        if hasattr(position, 'x'):
            return (round(position.x, 1), round(position.y, 1), round(position.z, 1))
        elif isinstance(position, (list, tuple, np.ndarray)) and len(position) >= 3:
            return (round(position[0], 1), round(position[1], 1), round(position[2], 1))
        else:
            pos_list = list(position)
            return (round(pos_list[0], 1), round(pos_list[1], 1), round(pos_list[2], 1))
    
    def get_position_as_list(self, position):
        #Convert any position format to list [x, y, z] for pathfinding
        if hasattr(position, 'x'):
            return [position.x, position.y, position.z]
        elif isinstance(position, (list, tuple, np.ndarray)) and len(position) >= 3:
            return [float(position[0]), float(position[1]), float(position[2])]
        else:
            try:
                return [float(position[0]), float(position[1]), float(position[2])]
            except:
                return [0.0, 0.0, 0.0]
    
    def is_within_bounds(self, x, z):
        #Check if position is within room boundaries
        return (self.room_bounds['x_min'] <= x <= self.room_bounds['x_max'] and
                self.room_bounds['z_min'] <= z <= self.room_bounds['z_max'])
    
    def is_in_safe_zone(self, x, z):
        #Check if position is in safe zone (not too close to walls)
        return (self.safe_zone['x_min'] <= x <= self.safe_zone['x_max'] and
                self.safe_zone['z_min'] <= z <= self.safe_zone['z_max'])
    
    def is_near_wall(self, x, z, threshold=0.8):
        #Check if position is too close to any wall
        dist_to_left = abs(x - self.room_bounds['x_min'])
        dist_to_right = abs(x - self.room_bounds['x_max'])
        dist_to_front = abs(z - self.room_bounds['z_min'])
        dist_to_back = abs(z - self.room_bounds['z_max'])
        min_dist = min(dist_to_left, dist_to_right, dist_to_front, dist_to_back)
        return min_dist < threshold # Consider near wall if within 0.8 meters
    
    def get_wall_distance(self, x, z):
        #Get distance to nearest wall
        dist_to_left = abs(x - self.room_bounds['x_min'])
        dist_to_right = abs(x - self.room_bounds['x_max'])
        dist_to_front = abs(z - self.room_bounds['z_min'])
        dist_to_back = abs(z - self.room_bounds['z_max'])
        
        return min(dist_to_left, dist_to_right, dist_to_front, dist_to_back)
    
    def enforce_agent_height(self, position):
        #Set agent position at floor level (y=0)
        pos_list = self.get_position_as_list(position)
        pos_list[1] = self.agent_height  # Agent stands on floor
        return np.array(pos_list, dtype=np.float32)
    
    def enforce_bounds(self, position):
        #Keep agent within room bounds
        pos_list = self.get_position_as_list(position)
        pos_list[0] = max(self.room_bounds['x_min'], min(self.room_bounds['x_max'], pos_list[0])) # Keep x within bounds
        pos_list[2] = max(self.room_bounds['z_min'], min(self.room_bounds['z_max'], pos_list[2])) # Keep z within bounds
        return np.array(pos_list, dtype=np.float32)
    
    def capture_frame(self, filename=None, pan=0, tilt=0):
        #Capture current view from eye level
        observations = self.sim.get_sensor_observations()
        frame = observations["color_sensor"]
        
        # Convert RGBA to BGR for saving (for OpenCV); later I can convert to RGB for Yolo
        if frame.shape[2] == 4:
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGBA2BGR)
        else:
            frame_bgr = frame
        
        agent_pos = self.get_position_as_list(self.agent.get_state().position) # Get positions for debugging info on frame
        
        # Add information to image
        cv2.putText(frame_bgr, f"Agent: ({agent_pos[0]:.1f}, {agent_pos[1]:.1f}, {agent_pos[2]:.1f})", 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame_bgr, f"Looking: pan={pan}, tilt={tilt}", 
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame_bgr, f"Step: {len(self.position_history)}", 
                    (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Add wall distance indicator
        wall_dist = self.get_wall_distance(agent_pos[0], agent_pos[2])
        if wall_dist < 0.8:
            status = f"NEAR WALL! ({wall_dist:.2f}m)"
            color = (0, 0, 255)
        elif wall_dist < 1.5:
            status = f"Getting close to wall ({wall_dist:.2f}m)"
            color = (0, 165, 255)
        else:
            status = f"Safe from walls ({wall_dist:.2f}m)"
            color = (0, 255, 0)
        cv2.putText(frame_bgr, status, (10, 105), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        # Add safe zone indicator
        if self.is_in_safe_zone(agent_pos[0], agent_pos[2]):
            zone_text = "In safe zone"
            zone_color = (0, 255, 0)
        else:
            zone_text = "Outside safe zone!"
            zone_color = (0, 0, 255)
        cv2.putText(frame_bgr, zone_text, (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, zone_color, 2)
        
        # Generate filename if not provided
        if filename is None:
            self.frame_count += 1
            filename = f"{self.frame_dir}/frame_{self.frame_count:04d}_x{agent_pos[0]:.1f}_z{agent_pos[2]:.1f}_p{pan}_t{tilt}.png"
        
        cv2.imwrite(filename, frame_bgr)
        return frame_bgr
    
    def calculate_distance(self, pos1, pos2):
        #Calculate Euclidean distance (2D only - ignore height)
        pos1_list = self.get_position_as_list(pos1)
        pos2_list = self.get_position_as_list(pos2)
        return math.sqrt((pos1_list[0] - pos2_list[0])**2 + 
                        (pos1_list[2] - pos2_list[2])**2)
    
    def calculate_coverage(self):
        #Calculate current coverage percentage of safe zone
        safe_positions = sum(1 for pos_key in self.visited_positions 
                           if self.is_in_safe_zone(pos_key[0], pos_key[2]))
        estimated_total = 50
        return min(100.0, (safe_positions / estimated_total) * 100)
    
    def discover_frontiers(self, current_position, max_frontiers=8):
        #Discover frontiers within room bounds, prioritizing safe zone
        frontiers = []
        current_pos_list = self.get_position_as_list(current_position)
        
        # Sample points in different directions
        for angle in np.linspace(0, 2 * math.pi, 24): # Sample 24 directions around the agent
            for distance in [1.5, 2.5, 3.5, 4.5]: # 4 distances to check for frontiers
                candidate = [ # x, y, z position of candidate frontier
                    current_pos_list[0] + distance * math.cos(angle), 
                    self.agent_height,
                    current_pos_list[2] + distance * math.sin(angle)
                ]
                
                if not self.is_within_bounds(candidate[0], candidate[2]): # Skip candidates outside room bounds
                    continue
                
                candidate_vec = np.array(candidate, dtype=np.float32) # array for candidate position for pathfinding
                
                if self.pathfinder.is_navigable(candidate_vec):
                    candidate_key = self.get_position_key(candidate) # Get position key for candidate to check if visited
                    
                    if candidate_key not in self.visited_positions: # A* move to candidate to check if it's reachable and get path length
                        path = habitat_sim.ShortestPath()
                        path.requested_start = np.array(current_pos_list, dtype=np.float32)
                        path.requested_end = candidate_vec
                        
                        if self.pathfinder.find_path(path): # Each candidate gets a score based on three factors
                            wall_dist = self.get_wall_distance(candidate[0], candidate[2])
                            distance_score = distance / 5.0 # Closer frontiers get higher score (normalized by max distance) 
                            # Safety score based on distance to walls
                            if wall_dist > 2.0:
                                safe_bonus = 10.0
                            elif wall_dist > 1.5:
                                safe_bonus = 5.0
                            elif wall_dist > 1.0:
                                safe_bonus = 2.0
                            else:
                                safe_bonus = -5.0
                            
                            novelty_bonus = self.get_exploration_bonus(candidate, current_pos_list) # Checks if agent has been in this direction recently
                            total_score = distance_score + safe_bonus + novelty_bonus
                            frontiers.append((candidate, total_score))
        
        frontiers.sort(key=lambda x: x[1], reverse=True)
        
        unique_frontiers = []
        seen_keys = set()
        for f in frontiers:
            key = self.get_position_key(f[0])
            if key not in seen_keys:
                seen_keys.add(key)
                unique_frontiers.append(f[0])
                if len(unique_frontiers) >= max_frontiers:
                    break
        
        return unique_frontiers
    
    def get_exploration_bonus(self, candidate, current_pos):
        #Give bonus to frontiers in less explored directions
        if len(self.position_history) < 10:
            return 2.0
        
        dx = candidate[0] - current_pos[0]
        dz = candidate[2] - current_pos[2]
        magnitude = math.sqrt(dx*dx + dz*dz)
        
        if magnitude < 0.01:
            return 0.0
        
        dx_norm = dx / magnitude
        dz_norm = dz / magnitude
        
        recent = self.position_history[-15:] if len(self.position_history) > 15 else self.position_history
        
        direction_count = 0
        for pos in recent:
            pdx = pos[0] - current_pos[0]
            pdz = pos[2] - current_pos[2]
            pmag = math.sqrt(pdx*pdx + pdz*pdz)
            
            if pmag > 0.2:
                pdx_norm = pdx / pmag
                pdz_norm = pdz / pmag
                
                similarity = dx_norm * pdx_norm + dz_norm * pdz_norm
                if similarity > 0.7:
                    direction_count += 1
        
        return max(1.0, 5.0 - direction_count)
    
    def navigate_to_point(self, target_point, max_steps=20):
        """
        Use A* to navigate to target point. Agent has a A* algorithm in his initial capabilities, so I can use that to 
        move towards frontiers. Capture frames during navigation and check for wall proximity to avoid 
        getting stuck.
        """
        current_pos = self.agent.get_state().position
        current_pos_list = self.get_position_as_list(current_pos) 
        # Convert to list for easier manipulation and to ensure it's in the right format for pathfinding
        target_list = self.get_position_as_list(target_point) 
        target_list[1] = self.agent_height
        target_vec = np.array(target_list, dtype=np.float32)
        target_vec = self.enforce_bounds(target_vec) # Ensure target is within bounds
        
        path = habitat_sim.ShortestPath() # A*
        path.requested_start = np.array(current_pos_list, dtype=np.float32) 
        path.requested_end = target_vec
        
        if not self.pathfinder.find_path(path):
            return False
        
        path_points = list(path.points)
        
        if len(path_points) < 2:
            return False
        
        steps_taken = 0
        
        for i, point in enumerate(path_points[1:], 1):
            if steps_taken >= max_steps: # Limit steps to prevent getting stuck in long paths
                break
            
            point_np = np.array(point, dtype=np.float32)
            point_np[1] = self.agent_height
            point_np = self.enforce_bounds(point_np)
            
            agent_state = self.agent.get_state()
            agent_state.position = point_np
            self.agent.set_state(agent_state) # Move agent to next point in path
            
            point_list = self.get_position_as_list(point_np)
            self.position_history.append(point_list)
            self.visited_positions.add(self.get_position_key(point_list))
            
            if self.is_near_wall(point_list[0], point_list[2]): # Check if near wall and track how many steps in a row
                self.near_wall_count += 1
            else:
                self.near_wall_count = 0
            
            # Capture forward-facing frame during navigation
            if steps_taken % 2 == 0:
                self.capture_frame(pan=0, tilt=0)
            
            steps_taken += 1
            self.steps_since_last_yolo += 1
            
            if self.steps_since_last_yolo >= self.yolo_stop_interval:
                self.process_stop_point() # Record stop point and perform look-around
                self.steps_since_last_yolo = 0 # Reset YOLO step counter
            
            if self.calculate_distance(point_list, target_list) < 0.8: # If we're close enough to the target, consider it reached
                return True
        
        final_pos = self.get_position_as_list(self.agent.get_state().position)
        return self.calculate_distance(final_pos, target_list) < 1.0
    
    def process_stop_point(self):
        """Record a stop point and perform look-around scan"""
        current_pos = self.agent.get_state().position
        current_pos_list = self.get_position_as_list(current_pos)
        
        self.stop_points.append(current_pos_list)
        
        # Perform look-around scan at this stop point
        stop_number = len(self.stop_points)
        self.look_around_at_position(position_name=f"stop{stop_number:02d}")
        
        # Also capture a standard forward-facing frame
        self.capture_frame(f"{self.frame_dir}/stop_{stop_number:04d}_forward.png", pan=0, tilt=0)
    
    def explore_room(self):
        """
        Main exploration algorithm with look-around capability : Frontier-Based Exploration
        """
        print(f"\nStarting Exploration with Look-Around")
        print(f"Agent: on floor (y=0)")
        print(f"Camera: {self.sensor_height}m above floor (eye level)")
        print(f"Look-around: {len(self.look_angles)} pan angles * {len(self.look_tilts)} tilts = {len(self.look_angles) * len(self.look_tilts)} views per stop")
        print(f"Room: 7x7 meters")
        print(f"Safe zone: |x|≤2.5, |z|≤2.5 (away from walls)")
        
        # Start at center of room
        start_point = np.array([0.0, self.agent_height, 0.0], dtype=np.float32)
        
        agent_state = self.agent.get_state()
        agent_state.position = start_point
        agent_state.rotation = np.array([1.0, 0.0, 0.0, 0.0])
        self.agent.set_state(agent_state)
        
        # Initialize tracking
        start_list = [0.0, self.agent_height, 0.0]
        self.position_history = [start_list]
        self.visited_positions.add(self.get_position_key(start_point))
        self.stop_points = [start_list]
        
        # Perform initial look-around at start position
        print(f"\nInitial look-around at start position:")
        self.look_around_at_position(position_name="start")
        
        step_count = 0
        failures = 0 # Count failures to reach frontiers to trigger local exploration or recovery behavior
        max_failures = 8
        
        # Store all discovered frontiers for visualization
        all_frontiers = []
        
        while (len(self.position_history) < self.max_total_steps and 
               self.calculate_coverage() < self.target_coverage and 
               failures < max_failures):
            
            current_position = self.agent.get_state().position
            current_pos_list = self.get_position_as_list(current_position)
            step_count += 1
            current_coverage = self.calculate_coverage()
            
            print(f"\nStep {step_count}:")
            print(f"Coverage: {current_coverage:.1f}% ({len(self.visited_positions)} unique positions)")
            print(f"Position: ({current_pos_list[0]:.2f}, {current_pos_list[2]:.2f})")
            print(f"Wall distance: {self.get_wall_distance(current_pos_list[0], current_pos_list[2]):.2f}m")
            
            # Check if stuck near wall
            if self.near_wall_count >= self.max_wall_steps:
                print(f"Stuck near wall! Moving to center")
                center_point = [0.0, self.agent_height, 0.0]
                self.navigate_to_point(center_point, max_steps=10)
                self.near_wall_count = 0
                
                # Look around after moving to center
                self.look_around_at_position(position_name="center_recovery")
                continue
            
            # Discover new frontiers
            frontiers = self.discover_frontiers(current_position)
            all_frontiers.extend(frontiers)
            
            if frontiers:
                # Filter frontiers that are in safe zone first
                safe_frontiers = [f for f in frontiers if self.is_in_safe_zone(f[0], f[2])]
                
                if safe_frontiers:
                    next_target = safe_frontiers[0]
                    print(f"Found {len(safe_frontiers)} safe frontiers")
                else:
                    frontiers_with_dist = [(f, self.get_wall_distance(f[0], f[2])) for f in frontiers]
                    frontiers_with_dist.sort(key=lambda x: x[1], reverse=True)
                    next_target = frontiers_with_dist[0][0]
                    print(f"No safe frontiers, picking farthest from walls")
                
                print(f"Moving to frontier at ({next_target[0]:.1f}, {next_target[2]:.1f})")
                
                success = self.navigate_to_point(next_target)
                
                if success:
                    print(f"Reached frontier area")
                    failures = 0
                    self.look_around_at_position(position_name=f"frontier{step_count}") # Look around at the new position
                else:
                    print(f"Failed to reach frontier")
                    failures += 1
            else:
                print(f"No frontiers, performing obstacle-aware local exploration")
                self.obstacle_aware_local_exploration(5)
                failures += 1
        
        self.frontier_points = all_frontiers # Store all frontiers for visualization
        self.generate_visualizations() # Exploration complete - generate visualizations
        
        return self.position_history, self.stop_points
    
    def obstacle_aware_local_exploration(self, num_steps):
        #Local exploration that actively avoids obstacles and walls
        for i in range(num_steps):
            current_pos = self.get_position_as_list(self.agent.get_state().position)
            
            if self.is_near_wall(current_pos[0], current_pos[2]):
                # Determine which wall agent is near and turn away
                dist_to_left = abs(current_pos[0] - self.room_bounds['x_min'])
                dist_to_right = abs(current_pos[0] - self.room_bounds['x_max'])
                dist_to_front = abs(current_pos[2] - self.room_bounds['z_min'])
                dist_to_back = abs(current_pos[2] - self.room_bounds['z_max'])
                
                min_dist = min(dist_to_left, dist_to_right, dist_to_front, dist_to_back) # Find the closest wall
                
                # Turn away from the closest wall
                if min_dist == dist_to_left:
                    for _ in range(3):
                        self.sim.step("turn_right")
                elif min_dist == dist_to_right:
                    for _ in range(3):
                        self.sim.step("turn_left")
                elif min_dist == dist_to_front:
                    for _ in range(4):
                        self.sim.step("turn_left")
                elif min_dist == dist_to_back:
                    for _ in range(4):
                        self.sim.step("turn_left")
                
                self.sim.step("move_forward")
            else: # If not near wall, move in a pattern to explore locally
                if i % 3 == 0:
                    self.sim.step("move_forward")
                else:
                    if i % 2 == 0:
                        self.sim.step("turn_left")
                    else:
                        self.sim.step("turn_right")
            
            # Use bounds
            new_state = self.agent.get_state()
            new_state.position = self.enforce_bounds(new_state.position) # Keep agent within room bounds
            new_state.position[1] = self.agent_height
            self.agent.set_state(new_state)
            
            # Record position
            new_pos = new_state.position
            new_pos_list = self.get_position_as_list(new_pos)
            self.position_history.append(new_pos_list)
            self.visited_positions.add(self.get_position_key(new_pos))
            
            if self.is_near_wall(new_pos_list[0], new_pos_list[2]):
                self.near_wall_count += 1
            else:
                self.near_wall_count = 0
            
            if i % 2 == 0:
                self.capture_frame(pan=0, tilt=0)
    
    def generate_visualizations(self):
        #Generate 2D and 3D visualizations of the exploration path
        print("Visulization of Exploration Results")
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create 3D visualization
        self.create_3d_path_visualization(
            self.position_history,
            self.stop_points,
            self.frontier_points,
            f"exploration_{timestamp}"
        )
        
        # Create 2D map
        self.create_2d_map(
            self.position_history,
            self.stop_points,
            self.frontier_points,
            f"exploration_{timestamp}"
        )
        
        print(f"Visualizations saved in '{self.results_dir}'")
        
        # Print summary statistics
        print(f"\nExploration Summary:")
        print(f"Total steps: {len(self.position_history)}")
        print(f"Unique positions: {len(self.visited_positions)}")
        print(f"YOLO stop points: {len(self.stop_points)}")
        print(f"Frontiers discovered: {len(self.frontier_points)}")
        print(f"Final coverage: {self.calculate_coverage():.1f}%")
        print(f"Look-around views per stop: {len(self.look_angles) * len(self.look_tilts)}")
    
    def create_3d_path_visualization(self, path_points, stop_points, frontier_points, filename):
        #Create a 3D visualization of the path with stops and frontiers
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        x_coords = [point[0] for point in path_points]
        y_coords = [point[1] for point in path_points]
        z_coords = [point[2] for point in path_points]
        
        # Color path points based on wall distance
        colors = []
        for x, z in zip(x_coords, z_coords):
            wall_dist = self.get_wall_distance(x, z)
            if wall_dist > 2.0:
                colors.append('green')
            elif wall_dist > 1.0:
                colors.append('yellow')
            else:
                colors.append('red')
        
        # Plot the path
        ax.scatter(x_coords, z_coords, y_coords, c=colors, s=30, alpha=0.6)
        ax.plot(x_coords, z_coords, y_coords, 'b-', linewidth=1, alpha=0.3)
        
        # Mark start and end
        ax.scatter(x_coords[0], z_coords[0], y_coords[0], 
                   c='green', s=200, marker='D', label='Start', edgecolors='black')
        ax.scatter(x_coords[-1], z_coords[-1], y_coords[-1], 
                   c='orange', s=200, marker='D', label='End', edgecolors='black')
        
        # Mark stop points (YOLO stops)
        if stop_points:
            stop_x = [point[0] for point in stop_points]
            stop_z = [point[2] for point in stop_points]
            stop_y = [point[1] for point in stop_points]
            ax.scatter(stop_x, stop_z, stop_y, c='yellow', s=120, marker='s', 
                      label='Look-around Stops', edgecolors='black', alpha=0.8)
        
        # Mark frontier points
        if frontier_points:
            sample_rate = max(1, len(frontier_points) // 30)
            sampled_frontiers = frontier_points[::sample_rate]
            
            frontier_x = [point[0] for point in sampled_frontiers]
            frontier_z = [point[2] for point in sampled_frontiers]
            frontier_y = [point[1] for point in sampled_frontiers]
            ax.scatter(frontier_x, frontier_z, frontier_y, c='purple', s=80, marker='*', 
                      label='Frontiers', edgecolors='black', alpha=0.7)
        
        # Draw room boundaries
        room_x = [-3.5, 3.5, 3.5, -3.5, -3.5]
        room_z = [-3.5, -3.5, 3.5, 3.5, -3.5]
        room_y = [0, 0, 0, 0, 0]
        ax.plot(room_x, room_z, room_y, 'k--', alpha=0.3, label='Room Boundary')
        
        # Draw safe zone
        safe_x = [-2.5, 2.5, 2.5, -2.5, -2.5]
        safe_z = [-2.5, -2.5, 2.5, 2.5, -2.5]
        safe_y = [0, 0, 0, 0, 0]
        ax.plot(safe_x, safe_z, safe_y, 'g--', alpha=0.3, label='Safe Zone')
        
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Z Position (m)')
        ax.set_zlabel('Y Position (m)')
        ax.set_title(f'Exploration with Look-Around\n{len(path_points)} steps, {len(stop_points)} stops')
        ax.legend()
        
        # Save from different angles
        for angle in [30, 60, 90]:
            ax.view_init(elev=20, azim=angle)
            plt.tight_layout()
            plt.savefig(f'{self.results_dir}/{filename}_3d_angle_{angle}.png', 
                       dpi=150, bbox_inches='tight')
        
        plt.close()
    
    def create_2d_map(self, path_points, stop_points, frontier_points, filename):
        #Create a 2D top-down map with look-around stops
        fig, ax = plt.subplots(figsize=(12, 10))
        
        x_coords = [point[0] for point in path_points]
        z_coords = [point[2] for point in path_points]
        
        # Color path points based on wall distance
        colors = []
        for x, z in zip(x_coords, z_coords):
            wall_dist = self.get_wall_distance(x, z)
            if wall_dist > 2.0:
                colors.append('green')
            elif wall_dist > 1.0:
                colors.append('yellow')
            else:
                colors.append('red')
        
        # Plot the path with colored points
        ax.scatter(x_coords, z_coords, c=colors, s=30, alpha=0.6)
        ax.plot(x_coords, z_coords, 'b-', linewidth=1, alpha=0.3)
        
        # Mark start and end
        ax.scatter(x_coords[0], z_coords[0], c='green', s=150, marker='D', 
                   label='Start', edgecolors='black', zorder=5)
        ax.scatter(x_coords[-1], z_coords[-1], c='orange', s=150, marker='D', 
                   label='End', edgecolors='black', zorder=5)
        
        # Mark stop points (with look-around)
        if stop_points:
            stop_x = [point[0] for point in stop_points]
            stop_z = [point[2] for point in stop_points]
            
            # Draw larger circles for look-around stops
            ax.scatter(stop_x, stop_z, c='yellow', s=200, marker='o', 
                      label='Look-around Stops', edgecolors='black', alpha=0.5, zorder=3)
            ax.scatter(stop_x, stop_z, c='yellow', s=100, marker='s', 
                      edgecolors='black', alpha=0.9, zorder=4)
            
            # Add numbers to identify stops
            for i, (x, z) in enumerate(zip(stop_x, stop_z)):
                ax.text(x, z, str(i), fontsize=9, fontweight='bold', 
                       ha='center', va='center', color='black')
        
        # Mark frontier points
        if frontier_points:
            sample_rate = max(1, len(frontier_points) // 20)
            sampled_frontiers = frontier_points[::sample_rate]
            
            frontier_x = [point[0] for point in sampled_frontiers]
            frontier_z = [point[2] for point in sampled_frontiers]
            ax.scatter(frontier_x, frontier_z, c='purple', s=60, marker='*', 
                      label='Frontiers', edgecolors='black', alpha=0.8, zorder=3)
        
        # Draw room boundaries
        ax.plot([-3.5, 3.5, 3.5, -3.5, -3.5], 
                [-3.5, -3.5, 3.5, 3.5, -3.5], 
                'k-', alpha=0.5, label='Room Boundary', linewidth=2)
        
        # Draw safe zone
        ax.plot([-2.5, 2.5, 2.5, -2.5, -2.5], 
                [-2.5, -2.5, 2.5, 2.5, -2.5], 
                'g--', alpha=0.5, label='Safe Zone', linewidth=2)
        
        # Add heat map of wall proximity
        grid_x = np.linspace(-3.5, 3.5, 50)
        grid_z = np.linspace(-3.5, 3.5, 50)
        X, Z = np.meshgrid(grid_x, grid_z)
        
        wall_dist = np.zeros_like(X)
        for i in range(len(grid_x)):
            for j in range(len(grid_z)):
                wall_dist[j, i] = self.get_wall_distance(X[j, i], Z[j, i])
        
        ax.contourf(X, Z, wall_dist, levels=10, alpha=0.1, cmap='RdYlGn')
        
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Z Position (m)')
        ax.set_title(f'Exploration Map with Look-Around Stops (numbered)\n'
                    f'{len(path_points)} steps, {len(stop_points)} stops, {len(self.look_angles) * len(self.look_tilts)} views/stop')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        ax.set_xlim(-4, 4)
        ax.set_ylim(-4, 4)
        
        plt.tight_layout()
        plt.savefig(f'{self.results_dir}/{filename}_2d_map.png', dpi=150, bbox_inches='tight')
        plt.close()

def setup_habitat_sim(scene_path):
    #Setup Habitat simulator with eye-level camera
    sim_config = habitat_sim.SimulatorConfiguration()
    sim_config.enable_physics = True
    sim_config.gpu_device_id = -1
    sim_config.scene_id = scene_path
    
    # Sensor at eye level
    sensor_spec = habitat_sim.CameraSensorSpec()
    sensor_spec.uuid = "color_sensor"
    sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
    sensor_spec.resolution = [640, 480]
    sensor_spec.position = mn.Vector3(0, 1.5, 0) # 1.5m above floor for eye level
    sensor_spec.orientation = mn.Vector3(0, 0, 0)  # Straight ahead initially
    agent_config = habitat_sim.AgentConfiguration()
    agent_config.sensor_specifications = [sensor_spec] # Use the same sensor spec for the agent configuration
    
    cfg = habitat_sim.Configuration(sim_config, [agent_config])
    
    sim = habitat_sim.Simulator(cfg)
    print("Simulator created")
    
    return sim

scene_path = "data/scene_datasets/myModelRoom/room_1.glb"

if not os.path.exists(scene_path):
    print(f"Scene not found at {scene_path}")

sim = setup_habitat_sim(scene_path) # Create simulator with custom model and eye-level camera

# Create and run exploration agent with look-around
agent = SimpleExplorationAgent(sim)
history, stop_points = agent.explore_room()

print(f"\nExploration complete! Check these directories:")
print(f"Frames: '{agent.frame_dir}/' (includes look-around views)")
print(f"Visualizations: '{agent.results_dir}/'")

sim.close() # Cleanup
print("\nSimulator closed")