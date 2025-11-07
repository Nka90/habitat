import os
os.environ.update({
    'LIBGL_ALWAYS_SOFTWARE': '1',
    'HABITAT_SIM_HEADLESS': '1',
    'PYOPENGL_PLATFORM': 'osmesa'
})

import habitat_sim
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import math
import random

def create_3d_path_visualization(path_points, stop_points, frontier_points, filename):
    """Create a 3D visualization of the path with stops and frontiers"""
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(14, 10)) # Creating a figure
    ax = fig.add_subplot(111, projection='3d')
    
    x_coords = [point[0] for point in path_points]
    y_coords = [point[1] for point in path_points]
    z_coords = [point[2] for point in path_points]
    
    # Plot the path
    ax.plot(x_coords, z_coords, y_coords, 'b-', linewidth=2, alpha=0.7, label='Exploration Path')
    ax.scatter(x_coords, z_coords, y_coords, c='red', s=30, alpha=0.6)
    
    # Mark start and end
    ax.scatter(x_coords[0], z_coords[0], y_coords[0], 
               c='green', s=200, marker='D', label='Start', edgecolors='black')
    ax.scatter(x_coords[-1], z_coords[-1], y_coords[-1], 
               c='orange', s=200, marker='D', label='End', edgecolors='black')
    
    # Mark stop points for YOLO
    if stop_points:
        stop_x = [point[0] for point in stop_points]
        stop_z = [point[2] for point in stop_points]
        stop_y = [point[1] for point in stop_points]
        ax.scatter(stop_x, stop_z, stop_y, c='yellow', s=120, marker='s', 
                  label='YOLO Stops', edgecolors='black', alpha=0.8)
    
    # Mark frontier points (points discovered)
    if frontier_points:
        frontier_x = [point[0] for point in frontier_points]
        frontier_z = [point[2] for point in frontier_points]
        frontier_y = [point[1] for point in frontier_points]
        ax.scatter(frontier_x, frontier_z, frontier_y, c='purple', s=80, marker='*', 
                  label='Frontiers', edgecolors='black', alpha=0.7)
    
    ax.set_xlabel('X Position')
    ax.set_ylabel('Z Position')
    ax.set_zlabel('Y Position (Height)')
    ax.set_title(f'Frontier Exploration + A*\n{len(path_points)} steps, {len(stop_points)} YOLO stops')
    ax.legend()
    
    for angle in [30, 60, 90]: # Different viewing angles of the path
        ax.view_init(elev=20, azim=angle)
        plt.tight_layout()
        plt.savefig(f'results/{filename}_angle_{angle}.png', dpi=150, bbox_inches='tight')
    
    plt.close()

def create_2d_map(path_points, stop_points, frontier_points, filename):
    """Create a 2D top-down map"""
    fig, ax = plt.subplots(figsize=(12, 10))
    
    x_coords = [point[0] for point in path_points]
    z_coords = [point[2] for point in path_points]
    
    ax.plot(x_coords, z_coords, 'b-', linewidth=1.5, alpha=0.6, label='Path')
    ax.scatter(x_coords, z_coords, c='red', s=20, alpha=0.4)
    
    ax.scatter(x_coords[0], z_coords[0], c='green', s=150, marker='D', 
               label='Start', edgecolors='black')
    ax.scatter(x_coords[-1], z_coords[-1], c='orange', s=150, marker='D', 
               label='End', edgecolors='black')
    
    if stop_points:
        stop_x = [point[0] for point in stop_points]
        stop_z = [point[2] for point in stop_points]
        ax.scatter(stop_x, stop_z, c='yellow', s=100, marker='s', 
                  label='YOLO Stops', edgecolors='black')
        
        for i, (x, z) in enumerate(zip(stop_x, stop_z)):
            ax.text(x, z, f'Y{i}', fontsize=8, fontweight='bold', 
                   ha='center', va='center',
                   bbox=dict(boxstyle="circle,pad=0.2", facecolor="yellow", alpha=0.7))
    
    if frontier_points:
        frontier_x = [point[0] for point in frontier_points]
        frontier_z = [point[2] for point in frontier_points]
        ax.scatter(frontier_x, frontier_z, c='purple', s=60, marker='*', 
                  label='Frontiers', edgecolors='black', alpha=0.8)
    
    ax.set_xlabel('X Position')
    ax.set_ylabel('Z Position')
    ax.set_title(f'Exploration Map\n{len(path_points)} steps, {len(stop_points)} YOLO stops')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    plt.tight_layout()
    plt.savefig(f'results/{filename}_2d_map.png', dpi=150, bbox_inches='tight')
    plt.close()

class SimpleExplorationAgent:
    """
    Discovers environment without sensors using Frontier-Based Exploration + A* pathfinding.
    Habitat agent use A* to navigate from A to B on the base.
    We don't know the point B, because it will be the subject the agent is looking for using YOLO.
    I use additionaly Frontier-Based algorithm to discover new areas and find the object.
    Frontier-Based algorithm for explore the environment and find the point B.
    A* algorithm for pathfinding to target point.
    """
    
    def __init__(self, sim):
        self.sim = sim # Habitat simulator
        self.agent = sim.agents[0] # Single agent
        self.pathfinder = sim.pathfinder # Pathfinding module
        
        # Exploration state
        self.visited_positions = set() # Track unique positions
        self.position_history = [] # Full path history
        self.stop_points = [] # stop points for YOLO
        self.frontier_points = [] # discovered frontiers
        
        # YOLO integration setup
        self.yolo_stop_interval = 10 # Stop every 10 steps
        self.steps_since_last_yolo = 0 #number of steps since last stop
        
        self.target_object = None # Target object to find
        self.object_found = False # Whether object has been found
        
        # Coverage tracking
        self.target_coverage = 30.0  # Target 30% coverage (optimal for testing, because no actual YOLO and the model is big)
        self.max_total_steps = 200   # Increased for better coverage
        #CAN BE ADJUSTED BASED ON ENVIRONMENT SIZE
        
    def get_position_key(self, position): 
        """
        Create a unique identifier for each position to track visited locations
        Python sets can only store hashable objects, so we convert positions to tuples for the visited_positions set
        """
        if hasattr(position, 'x'): 
            return (round(position.x, 1), round(position.y, 1), round(position.z, 1)) #Rounds coordinates to 1 decimal place and creates a tuple
        elif isinstance(position, (list, tuple, np.ndarray)) and len(position) >= 3:
            return (round(position[0], 1), round(position[1], 1), round(position[2], 1))
        else:
            # Fallback - try to convert to list
            pos_list = list(position)
            return (round(pos_list[0], 1), round(pos_list[1], 1), round(pos_list[2], 1))
    
    def get_position_as_list(self, position):
        """Convert any position format to list [x, y, z]"""
        if hasattr(position, 'x'):
            return [position.x, position.y, position.z]
        elif isinstance(position, (list, tuple, np.ndarray)) and len(position) >= 3:
            return [float(position[0]), float(position[1]), float(position[2])]
        else:
            # Try to convert
            try:
                return [float(position[0]), float(position[1]), float(position[2])]
            except:
                print(f"Warning: Could not convert position: {position}")
                return [0.0, 0.0, 0.0]
    
    def calculate_distance(self, pos1, pos2):
        """Calculate Euclidean distance - handles all position formats"""
        pos1_list = self.get_position_as_list(pos1)
        pos2_list = self.get_position_as_list(pos2)
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(pos1_list, pos2_list)))
    
    def calculate_coverage(self):
        """Calculate current coverage percentage"""
        # This is a simplified coverage estimation
        unique_positions = len(self.visited_positions)
        return min(100.0, (unique_positions / 120.0) * 100)  # Adjusted for better 30% target
    
    # the base for Frontier-Based Exploration. Finds boundary points between explored and unexplored areas
    def discover_frontiers(self, current_position, max_frontiers=8):  # Increased for better coverage
        """
        Simple frontier discovery - find unexplored areas

        Algorithm steps:
        -Sample random points around current position (30 samples)
        -Check navigability - point must be reachable
        -Check novelty - point must not be in visited set
        -Score candidates - closer points and novel directions get higher scores
        -Return top candidates - best 8 frontiers
        """
        frontiers = []
        current_pos_list = self.get_position_as_list(current_position)
        
        #Sample random points around current position (30 samples)
        for _ in range(30):  # Increased samples for better coverage
            # Generate random direction and distance
            angle = random.uniform(0, 2 * math.pi)
            distance = random.uniform(4.0, 10.0)  # Increased range for better exploration
            
            # Calculate candidate point
            candidate = [
                current_pos_list[0] + distance * math.cos(angle),
                current_pos_list[1],  # Keep same height
                current_pos_list[2] + distance * math.sin(angle)
            ]
            
            # Convert to numpy array for habitat_sim
            candidate_vec = np.array(candidate, dtype=np.float32)
            
            # Check if navigable and not visited
            if self.pathfinder.is_navigable(candidate_vec):
                candidate_key = self.get_position_key(candidate)
                if candidate_key not in self.visited_positions:
                    # Score by distance and area coverage potential
                    distance_score = 1.0 / (1.0 + self.calculate_distance(current_pos_list, candidate))
                    # Add bonus for directions we haven't explored much
                    exploration_bonus = self.get_exploration_bonus(candidate, current_pos_list)
                    total_score = distance_score + exploration_bonus
                    frontiers.append((candidate, total_score))
        
        # Return top frontiers
        frontiers.sort(key=lambda x: x[1], reverse=True)
        return [point for point, score in frontiers[:max_frontiers]]
    
    def get_exploration_bonus(self, candidate, current_pos):
        """
        Give bonus to frontiers that expand exploration in new directions. 
        Encourages exploration in new directions to maximize coverage
        Less explored directions get higher scores (up to 1.0)
        """
        if not self.position_history:
            return 1.0
        
        # Calculate the main direction of this candidate
        direction_x = candidate[0] - current_pos[0]
        direction_z = candidate[2] - current_pos[2]
        
        # Normalize
        magnitude = math.sqrt(direction_x**2 + direction_z**2)
        if magnitude > 0:
            direction_x /= magnitude
            direction_z /= magnitude
        
        # Check if we've explored much in this direction
        direction_count = 0
        for pos in self.position_history[-20:]:  # Check recent positions
            vec_x = pos[0] - current_pos[0]
            vec_z = pos[2] - current_pos[2]
            vec_mag = math.sqrt(vec_x**2 + vec_z**2)
            if vec_mag > 0:
                vec_x /= vec_mag
                vec_z /= vec_mag
                # Dot product to check similarity
                similarity = direction_x * vec_x + direction_z * vec_z
                if similarity > 0.7:  # Similar direction
                    direction_count += 1
        
        # Higher bonus for less explored directions
        return max(0.0, 1.0 - (direction_count / 20.0))
    
    def navigate_to_point(self, target_point, max_steps=20):  # Increased steps for longer paths
        """
        Use A* to navigate to target point
        """
        current_pos = self.agent.get_state().position #actual position
        
        # Convert target to Vector3
        target_vec = np.array(target_point, dtype=np.float32) #target position
        
        # Find path using A*
        path = habitat_sim.ShortestPath() #agent already has pathfinder with A*
        path.requested_start = current_pos
        path.requested_end = target_vec
        
        if not self.pathfinder.find_path(path): # no path found
            return False
        
        # Follow the path
        path_points = list(path.points)
        steps_taken = 0
        
        for point in path_points[1:]:  # Skip current position
            if steps_taken >= max_steps: # limit steps
                break
                
            # Move agent
            agent_state = self.agent.get_state()
            agent_state.position = point
            self.agent.set_state(agent_state)
            
            # Record position
            point_list = self.get_position_as_list(point)
            self.position_history.append(point_list)
            self.visited_positions.add(self.get_position_key(point_list))
            
            steps_taken += 1
            self.steps_since_last_yolo += 1
            
            # YOLO processing check
            if self.steps_since_last_yolo >= self.yolo_stop_interval: # time to stop for YOLO
                self.process_yolo_detection()  # Don't check return value - always continue
                self.steps_since_last_yolo = 0
            
            # Check if close to target
            current_pos_list = self.get_position_as_list(point)
            if self.calculate_distance(current_pos_list, target_point) < 1.5: # close enough to target
                return True
        
        # Check final distance
        if self.position_history:
            current_pos_list = self.position_history[-1]
            return self.calculate_distance(current_pos_list, target_point) < 2.0
        return False
    
    def process_yolo_detection(self):
        """
        YOLO processing point - always returns False (no object found)
        """
        current_position = self.agent.get_state().position
        current_step = len(self.position_history)
        
        # Convert position to list for consistent handling
        current_pos_list = self.get_position_as_list(current_position)
        
        print(f"YOLO PROCESSING at step {current_step}")
        print(f"  Position: [{current_pos_list[0]:.2f}, {current_pos_list[1]:.2f}, {current_pos_list[2]:.2f}]")
        
        # Record stop point
        self.stop_points.append(current_pos_list)
        
        # LATE I INTEGRATE YOLO HERE
        # For now, always return False (no object found)
        print(f"  No {self.target_object} detected")
        return False  # Always return False
    
    def explore_room(self, target_object="chair"):
        """
        Main exploration algorithm - continues until 30% coverage or max steps
        """
        print("Starting Frontier-Based Exploration with A*")
        print(f"Target: {target_object}, YOLO every {self.yolo_stop_interval} steps")
        print(f"Target coverage: {self.target_coverage}%")
        
        self.target_object = target_object # the object to find
        self.object_found = False  # Always false - no actual object detection
        
        # Initialize at random position (Later to set specific start near doors)
        start_point = self.pathfinder.get_random_navigable_point() #TO CHANGE TO START POINT NEAR DOOR
        agent_state = self.agent.get_state() 
        agent_state.position = start_point
        agent_state.rotation = np.quaternion(1, 0, 0, 0) # facing forward
        self.agent.set_state(agent_state)
        
        # Convert start point to list
        start_list = self.get_position_as_list(start_point)
        self.position_history = [start_list]
        self.visited_positions.add(self.get_position_key(start_point))
        self.stop_points = [start_list]
        self.frontier_points = []
        self.steps_since_last_yolo = 0
        
        step_count = 0
        failures = 0
        max_failures = 8  # Prevents infinite loops when the agent gets stuck
        
        current_coverage = self.calculate_coverage()
        
        #Infinite loops in dead ends or impossible navigation
        while (len(self.position_history) < self.max_total_steps and 
               current_coverage < self.target_coverage and 
               failures < max_failures):
            
            current_position = self.agent.get_state().position
            step_count += 1
            
            print(f"Step {step_count}: {len(self.visited_positions)} unique positions, Coverage: {current_coverage:.1f}%")
            
            # Find new frontiers periodically. Keeps fresh exploration targets available
            if not self.frontier_points or step_count % 15 == 0:
                self.frontier_points = self.discover_frontiers(current_position)
                print(f"  Found {len(self.frontier_points)} frontiers")
            
            # Navigate to frontiers. Takes the highest-scored frontier and navigates to it
            if self.frontier_points:
                next_target = self.frontier_points.pop(0)
                
                print(f"  Moving to frontier: [{next_target[0]:.2f}, {next_target[1]:.2f}, {next_target[2]:.2f}]")
                
                success = self.navigate_to_point(next_target)
                
                if success:
                    print("  Reached frontier")
                    failures = 0
                    # Extra YOLO at important locations
                    self.process_yolo_detection()
                else:
                    print("  Failed to reach frontier")
                    failures += 1
            else:
                # Local exploration fallback. Ensures the agent doesn't stop when frontier discovery fails
                print("  No frontiers, doing extended local exploration")
                self.extended_local_exploration(8)  # More local exploration
                failures += 1
            
            # Update coverage
            current_coverage = self.calculate_coverage()
            
            # Progress reporting
            if step_count % 10 == 0:
                print(f"  Progress: {current_coverage:.1f}% coverage, {len(self.position_history)} total steps")
        
        # Final stats
        final_coverage = self.calculate_coverage()
        
        print(f"\nEXPLORATION COMPLETE:")
        print(f"  Total steps: {len(self.position_history)}")
        print(f"  Unique positions: {len(self.visited_positions)}")
        print(f"  YOLO stops: {len(self.stop_points)}")
        print(f"  Final coverage: {final_coverage:.1f}%")
        print(f"  Object found: {self.object_found} (always False - no YOLO yet)")
        
        return self.position_history, self.stop_points, self.frontier_points
    
    #Smart local movement when no good frontiers are available
    def extended_local_exploration(self, num_steps):
        """Extended local movement for better coverage"""
        actions = ["move_forward", "turn_left", "turn_right"]
        
        for i in range(num_steps):
            # Slightly smarter action selection
            if i % 3 == 0:  # Prefer forward movement 66%. Forward movement covers more ground than turning in place
                action = "move_forward"
            else:
                action = random.choice(["turn_left", "turn_right"]) #33%
                
            self.sim.step(action)
            
            new_state = self.agent.get_state()
            new_pos = new_state.position
            new_pos_list = self.get_position_as_list(new_pos)
            
            self.position_history.append(new_pos_list)
            self.visited_positions.add(self.get_position_key(new_pos))
            
            self.steps_since_last_yolo += 1
            
            if self.steps_since_last_yolo >= self.yolo_stop_interval:
                self.process_yolo_detection()
                self.steps_since_last_yolo = 0

def setup_simulator_simple():
    """Simple simulator setup without sensors"""
    SCENE_PATH = "/home/minso/projectHabitat/data/scene_datasets/mp3d/17DRP5sb8fy/17DRP5sb8fy.glb"
    
    # Basic configuration
    sim_cfg = habitat_sim.SimulatorConfiguration() # simulator config
    sim_cfg.scene_id = SCENE_PATH
    sim_cfg.enable_physics = False
    sim_cfg.gpu_device_id = -1 # CPU mode
    
    # Simple agent configuration
    agent_cfg = habitat_sim.agent.AgentConfiguration() # agent config
    
    cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg]) # full config
    
    try:
        sim = habitat_sim.Simulator(cfg)
        print("Simulator created successfully")
        return sim, False  # No sensor
        
    except Exception as e:
        print(f"Failed to create simulator: {e}")
        return None, False

def calculate_path_length(points):
    total_distance = 0
    for i in range(len(points) - 1):
        dist = np.linalg.norm(np.array(points[i]) - np.array(points[i+1]))
        total_distance += dist
    return total_distance

def main():
    print("Habitat-Sim: Frontier Exploration with A*")
    
    os.makedirs('results', exist_ok=True)
    
    # Create simple simulator
    sim, has_sensor = setup_simulator_simple()
    
    if sim is None:
        print("Cannot create simulator. Exiting.")
        return
    
    # Create exploration agent
    explorer = SimpleExplorationAgent(sim)
    
    # Run exploration
    path_points, stop_points, frontier_points = explorer.explore_room(
        target_object="chair"
    )
    
    if path_points:
        # Create visualizations
        create_3d_path_visualization(
            path_points, stop_points, frontier_points, 
            "exploration"
        )
        create_2d_map(
            path_points, stop_points, frontier_points,
            "exploration"
        )
        
        # Calculate statistics
        path_length = calculate_path_length(path_points)
        unique_positions = len(explorer.visited_positions)
        final_coverage = explorer.calculate_coverage()
        
        # Save data
        exploration_data = {
            'algorithm': 'Frontier-Based + A*',
            'target_object': explorer.target_object,
            'object_found': explorer.object_found,  # Always False
            'total_steps': len(path_points),
            'path_length': path_length,
            'unique_positions': unique_positions,
            'final_coverage': final_coverage,
            'yolo_stops': len(stop_points),
            'frontiers': len(frontier_points),
            'yolo_interval': explorer.yolo_stop_interval,
            'target_coverage': explorer.target_coverage,
            'path_points': path_points,
            'stop_points': stop_points,
            'frontier_points': frontier_points
        }
        
        with open('results/exploration_data.json', 'w') as f:
            json.dump(exploration_data, f, indent=2)
        
        print(f"\nResults saved:")
        print(f"  - Visualizations in 'results/' folder")
        print(f"  - Data in 'results/exploration_data.json'")
        print(f"  - Final coverage: {final_coverage:.1f}%")
        print(f"  - Object found: {explorer.object_found}")
        
    else:
        print("Exploration failed")
    
    sim.close()

if __name__ == "__main__":
    main()