"""
Main program for the exploration agent
"""
#import configurations and modules
import argparse
import os
import numpy as np
import math
import cv2

from config.settings import (RoomBounds, AgentConfig, PathConfig, PathsConfig, CameraConfig)
from environment.simulator_setup import SimulatorManager
from environment.path_utils import PathUtils
from agent.movement_controller import MovementController
from agent.exploration_strategy import FrontierExplorer
from perception.perception_cycle import PerceptionCycle
from detection.yolo_detector import YOLODetector
from detection.detection_processor import DetectionProcessor
from storage.file_manager import FileManager
from storage.visualization_saver import VisualizationSaver

class ExplorationAgent:
    def __init__(self, paths_config: PathsConfig, target_object: str = None): 
        #load configurations of environment, storage, perception, detection, and agent controllers, initialize components
        self.paths = paths_config
        self.room_bounds = RoomBounds()
        self.agent_config = AgentConfig()
        self.path_config = PathConfig()
        self.camera_config = CameraConfig()
        self.target_object = target_object.lower()
        
        #initialize components
        self._init_environment()
        self._init_storage()
        self._init_perception()
        self._init_detection()
        self._init_agent_controllers()

        #store visited positions and frontier points by agent
        self.position_history = []
        self.frontier_points = []
    
    def _init_environment(self):
        #Initialize environment components: simulator lifecycle and settings of simulator, pathfinder, generator of 2D/3D maps of environment
        self.sim_manager = SimulatorManager(self.paths.scene_path, self.agent_config, self.camera_config) 
        self.sim = self.sim_manager.setup() 
        self.pathfinder = self.sim.pathfinder 
    
    def _init_storage(self):
        #Initialize storage components: create directories and set up file manager, visualization saver, and results exporter
        self.paths.initialDeleteDirectories() # Clear previous results before starting a new run
        self.paths.create_directories()
        self.file_manager = FileManager(self.paths)
        self.visualization_saver = VisualizationSaver(self.file_manager, self.room_bounds, target_object=self.target_object)

    def _init_perception(self):
        #Initialize perception components
        self.perception_cycle = PerceptionCycle(
            sim = self.sim, 
            yolo_stop_interval = self.agent_config.yolo_stop_interval
        )
        
    def _init_detection(self):
        #Initialize detection components
        if os.path.exists(self.paths.yolo_model_path):
            self.detector = YOLODetector(self.paths.yolo_model_path)
            self.detection_processor = DetectionProcessor(
                self.sim,
                self.file_manager,
                detector=self.detector,
                camera_params=self.camera_config 
            )
            self.perception_cycle.register_frame_callback(
                self._on_frame_captured #this will be called every frame and trigger yolo detection when it is time
            )
        else:
            print(f"YOLO model not found at {self.paths.yolo_model_path}")
            self.detector = None
            self.detection_processor = None
    
    def _init_agent_controllers(self):
        #Initialize agent movement controllers: movement controller, frontier explorer, and obstacle avoidance
        agent = self.sim.get_agent(0)     
        self.movement = MovementController(
            agent, self.pathfinder, self.agent_config
        )     
        self.explorer = FrontierExplorer(
            self.pathfinder, self.movement, self.path_config
        )
        
    def _on_frame_captured(self, frame_rgb, metadata):
        #trigger yolo if it is time to detect
        if self.detector and self.detection_processor:
            # Check if this is a detection frame
            if metadata.get('view_type') == "stop":
                results = self.detector.detect(frame_rgb)
                self.detection_processor.process_detection(
                    frame_rgb,
                    metadata,
                    results,
                    target_class=self.target_object
                )
            # else: skip YOLO, just save the frame

        # Always save the frame (for visualization of agents path)
        pos = [metadata['x'], metadata['y'], metadata['z']]
        view_type = metadata.get('view_type', 'unknown')
        
        filename = self.file_manager.get_frame_filename(
            self.perception_cycle.frame_count,
            pos,
            metadata['direction'],
            view_type
        )
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        self.file_manager.save_frame(filename, frame_bgr)
        
    def explore(self):
        #Main exploration loop: set start point, store agent positions, counter for steps, discover and move to frontiers, calculate paths, call perception cycle, call generating results of exploration 
        print("\nStarting Exploration")
 
        #set start point and orientation of agent    
        start_point = [0.0, self.agent_config.agent_height, 0.0] #agent has specific height
        self.movement.set_orientation(math.radians(0)) #need to convert to radians because habitat simulator works with them
        agent_state = self.movement.get_state()
        agent_state.position = np.array(start_point, dtype=np.float32)
        self.movement.set_state(agent_state)
        self.explorer.add_visited_position(start_point)
        self.position_history = [start_point]
        self.perception_cycle.stop_points = [start_point]
        self.perception_cycle.set_direction(self.movement.current_direction)

        #take first frame (to make sure agent position and orientation are correct)
        self.perception_cycle.capture_frame(
            movement_controller=self.movement,
            position=start_point,
            step_number=0,
            view_type="start"
        )

        step_count = 0
        failures = 0
        max_failures = 8 #limit of failures to reach frontier before stopping exploration (to prevent getting stuck)
        
        while (
            len(self.position_history) < self.agent_config.max_total_steps #use limit on total steps to prevent long exploration
            and failures < max_failures
        ):
            step_count += 1
            current_pos = self.movement.get_state().position #record current position
            current_list = PathUtils.get_position_as_list(current_pos) #convert position to list [x,y,z]
            self.perception_cycle.set_direction(self.movement.current_direction)
            
            frontiers = self.explorer.discover_frontiers(current_pos) #discover frontiers around current position
            self.frontier_points.extend(frontiers) #accumulates all discovered frontiers over time

            if frontiers:
                target = self.explorer.select_next_target(frontiers) #choose frontier
                if target:
                    print(f"Moving to frontier at ({target[0]:.1f}, {target[2]:.1f})")
                    direction_to_target = PathUtils.calculate_direction_to_point(current_list, target) #calculate direction to the target frontier
                    self.movement.set_orientation(direction_to_target) #turn agent to face the target frontier
                    success = self.movement.move_to_point( #move to the target frontier (check if frontier is navigable, calculate shortest path with obstacle avoidance)
                        target,
                        step_callback=self._on_step_complete, #store visited position
                        max_steps=100 #limit steps to reach frontier to prevent getting stuck
                    )
                    if success:
                        failures = 0
                    else:
                        failures += 1
                else:
                    failures += 1

            else:
                print("No more frontiers found. Exploration complete.") 
                break

            self.perception_cycle.step(self.movement, step_count) #perception - connect with model to detect objects

        self._generate_results() #generating results based on exploration
        return self.position_history, self.perception_cycle.stop_points
    
    def _on_step_complete(self, position, step_number):
        #store visited position after step
        self.explorer.add_visited_position(position)
        self.position_history.append(position)
        
    def _generate_results(self):
        #Generate all results and visualizations: summary of detections with coordinates, export to text file, maps, mechanism of filtering by the high confidence detection
        print("\nGenerating results")
        #generating summary of exploration
        detection_summary = (self.detection_processor.generate_summary() if self.detection_processor else {'total_detections': 0})
        
        if self.detection_processor:
            self.detection_processor.save_highest_confidence_frame() #save detection frame of the object with the highest confidence
        
        object_path = self.detection_processor.process_obj_location(self.target_object, self.explorer) if self.detection_processor else None #process location of detected object and calculate path to it
        
        #save all vizualizations of exploration: 2d/3d maps
        self.visualization_saver.save_all_visualizations(
            self.position_history, #all positions during exploration
            self.perception_cycle.stop_points, #positions with yolo detections
            self.frontier_points, #all discovered frontiers
            object_path=object_path #path to detected object from agent
        )
        
        print("Exploration summary: ")
        print(f"Total steps: {len(self.position_history)}")
        print(f"YOLO stop points: {len(self.perception_cycle.stop_points)}")
        print(f"Frontiers discovered: {len(self.frontier_points)}")
        print(f"Detection frames: {detection_summary.get('total_frames', 0)}")
        
def main():    
    print("\n")
    print("\nInitializing Exploration Agent")
    #take arguements from gui
    parser = argparse.ArgumentParser()
    parser.add_argument('--object', type=str, required=True) #object to search for
    args = parser.parse_args()
    
    print(f"Looking for object: {args.object}")
    
    #checking of scene
    paths = PathsConfig()
    if not os.path.exists(paths.scene_path):
        print(f"Scene not found at {paths.scene_path}")
    
    #checking of yolo model
    agent = ExplorationAgent(paths, target_object=args.object)
    try:
        agent.explore()
        print(f"\nExploration complete!")
    except Exception as e:
        print(f"\nError during exploration: {e}")
    finally:
        #agent.close()
        agent.sim_manager.close()
        
if __name__ == "__main__":
    main()