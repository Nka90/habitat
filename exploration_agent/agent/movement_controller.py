"""
Agent movement and A* pathfinding controller
"""
import numpy as np
import habitat_sim
from typing import Optional, Callable
from config.settings import AgentConfig
from environment.path_utils import PathUtils


class MovementController:    
    def __init__(self, agent, pathfinder, agent_config: AgentConfig): 
        self.agent = agent
        self.pathfinder = pathfinder
        self.agent_config = agent_config
        self.current_direction = 0.0  # radians
        
    def set_orientation(self, direction_radians: float):
        #Set agent orientation to a specific direction (in radians) - habitat-sim work with radians     
        rotation = PathUtils.create_rotation_quaternion(direction_radians) #Habitat-Sim expects quaternion in [w, x, y, z] order, so we need to set rotation in right format

        # Set the agent state after eah rotation
        agent_state = self.agent.get_state()
        agent_state.rotation = rotation
        self.agent.set_state(agent_state)
        self.current_direction = direction_radians

    #states of agent
    def get_state(self):
        return self.agent.get_state()
    
    def set_state(self, state):
        self.agent.set_state(state)

    @property
    def sim(self):
        return self.agent.sim
    
    def move_to_point(
            self,
            target_point,
            step_callback: Optional[Callable] = None,
            max_steps: int = 100
        ) -> bool:
        #process of moving the agent to a target
        
        #get position of agent and target point (frontier)
        start = np.array(PathUtils.get_position_as_list(
            self.agent.get_state().position
        ), dtype=np.float32)

        target = np.array(PathUtils.get_position_as_list(target_point), dtype=np.float32)
        target[1] = self.agent_config.agent_height

        #if target point is not navigable, snap it to nearest navigable point
        if not self.pathfinder.is_navigable(target):
            target = self.pathfinder.snap_point(target)

        #calculate the shortest path using A* algorithm
        path = habitat_sim.ShortestPath()
        path.requested_start = start
        path.requested_end = target

        if not self.pathfinder.find_path(path) or len(path.points) < 2:
            return False

        steps_taken = 0
        current_pos = start

        for waypoint in path.points[1:]:
            if steps_taken >= max_steps:
                break

            waypoint = np.array(waypoint, dtype=np.float32)
            waypoint[1] = self.agent_config.agent_height

            #if the next step is can be taken
            next_pos = self.pathfinder.try_step(current_pos, waypoint)

            #if the current position is the same as the next position - stop
            if np.allclose(next_pos, current_pos):
                return False

            #rotation to a face the next position (frontier point)
            direction = PathUtils.calculate_direction_to_point(
                PathUtils.get_position_as_list(current_pos),
                PathUtils.get_position_as_list(next_pos)
            )
            self.set_orientation(direction)

            #new state of agent after step
            agent_state = self.agent.get_state()
            agent_state.position = next_pos
            self.agent.set_state(agent_state)

            current_pos = next_pos
            steps_taken += 1

            # Callback
            if step_callback:
                step_callback(PathUtils.get_position_as_list(current_pos), steps_taken)

            #if the target is reached
            if PathUtils.calculate_distance(
                PathUtils.get_position_as_list(current_pos),
                PathUtils.get_position_as_list(target)
            ) < 0.5:
                return True

        final_pos = PathUtils.get_position_as_list(self.agent.get_state().position)
        return PathUtils.calculate_distance(final_pos, PathUtils.get_position_as_list(target)) < 0.5