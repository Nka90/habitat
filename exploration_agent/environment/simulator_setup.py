"""
Habitat simulator setup and configuration
"""
import os
import habitat_sim
import magnum as mn
from typing import Optional
from config.settings import AgentConfig, CameraConfig


class SimulatorManager:
    def __init__(self, scene_path: str, agent_config: AgentConfig, camera_config: CameraConfig):
        self.scene_path = scene_path
        self.agent_config = agent_config
        self.camera_config = camera_config
        self.sim: Optional[habitat_sim.Simulator] = None
    
    def setup(self) -> habitat_sim.Simulator:
        # Setup environment for CPU rendering
        os.environ['LIBGL_ALWAYS_SOFTWARE'] = '1'
        os.environ['GALLIUM_DRIVER'] = 'llvmpipe'
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        
        #create agent
        sim_config = habitat_sim.SimulatorConfiguration()
        sim_config.enable_physics = True
        sim_config.gpu_device_id = -1
        sim_config.scene_id = self.scene_path
        
        #configuration of sensor
        #rgb
        color_spec = habitat_sim.CameraSensorSpec()
        color_spec.uuid = "color_sensor"
        color_spec.sensor_type = habitat_sim.SensorType.COLOR
        color_spec.resolution = [640, 640]
        color_spec.position = mn.Vector3(0, self.camera_config.sensor_height, 0)
        color_spec.orientation = mn.Vector3(0, 0, 0)

        #depth
        depth_spec = habitat_sim.CameraSensorSpec()
        depth_spec.uuid = "depth_sensor"
        depth_spec.sensor_type = habitat_sim.SensorType.DEPTH
        depth_spec.resolution = [640, 640]
        depth_spec.position = mn.Vector3(0, self.camera_config.sensor_height, 0)
        depth_spec.orientation = mn.Vector3(0, 0, 0)
        depth_spec.min_depth = 0.0
        depth_spec.max_depth = 10.0

        #create agent and simulator
        agent_config = habitat_sim.AgentConfiguration()
        agent_config.sensor_specifications = [color_spec, depth_spec]
        
        cfg = habitat_sim.Configuration(sim_config, [agent_config])
        
        self.sim = habitat_sim.Simulator(cfg)
        
        #building navmesh for model
        self.build_navmesh()
        
        return self.sim
    
    def build_navmesh(self):

        settings = habitat_sim.NavMeshSettings()
        settings.agent_radius = self.agent_config.agent_radius
        settings.agent_height = self.camera_config.sensor_height
        settings.agent_max_climb = 0.2
        settings.agent_max_slope = 30.0
        settings.include_static_objects = True 

        success = self.sim.recompute_navmesh(self.sim.pathfinder, settings)

        if success:
            print(f"NavMesh built successfully")
        else:
            print("NavMesh build failed!")
    
    def close(self):
        #close the simulator
        if self.sim:
            self.sim.close()
            print("Simulator closed")