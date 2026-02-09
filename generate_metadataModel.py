import os
os.environ.update({
    'LIBGL_ALWAYS_SOFTWARE': '1',
    'HABITAT_SIM_HEADLESS': '1',
    'PYOPENGL_PLATFORM': 'osmesa'
})

import habitat_sim
import numpy as np

def create_navmesh_for_model():

    SCENE_PATH = "/home/minso/projectHabitat/data/scene_datasets/myModelRoom/room_1.glb"
    NAVMESH_PATH = "/home/minso/projectHabitat/data/scene_datasets/myModelRoom/room_1.navmesh"

    print(f"Creating navmesh for: {SCENE_PATH}")

    # --- Simulator config ---
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = SCENE_PATH
    sim_cfg.enable_physics = False
    sim_cfg.gpu_device_id = -1   # CPU-only

    # --- Agent config ---
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])

    try:
        sim = habitat_sim.Simulator(cfg)
        print("Simulator created successfully")

        # ===== OLD API — WORKS ON CPU BUILDS =====
        settings = habitat_sim.NavMeshSettings()
        settings.set_defaults()

        # modify if needed
        settings.agent_height = 1.5      # Total height of agent
        settings.agent_radius = 0.1      # Radius of agent
        settings.agent_max_climb = 0.1   # Maximum step height agent can climb
        settings.agent_max_slope = 45.0  # Maximum slope angle in degrees
        
        # These control the navmesh precision:
        settings.cell_size = 0.05
        settings.cell_height = 0.01

        print("Recomputing NavMesh...")
        success = sim.recompute_navmesh(sim.pathfinder, settings)

        if not success:
            raise RuntimeError("Navmesh recompute failed")

        print("Saving navmesh...")
        sim.pathfinder.save_nav_mesh(NAVMESH_PATH)

        if os.path.exists(NAVMESH_PATH):
            print(f"Navmesh created successfully: {NAVMESH_PATH}")
        else:
            print("ERROR: navmesh file was not created")

        sim.close()
        return True

    except Exception as e:
        print(f"Navmesh creation FAILED: {e}")
        return False

def main():
    print("=== NAVMESH CREATION & TEST ===")

    create_navmesh_for_model()


if __name__ == "__main__":
    main()
