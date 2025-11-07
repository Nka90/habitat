import os
os.environ.update({
    'LIBGL_ALWAYS_SOFTWARE': '1',
    'HABITAT_SIM_HEADLESS': '1',
    'PYOPENGL_PLATFORM': 'osmesa'  # Force OSMesa instead of GLX
})

import habitat_sim
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json

def create_3d_path_visualization(path_points, filename):
    """Create a 3D visualization of the path"""
    from mpl_toolkits.mplot3d import Axes3D
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    x_coords = [point[0] for point in path_points]
    y_coords = [point[1] for point in path_points]  # Height
    z_coords = [point[2] for point in path_points]
    
    # Plot the path in 3D
    ax.plot(x_coords, z_coords, y_coords, 'b-', linewidth=3, alpha=0.7, label='Path')
    ax.scatter(x_coords, z_coords, y_coords, c='red', s=100, alpha=0.8, zorder=5)
    
    # Mark start and end
    ax.scatter(x_coords[0], z_coords[0], y_coords[0], 
               c='green', s=300, marker='D', label='Start', edgecolors='black')
    ax.scatter(x_coords[-1], z_coords[-1], y_coords[-1], 
               c='orange', s=300, marker='D', label='End', edgecolors='black')
    
    # Add step numbers
    for i, (x, z, y) in enumerate(zip(x_coords, z_coords, y_coords)):
        ax.text(x, z, y, f'{i}', fontsize=8, fontweight='bold')
    
    ax.set_xlabel('X Position')
    ax.set_ylabel('Z Position')
    ax.set_zlabel('Y Position (Height)')
    ax.set_title(f'3D Navigation Path\n{len(path_points)} steps')
    ax.legend()
    
    # Different viewing angles
    for angle in [30, 60, 90]:
        ax.view_init(elev=20, azim=angle)
        plt.tight_layout()
        plt.savefig(f'results/{filename}_angle_{angle}.png', dpi=150, bbox_inches='tight')
        print(f"3D Visualization saved: results/{filename}_angle_{angle}.png")
    
    plt.close()

def calculate_path_length(points):
    total_distance = 0
    for i in range(len(points) - 1):
        dist = np.linalg.norm(np.array(points[i]) - np.array(points[i+1]))
        total_distance += dist
    return total_distance

def try_minimal_sensor():
    """Try the absolute minimum sensor setup"""
    SCENE_PATH = "/home/minso/projectHabitat/data/scene_datasets/mp3d/17DRP5sb8fy/17DRP5sb8fy.glb"
    
    # Try with NO sensor first, then add one later
    sim_cfg = habitat_sim.SimulatorConfiguration()
    sim_cfg.scene_id = SCENE_PATH
    sim_cfg.enable_physics = False
    sim_cfg.gpu_device_id = -1
    
    agent_cfg = habitat_sim.agent.AgentConfiguration()
    cfg = habitat_sim.Configuration(sim_cfg, [agent_cfg])
    
    try:
        # First create simulator without sensors
        sim = habitat_sim.Simulator(cfg)
        print("✓ Simulator created without sensors")
        
        # Now try to add a sensor dynamically (this sometimes works better)
        try:
            # Create a minimal RGB sensor
            sensor_spec = habitat_sim.CameraSensorSpec()
            sensor_spec.uuid = "color_sensor"
            sensor_spec.sensor_type = habitat_sim.SensorType.COLOR
            sensor_spec.resolution = [64, 64]  # Absolute minimum
            sensor_spec.position = [0.0, 1.5, 0.0]
            sensor_spec.hfov = 90
            
            # Add sensor to existing agent
            sim.agents[0].add_sensor(sensor_spec)
            print("✓ Sensor added dynamically")
            
            # Test the sensor
            observations = sim.get_sensor_observations()
            if "color_sensor" in observations:
                print("✓ Sensor is working!")
                return sim, True
            else:
                print("✗ Sensor added but no observations")
                return sim, False
                
        except Exception as e:
            print(f"✗ Could not add sensor dynamically: {e}")
            return sim, False
            
    except Exception as e:
        print(f"✗ Could not create simulator: {e}")
        return None, False

def main():
    print("=== Habitat-Sim with 3D Path Visualization ===")
    
    os.makedirs('results', exist_ok=True)
    
    # Try to get a simulator with sensor
    sim, has_sensor = try_minimal_sensor()
    
    if sim is None:
        print("Failed to create simulator, exiting...")
        return
    
    pathfinder = sim.pathfinder
    
    # Generate paths with 3D visualizations
    for path_num in range(3):
        print(f"\n--- Generating Path {path_num + 1} ---")
        
        start_point = pathfinder.get_random_navigable_point()
        end_point = pathfinder.get_random_navigable_point()
        
        path = habitat_sim.ShortestPath()
        path.requested_start = start_point
        path.requested_end = end_point
        
        if pathfinder.find_path(path):
            path_points = [list(point) for point in path.points]
            
            # Create 3D visualization
            create_3d_path_visualization(path_points, f"3d_path_{path_num+1}")
            
            # Save detailed path data
            path_data = {
                'path_number': path_num + 1,
                'start_point': list(start_point),
                'end_point': list(end_point),
                'path_points': path_points,
                'distance': path.geodesic_distance,
                'num_points': len(path.points),
                'has_sensor': has_sensor
            }
            
            with open(f'results/path_3d_data_{path_num+1}.json', 'w') as f:
                json.dump(path_data, f, indent=2)
            
            print(f"Path {path_num+1}: {len(path.points)} steps, {path.geodesic_distance:.2f}m")
            
        else:
            print(f"Path {path_num+1}: No path found")
    
    sim.close()
    print("\n3D path visualizations completed!")
    print("Check 'results' folder for multiple 3D views of each path")

if __name__ == "__main__":
    main()