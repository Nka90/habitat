"""
Save 2d/3d maps
"""
import os
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
from typing import List, Dict, Optional
from config.settings import RoomBounds

class VisualizationSaver:
    def __init__(self, file_manager, room_bounds: RoomBounds, target_object: str = None):
        self.file_manager = file_manager
        self.room_bounds = room_bounds
        self.target_object = target_object
        
    def save_all_visualizations(self,
                           path_points: List[List[float]],
                           stop_points: List[List[float]],
                           frontier_points: List[List[float]],
                           object_path: Optional[Dict] = None): 
        #Save all visualizations
        # Create 3D visualization
        self._save_3d_visualization(path_points, stop_points, frontier_points)
        
        # Create 2D map with path
        self._save_2d_path_map(path_points, stop_points, frontier_points)
        
        #create object map with detected object and path to it
        self._save_object_map(object_path, self.target_object)
        
    def _save_3d_visualization(self,
                            path_points: List[List[float]],
                            stop_points: List[List[float]],
                            frontier_points: List[List[float]]):
        #save 3d map with 60 view angle (checking of height exploration)
        fig = plt.figure(figsize=(14, 10))
        ax = fig.add_subplot(111, projection='3d')

        #Extract coordinates of all points during exploration
        x_coords = [p[0] for p in path_points]
        y_coords = [p[1] for p in path_points]
        z_coords = [p[2] for p in path_points]

        # Plot path
        ax.scatter(x_coords, z_coords, y_coords, c="yellow", s=30, alpha=0.6)
        ax.plot(x_coords, z_coords, y_coords, 'b-', linewidth=1, alpha=0.3)
        ax.scatter(x_coords[0], z_coords[0], y_coords[0],
                c='green', s=200, marker='D', label='Start', edgecolors='black')
        ax.scatter(x_coords[-1], z_coords[-1], y_coords[-1],
                c='orange', s=200, marker='D', label='End', edgecolors='black')

        #Mark stop points for yolo detections
        if stop_points:
            stop_x = [p[0] for p in stop_points]
            stop_z = [p[2] for p in stop_points]
            stop_y = [p[1] for p in stop_points]
            ax.scatter(stop_x, stop_z, stop_y, c='yellow', s=120, marker='s',
                    label='Stops to YOLO Detections', edgecolors='black', alpha=0.8)

        # Mark frontiers
        if frontier_points:
            sample_rate = max(1, len(frontier_points) // 30)
            sampled = frontier_points[::sample_rate]
            frontier_x = [p[0] for p in sampled]
            frontier_z = [p[2] for p in sampled]
            frontier_y = [p[1] for p in sampled]
            ax.scatter(frontier_x, frontier_z, frontier_y, c='purple', s=80, marker='*',
                    label='Frontiers', edgecolors='black', alpha=0.7)

        # Draw boundaries 
        self._draw_room_boundaries_3d(ax)
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Z Position (m)')
        ax.set_zlabel('Y Position (m)')
        ax.set_title('Exploration Path 3D Visualization')
        ax.legend()

        # Set single angle (60°)
        ax.view_init(elev=20, azim=60)
        plt.tight_layout()

        # Save file
        filename = "3d_angle_60.png"
        filepath = os.path.join(self.file_manager.config.results_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
    
    def _draw_room_boundaries_3d(self, ax):
        #Draw room boundaries on 3D plot
        room_x, room_z = zip(*self.room_bounds.get_boundary_points())
        room_y = [0] * len(room_x)
        ax.plot(room_x, room_z, room_y, 'k--', alpha=0.3, label='Room Boundary')
    
    def _save_2d_path_map(self,
                         path_points: List[List[float]],
                         stop_points: List[List[float]],
                         frontier_points: List[List[float]]):
        #Save 2D map visualization with path only
        fig, ax = plt.subplots(figsize=(12, 10))
        
        x_coords = [p[0] for p in path_points]
        z_coords = [p[2] for p in path_points]
        
        # Plot path
        ax.scatter(x_coords, z_coords, c="blue", s=30, alpha=0.6)
        ax.plot(x_coords, z_coords, 'b-', linewidth=1, alpha=0.3)
        
        # Mark start and end
        ax.scatter(x_coords[0], z_coords[0], c='yellow', s=150, marker='D', 
                   label='Start', edgecolors='black', zorder=5)
        ax.scatter(x_coords[-1], z_coords[-1], c='red', s=150, marker='D', 
                   label='End', edgecolors='black', zorder=5)
        
        # Mark stop points
        if stop_points:
            stop_x = [p[0] for p in stop_points]
            stop_z = [p[2] for p in stop_points]
            
            ax.scatter(stop_x, stop_z, c='green', s=200, marker='o', 
                      label='Stops to YOLO Detections', edgecolors='black', alpha=0.5, zorder=3)
            ax.scatter(stop_x, stop_z, c='green', s=100, marker='s', 
                      edgecolors='black', alpha=0.9, zorder=4)
            
            # Add numbers to stop points
            for i, (x, z) in enumerate(zip(stop_x, stop_z)):
                ax.text(x, z, str(i), fontsize=9, fontweight='bold', 
                       ha='center', va='center', color='black')
        
        # Mark frontiers
        if frontier_points:
            sample_rate = max(1, len(frontier_points) // 20)
            sampled = frontier_points[::sample_rate]
            frontier_x = [p[0] for p in sampled]
            frontier_z = [p[2] for p in sampled]
            ax.scatter(frontier_x, frontier_z, c='purple', s=60, marker='*', 
                      label='Frontiers', edgecolors='black', alpha=0.8, zorder=3)
        
        # Draw boundaries
        self._draw_room_boundaries_2d(ax) 
        
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Z Position (m)')
        
        # Title for path map
        title = f'Exploration Path Map\n'
        title += f'{len(path_points)} steps, {len(stop_points)} stops'
        ax.set_title(title)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        #create legend
        legend_elements = [
        Line2D([0], [0], marker='o', color='w',
            markerfacecolor='blue', markersize=8,
            label='Step',
            markeredgecolor='black'),
        Line2D([0], [0], marker='D', color='w',
            markerfacecolor='yellow', markersize=10,
            label='Start point', markeredgecolor='black'),
        Line2D([0], [0], marker='D', color='w',
            markerfacecolor='red', markersize=10,
            label='End point', markeredgecolor='black'),
        Line2D([0], [0], marker='s', color='w',
            markerfacecolor='green', markersize=10,
            label='Detection points', markeredgecolor='black'),
        Line2D([0], [0], marker='*', color='w',
            markerfacecolor='purple', markersize=12,
            label='Frontiers', markeredgecolor='black'),
        ]
        
        ax.legend(
        handles=legend_elements,
        loc='upper left',
        bbox_to_anchor=(1.02, 1),
        fontsize=10,
        title="Legend",
        title_fontsize=11
        )

        plt.tight_layout()
        
        #save file
        filename = "2d_path_map.png"
        filepath = os.path.join(self.file_manager.config.results_dir, filename)
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print("2D path map png saved")
        
    def _save_object_map(self, object_path: Optional[Dict] = None, target_object: str = None):
        #save map with mapping target object, agent position and shortest path
        fig, ax = plt.subplots(figsize=(10, 8))

        # Draw boundaries
        self._draw_room_boundaries_2d(ax) 
        
        if object_path is None or not all(key in object_path for key in ['start', 'target', 'waypoints', 'distance']):
            print(f"No {target_object if target_object else 'object'} detected - skipping object map")
            return None
        
        try:
            agent_pos = object_path['start']
            obj_pos = object_path['target']
            waypoints = object_path['waypoints']
            distance = object_path['distance']
                                
            # Validate position arrays have at least 3 elements
            if len(agent_pos) < 3 or len(obj_pos) < 3:
                print(f"Position arrays have insufficient length")
            else:
                obj_x, obj_z = obj_pos[0], obj_pos[2]
                agent_x, agent_z = agent_pos[0], agent_pos[2]
                    
                #plot object
                ax.scatter(obj_x, obj_z, c='red', s=300, 
                        marker='*', edgecolors='black', linewidth=2, 
                        zorder=10, alpha=0.9)
                    
                #plot agent
                ax.scatter(agent_x, agent_z, c='blue', s=200, 
                        marker='^', edgecolors='black', linewidth=2, 
                        zorder=9, alpha=0.8)
                    
                #plot path
                if waypoints and len(waypoints) > 1:
                    path_x = [wp[0] for wp in waypoints]
                    path_z = [wp[2] for wp in waypoints]
                        
                    # Path line
                    ax.plot(path_x, path_z, color='green', linewidth=3, 
                        alpha=0.7, zorder=7, label=f'A* Path ({distance:.1f}m)')
                        
                    # Waypoints
                    ax.scatter(path_x, path_z, c='green', s=30, 
                            alpha=0.5, zorder=8, edgecolors='black', linewidth=0.5)
 
        except (KeyError, IndexError, TypeError) as e:
            print(f"Error processing path data: {e}")

        # Add legend
        legend_elements = [
            Line2D([0], [0], color='k', linestyle='--', alpha=0.3, label='Room Boundary'),
            Line2D([0], [0], marker='*', color='w', markerfacecolor='red', 
                markersize=10, label=f"Detected {target_object.capitalize()} position"),
            Line2D([0], [0], marker='^', color='w', markerfacecolor='blue', 
                markersize=10, label='Agent at detection'),
            Line2D([0], [0], color='green', linewidth=3, label=f'A* Path ({distance:.1f}m)')
        ]
        ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1), fontsize=10)
        
        # Labels and title
        title = f'\nObject Detection Map'
        title += f'\nTarget: {target_object.capitalize()} at ({obj_x:.2f}, {obj_z:.2f})'
        title += f'\nAgent at ({agent_x:.2f}, {agent_z:.2f})'
        title += f'\nPath length: {distance:.2f}m'
        ax.set_xlabel('X Position (m)')
        ax.set_ylabel('Z Position (m)')
        ax.set_title(title)
        
        # Grid and aspect
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        plt.tight_layout()
        
        #save map
        filename = "object_location.png"
        filepath = os.path.join(self.file_manager.config.results_dir, filename)
        fig.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f"Object map saved")
        
        return filepath 
        
    def _draw_room_boundaries_2d(self, ax):
        # Room boundaries
        room_x, room_z = zip(*self.room_bounds.get_boundary_points())
        ax.plot(room_x, room_z, 'k-', alpha=0.5, label='Room Boundary', linewidth=2)

