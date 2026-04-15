"""
Detection results processing and calculation coordinates of objects in environment using depth-based method
"""

import cv2
from config.settings import AgentConfig
import numpy as np
from typing import List, Dict, Any, Optional
import quaternion

class DetectionProcessor:
    def __init__(self, sim, storage_manager, detector=None, camera_params=None):
        self.sim = sim
        self.storage = storage_manager
        self.detector = detector
        self.agent_config = AgentConfig()
        self.detection_results = []
        self.highest_confidence_frame = None #track the frame with the highest confidence detection for later saving
        self.highest_confidence_detection = None #track the highest confidence detection data for summary and path calculation
        self.highest_confidence_value = 0.0 #track the highest confidence value for comparing
        self.highest_confidence_metadata = None #track the metadata of the highest confidence detection for path calculation
        self.camera_params = {
            'fov': camera_params.fov,
            'width': camera_params.width,
            'height': camera_params.height,
            'sensor_height': camera_params.sensor_height
        }
        
        self._calculate_camera_matrix()
        
    def _calculate_camera_matrix(self):
        #Calculate camera intrinsic matrix for depth-based coordinate calculation
        W = self.camera_params['width']
        H = self.camera_params['height'] #640*640
        fov_rad = np.radians(self.camera_params['fov'])

        focal_length = (W / 2) / np.tan(fov_rad / 2)

        self.camera_params["fx"] = focal_length
        self.camera_params["fy"] = focal_length
        self.camera_params["cx"] = W / 2
        self.camera_params["cy"] = H / 2

        self.camera_matrix = np.array([
            [focal_length, 0, self.camera_params['cx']],
            [0, focal_length, self.camera_params['cy']],
            [0, 0, 1]
        ])
    
    def set_detector(self, detector):
        self.detector = detector
    
    def calculate_3d_coordinates_depth(self, bbox, depth, agent_state):
        #Calculate 3D coordinates using depth information from sensor
        #ground contact point
        x1, y1, x2, y2 = bbox #coordinates of bbox
        u = int((x1 + x2) / 2) #calculate center of bbox
        v = int(y2) #use bottom center of bbox

        u = max(0, min(u, depth.shape[1] - 1))
        v = max(0, min(v, depth.shape[0] - 1))

        d = float(depth[v, u]) #get depth value of points 
        if d <= 0 or np.isnan(d):
            return None

        # Use camera intrinsics to calculate 3D coordinates in camera frame: focal length(how zoomed in the camera is) and principal point(optical center of the camera/frame center)
        fx = self.camera_params["fx"]
        fy = self.camera_params["fy"]
        cx = self.camera_params["cx"]
        cy = self.camera_params["cy"]

        #convert from pixel coordinates to camera 3d coordinates
        Xc = (u - cx) * d / fx #horizontal
        Yc = (v - cy) * d / fy #vertical
        Zc = d #how far the point is from the camera
        point_cam = np.array([Xc, -Yc, -Zc], dtype=np.float32) #invert Y and Z to match world coordinate system (habitat system)
        
        sensor_state = agent_state.sensor_states["color_sensor"]
        q = sensor_state.rotation
        R = quaternion.as_rotation_matrix(q).astype(np.float32) #direction the camera is facing in world coordinates
        t = np.array(sensor_state.position, dtype=np.float32) #position of the camera in world coordinates
        
        point_world = R @ point_cam + t #transforming to word coordinates

        return { #3d coordinates of the detected object
            "x": float(point_world[0]),
            "y": float(point_world[1]),
            "z": float(point_world[2]),
            "distance": float(np.linalg.norm(point_cam))
        }
    
    def process_detection(self, frame_rgb: np.ndarray, metadata: Dict[str, Any], 
                         detector_results, target_class: str = None) -> Optional[Dict]:
        #Process detection for one single frame on detection step in cycle
        if detector_results is None or len(detector_results) == 0:
            return None
        
        result = detector_results[0]
        boxes = result.boxes
        
        if boxes is None or len(boxes) == 0:
            return None
        
        all_detections = self._extract_detection_data_with_coords(result, metadata) #metadata (position of agent, direction..), num_detections, detections(info from yolo model)

        # Filter by target class and confidence - work with target_object class only
        if target_class:
            relevant_detections = [
                det for det in all_detections['detections'] 
                if (det['class_name'].lower() == target_class.lower() and 
                    det['confidence'] > 0.4) #filter high confidence for not saving too many uncertain detections
            ]
        
        if not relevant_detections:
            return None
                
        #if are more detections on the one frame, keep the highest
        highest_conf_detection = max(relevant_detections, key=lambda d: d['confidence'])
        relevant_detections = [highest_conf_detection]
        
        # Track overall highest confidence across all frames
        if highest_conf_detection['confidence'] > self.highest_confidence_value:
            self.highest_confidence_value = highest_conf_detection['confidence']
            self.highest_confidence_detection = highest_conf_detection
            self.highest_confidence_metadata = metadata.copy()
        
        # Create detection data (only one detection now)
        detection_data = {
            'metadata': all_detections['metadata'],
            'detections': relevant_detections
        }
        
        if self.detector:
            # Draw only the highest confidence detection
            frame_with_boxes = self.detector.draw_detections(frame_rgb, relevant_detections)
            frame_with_boxes = self._annotate_coordinates(frame_with_boxes, relevant_detections)
            if highest_conf_detection['confidence'] == self.highest_confidence_value:
                self.highest_confidence_frame = frame_with_boxes.copy()
            self._save_annotated_frame(frame_with_boxes, metadata, detection_data)
        self.detection_results.append(detection_data)
        return detection_data
    
    def _extract_detection_data_with_coords(self, result, metadata: Dict) -> Dict:
        #Extract detection data with depth-based 3D coordinates
        boxes = result.boxes

        if hasattr(boxes.xyxy, 'cpu'):
            xyxy = boxes.xyxy.cpu().numpy() #coordinates of bbox
            conf = boxes.conf.cpu().numpy() #confidence scores
            cls = boxes.cls.cpu().numpy() #class id
        else:
            xyxy = boxes.xyxy.numpy()
            conf = boxes.conf.numpy()
            cls = boxes.cls.numpy()

        names = result.names
        depth = metadata.get("depth")
        agent_state = self.sim.agents[0].state

        detections = []
        for box, confidence, class_id in zip(xyxy, conf, cls):
            x1, y1, x2, y2 = map(int, box)
            class_name = names[int(class_id)]
            
            coords = None
            if depth is not None:
                coords = self.calculate_3d_coordinates_depth(
                    [x1, y1, x2, y2], depth, agent_state
                )
            
            if coords is not None:
                detections.append({
                    'bbox': [x1, y1, x2, y2],
                    'confidence': float(confidence),
                    'class_id': int(class_id),
                    'class_name': class_name,
                    'coordinates_3d': coords
                })

        return {
            'metadata': metadata,
            'detections': detections
        }
    
    def _annotate_coordinates(self, frame: np.ndarray, detections: List[Dict]) -> np.ndarray:
        #Add 3D coordinate text to frame
        frame_with_anno = frame.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            coords = detection['coordinates_3d']
            
            text_y = y2 + 20
            coord_text = f"X:{coords['x']:.1f} Z:{coords['z']:.1f}"
            dist_text = f"Dist:{coords['distance']:.1f}m"
            
            cv2.putText(frame_with_anno, coord_text, (x1, text_y), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            cv2.putText(frame_with_anno, dist_text, (x1, text_y + 15), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
        
        return frame_with_anno
    
    def _save_annotated_frame(self, frame_with_boxes: np.ndarray, metadata: Dict, detection_data: Dict):
        #Save annotated frame
        direction_deg = np.degrees(metadata.get('direction', 0))
        view_type = metadata.get('view_type', 'unknown') #position of agent
        
        class_names = '_'.join(set(d['class_name'] for d in detection_data['detections']))
        filename = f"detection_{class_names}_x{metadata['x']:.1f}_z{metadata['z']:.1f}_dir{direction_deg:.0f}_{view_type}.jpg"
        
        frame_bgr = cv2.cvtColor(frame_with_boxes, cv2.COLOR_RGB2BGR)
        self.storage.save_detection_frame(filename, frame_bgr)
    
    def save_highest_confidence_frame(self):
        #Save the highest confidence annotated frame to result dir separately
        if self.highest_confidence_frame is None or self.highest_confidence_detection is None:
            return None
        
        frame_bgr = cv2.cvtColor(self.highest_confidence_frame, cv2.COLOR_RGB2BGR)
        import os
        filepath = os.path.join(self.storage.config.results_dir, "highest_confidence_detection.png")
        cv2.imwrite(filepath, frame_bgr)
        
        return filepath
    
    def generate_summary(self) -> Dict[str, Any]:
        #Generate summary of all detections
        if not self.detection_results:
            return {'total_detections': 0}
        
        total_detections = sum(len(r['detections']) for r in self.detection_results)
        object_locations = []
        
        for result in self.detection_results:
            for detection in result['detections']:
                coords = detection['coordinates_3d']
                object_locations.append({
                    'class': detection['class_name'],
                    'coordinates': {
                        'x': coords['x'],
                        'z': coords['z'],
                        'distance': coords['distance']
                    },
                    'confidence': detection['confidence']
                })
        
        return {
            'total_frames': len(self.detection_results),
            'total_detections': total_detections,
            'object_locations': object_locations
        }
    
    def process_obj_location(self, target_object: str = None, explorer=None) -> Optional[List[List[float]]]:
        #Process location of target detected object and calculate path for visualization later
        # Check if we have any detection
        if not self.highest_confidence_detection: 
            print(f"No detections found for target: {target_object}")
            return None
        
        agent_pos_at_detection = [ #from tracked metadata of the highest confidence detection
            self.highest_confidence_metadata['x'], 
            self.agent_config.agent_height,
            self.highest_confidence_metadata['z']
        ]
        
        coords = self.highest_confidence_detection['coordinates_3d']
        raw_obj_pos = np.array([
            coords['x'],
            self.agent_config.agent_height,
            coords['z']
        ], dtype=np.float32)
        
        #check if the object position is navigable, if not snap to nearest navigable point on navMesh
        is_navigable = self.sim.pathfinder.is_navigable(raw_obj_pos) 
        if is_navigable:
            obj_pos = raw_obj_pos.tolist()
        else:
            obj_pos = np.array(self.sim.pathfinder.snap_point(raw_obj_pos)).tolist() 
        
        return explorer.calculate_final_path(agent_pos_at_detection, obj_pos)