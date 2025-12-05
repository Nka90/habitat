#the cursor on the object & in center of the cs(0,0,0), face to y positive
import bpy
import os
import math

OBJECT_NAME = "chair"  # depends on the object
OUTPUT_DIR = f"D:/nka/bakalarka/materials/3d_model/render_results/{OBJECT_NAME}"

#parametres for rendering (light intensities, distances, angles of object and light)
CONFIGS = {
    'light_intensities': [
        {'name': 'low', 'main_intensity': 1.0, 'fill_intensity': 0.3},
        {'name': 'medium', 'main_intensity': 3.0, 'fill_intensity': 1.0},
        {'name': 'high', 'main_intensity': 6.0, 'fill_intensity': 2.0}
    ],
    'light_positions': [
        {'name': 'front', 'main_pos': (5, -5, 5), 'fill_pos': (-3, -3, 3), 'back_pos': (0, 4, 3)},
        {'name': 'side', 'main_pos': (8, 0, 4), 'fill_pos': (-3, 3, 3), 'back_pos': (0, 4, 3)},
        {'name': 'back', 'main_pos': (-5, 5, 4), 'fill_pos': (3, -3, 3), 'back_pos': (0, -4, 3)}
    ],
    'distances': [
        {'name': 'close', 'radius': 3.0, 'height': 1.8},
        {'name': 'medium', 'radius': 5.0, 'height': 2.2},
        {'name': 'far', 'radius': 8.0, 'height': 2.8}
    ],
    'object_angles': [
        {'name': 'front_0', 'angle_deg': 0},
        {'name': 'front_right_45', 'angle_deg': 45},
        {'name': 'right_90', 'angle_deg': 90},
        {'name': 'back_right_135', 'angle_deg': 135},
        {'name': 'back_180', 'angle_deg': 180},
        {'name': 'back_left_225', 'angle_deg': 225},
        {'name': 'left_270', 'angle_deg': 270},
        {'name': 'front_left_315', 'angle_deg': 315}
    ]
} 

#setup scene, object, and camera
def setup_scene():
    if bpy.context.selected_objects:
        obj = bpy.context.selected_objects[0]
        print(f"Using selected object: {obj.name}")
    else:
        obj = next((o for o in bpy.context.scene.objects 
                   if o.type == 'MESH' and OBJECT_NAME.lower() in o.name.lower()), None)
        if not obj:
            mesh_objects = [o for o in bpy.context.scene.objects if o.type == 'MESH']
            obj = mesh_objects[0] if mesh_objects else None

    if not obj:
        bpy.ops.mesh.primitive_cube_add()
        obj = bpy.context.active_object
        
    for o in bpy.context.scene.objects:
        if o != obj:
            o.hide_viewport = True         # hide in viewport
            o.hide_render = True           # hide in render
        else:
            o.hide_viewport = False
            o.hide_render = False

    bpy.ops.object.select_all(action='DESELECT')
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj

    if not any(o.type == 'CAMERA' for o in bpy.context.scene.objects):
        bpy.ops.object.camera_add(location=(0, -5, 2))
        camera = bpy.context.active_object
    else:
        camera = next(o for o in bpy.context.scene.objects if o.type == 'CAMERA')
    bpy.context.scene.camera = camera

    return obj, camera

def setup_lights(light_intensity, light_position):
    bpy.ops.object.select_all(action='DESELECT')
    for obj in bpy.context.scene.objects:
        if obj.type == 'LIGHT':
            obj.select_set(True)
    bpy.ops.object.delete()
    
    bpy.ops.object.light_add(type='SUN', location=light_position['main_pos'])
    main_light = bpy.context.active_object
    main_light.data.energy = light_intensity['main_intensity']
    main_light.data.color = (1, 1, 1)
    
    bpy.ops.object.light_add(type='AREA', location=light_position['fill_pos'])
    fill_light = bpy.context.active_object
    fill_light.data.energy = light_intensity['fill_intensity']
    fill_light.data.color = (1, 1, 1)
    fill_light.data.size = 4
    
    bpy.ops.object.light_add(type='POINT', location=light_position['back_pos'])
    back_light = bpy.context.active_object
    back_light.data.energy = light_intensity['main_intensity'] * 0.3
    back_light.data.color = (1, 1, 1)

def position_camera(camera, object_angle, distance_config, target_object):
    angle_rad = math.radians(object_angle['angle_deg'])
    radius = distance_config['radius']
    
    x = radius * math.cos(angle_rad)
    y = radius * math.sin(angle_rad)
    z = distance_config['height']
    
    camera.location = (x, y, z)
    direction = target_object.location - camera.location
    camera.rotation_euler = direction.to_track_quat('-Z', 'Y').to_euler()

#render loop
scene = bpy.context.scene
scene.render.resolution_x = 640
scene.render.resolution_y = 640
scene.render.image_settings.file_format = 'PNG'

os.makedirs(OUTPUT_DIR, exist_ok=True)
target_object, camera = setup_scene()

for light_intensity in CONFIGS['light_intensities']:
    for light_position in CONFIGS['light_positions']:
        print(f"Batch: {light_intensity['name']} intensity, {light_position['name']} light")
        
        setup_lights(light_intensity, light_position)
        
        for distance_config in CONFIGS['distances']:
            for object_angle in CONFIGS['object_angles']:
                position_camera(camera, object_angle, distance_config, target_object)
                
                filename = f"{OBJECT_NAME}_{light_intensity['name']}_{light_position['name']}_{distance_config['name']}_{object_angle['name']}_withoutBackground.png"
                scene.render.filepath = os.path.join(OUTPUT_DIR, filename)
                bpy.ops.render.render(write_still=True)
                
print(f"Completed images for {OBJECT_NAME}")