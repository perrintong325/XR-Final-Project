import bpy
import numpy as np

scene = bpy.context.scene

scale = scene.render.resolution_percentage / 100
width = scene.render.resolution_x * scale 
height = scene.render.resolution_y * scale 

camdata = scene.camera.data

focal = camdata.lens 
sensor_width = camdata.sensor_width 
sensor_height = camdata.sensor_height 
pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y

pixel_aspect_ratio = scene.render.pixel_aspect_x / scene.render.pixel_aspect_y
s_u = width / sensor_width
s_v = height * pixel_aspect_ratio / sensor_height
    

alpha_u = focal * s_u
alpha_v = focal * s_v
u_0 = width / 2
v_0 = height / 2
skew = 0 

K = np.array([
    [alpha_u,    skew, u_0],
    [      0, alpha_v, v_0],
    [      0,       0,   1]
], dtype=np.float32)

np.savetxt("cam_intrinsics.txt", K)