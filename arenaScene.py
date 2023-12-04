import socket
import threading
import pickle
import cv2
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import sys
import arena
import time
from multiprocessing import Process, Queue
from shared_memory_dict import SharedMemoryDict

scene = arena.Scene(host="arenaxr.org", scene="final")
smd_config = SharedMemoryDict(name='data', size=2048)
points_3d_record = np.load('points_3d_record.npy')

faketag = arena.Box(
    object_id='faketag',
    position=(-2, -2, 0),
    rotation=(0, 0, 1, 0),
)
circle11 = arena.Circle(
    object_id='circle11',
    position=(0, 0, 0),
    radius=0.1,
    color=(255, 0, 0),
    parent='faketag'
)
circle12 = arena.Circle(
    object_id='circle12',
    position=(0, 0, 0),
    radius=0.1,
    color=(255, 0, 0),
    parent='faketag'
)
circle13 = arena.Circle(
    object_id='circle13',
    position=(0, 0, 0),
    radius=0.1,
    color=(255, 0, 0),
    parent='faketag'
)
circle14 = arena.Circle(
    object_id='circle14',
    position=(0, 0, 0),
    radius=0.1,
    color=(255, 0, 0),
    parent='faketag'
)
circle15 = arena.Circle(
    object_id='circle15',
    position=(0, 0, 0),
    radius=0.1,
    color=(255, 0, 0),
    parent='faketag'
)
circle16 = arena.Circle(
    object_id='circle16',
    position=(0, 0, 0),
    radius=0.1,
    color=(255, 0, 0),
    parent='faketag'
)
circle17 = arena.Circle(
    object_id='circle17',
    position=(0, 0, 0),
    radius=0.1,
    color=(255, 0, 0),
    parent='faketag'
)
circle18 = arena.Circle(
    object_id='circle18',
    position=(0, 0, 0),
    radius=0.1,
    color=(255, 0, 0),
    parent='faketag'
)
circle19 = arena.Circle(
    object_id='circle19',
    position=(0, 0, 0),
    radius=0.1,
    color=(255, 0, 0),
    parent='faketag'
)
circle20 = arena.Circle(
    object_id='circle20',
    position=(0, 0, 0),
    radius=0.1,
    color=(255, 0, 0),
)
circle21 = arena.Circle(
    object_id='circle21',
    position=(0, 0, 0),
    radius=0.1,
    color=(255, 0, 0),
)
circle22 = arena.Circle(
    object_id='circle22',
    position=(0, 0, 0),
    radius=0.1,
    color=(255, 0, 0),
)
circle23 = arena.Circle(
    object_id='circle23',
    position=(0, 0, 0),
    radius=0.1,
    color=(255, 0, 0),
)
circle24 = arena.Circle(
    object_id='circle24',
    position=(0, 0, 0),
    radius=0.1,
    color=(255, 0, 0),
)
circle25 = arena.Circle(
    object_id='circle25',
    position=(0, 0, 0),
    radius=0.1,
    color=(255, 0, 0),
)
circle26 = arena.Circle(
    object_id='circle26',
    position=(0, 0, 0),
    radius=0.1,
    color=(255, 0, 0),
)
circle27 = arena.Circle(
    object_id='circle27',
    position=(0, 0, 0),
    radius=0.1,
    color=(255, 0, 0),
)
circle28 = arena.Circle(
    object_id='circle28',
    position=(0, 0, 0),
    radius=0.1,
    color=(255, 0, 0),
)
circle29 = arena.Circle(
    object_id='circle29',
    position=(0, 0, 0),
    radius=0.1,
    color=(255, 0, 0),
)
circle30 = arena.Circle(
    object_id='circle30',
    position=(0, 0, 0),
    radius=0.1,
    color=(255, 0, 0),
)
circle31 = arena.Circle(
    object_id='circle31',
    position=(0, 0, 0),
    radius=0.1,
    color=(255, 0, 0),
)
circle32 = arena.Circle(
    object_id='circle32',
    position=(0, 0, 0),
    radius=0.1,
    color=(255, 0, 0),
)
line1 = arena.Line(
        object_id="line1",
        start=(-2, -2, 0),
        end=(-2, -2, 0),
        color=(255,0,0)
)
line2 = arena.Line(
        object_id="line2",
        start=(-2, -2, 0),
        end=(-2, -2, 0),
        color=(255,0,0)
)
line3 = arena.Line(
        object_id="line3",
        start=(-2, -2, 0),
        end=(-2, -2, 0),
        color=(255,0,0)
)
line4 = arena.Line(
        object_id="line4",
        start=(-2, -2, 0),
        end=(-2, -2, 0),
        color=(255,0,0)
)   
line5 = arena.Line(
        object_id="line5",
        start=(-2, -2, 0),
        end=(-2, -2, 0),
        color=(255,0,0)
)
line6 = arena.Line(
        object_id="line6",
        start=(-2, -2, 0),
        end=(-2, -2, 0),
        color=(255,0,0)
)
line7 = arena.Line(
        object_id="line7",
        start=(-2, -2, 0),
        end=(-2, -2, 0),
        color=(255,0,0)
)   
line8 = arena.Line(
        object_id="line8",
        start=(-2, -2, 0),
        end=(-2, -2, 0),
        color=(255,0,0)
)
line9 = arena.Line(
        object_id="line9",
        start=(-2, -2, 0),
        end=(-2, -2, 0),
        color=(255,0,0)
)
line10 = arena.Line(
        object_id="line10",
        start=(-2, -2, 0),
        end=(-2, -2, 0),
        color=(255,0,0)
)
line11 = arena.Line(
        object_id="line11",
        start=(-2, -2, 0),
        end=(-2, -2, 0),
        color=(255,0,0)
)
line12 = arena.Line(
        object_id="line12",
        start=(-2, -2, 0),
        end=(-2, -2, 0),
        color=(255,0,0)
)
line13 = arena.Line(
        object_id="line13",
        start=(-2, -2, 0),
        end=(-2, -2, 0),
        color=(255,0,0)
)
line14 = arena.Line(    
        object_id="line14",
        start=(-2, -2, 0),
        end=(-2, -2, 0),
        color=(255,0,0)
)
line15 = arena.Line(
        object_id="line15",
        start=(-2, -2, 0),
        end=(-2, -2, 0),
        color=(255,0,0)
)
line16 = arena.Line(
        object_id="line16",
        start=(-2, -2, 0),
        end=(-2, -2, 0),
        color=(255,0,0)
)
line17 = arena.Line(
        object_id="line17",
        start=(-2, -2, 0),
        end=(-2, -2, 0),
        color=(255,0,0)
)
line18 = arena.Line(
        object_id="line18",
        start=(-2, -2, 0),
        end=(-2, -2, 0),
        color=(255,0,0)
)
line19 = arena.Line(
        object_id="line19",
        start=(-2, -2, 0),
        end=(-2, -2, 0),
        color=(255,0,0)
)
line20 = arena.Line(
        object_id="line20",
        start=(-2, -2, 0),
        end=(-2, -2, 0),
        color=(255,0,0)
)
line21 = arena.Line(
        object_id="line21",
        start=(-2, -2, 0),
        end=(-2, -2, 0),
        color=(255,0,0)
)
line22 = arena.Line(
        object_id="line22",
        start=(-2, -2, 0),
        end=(-2, -2, 0),
        color=(255,0,0)
)
scene.add_object(faketag)
scene.add_object(line1)
scene.add_object(line2)
scene.add_object(line3)
scene.add_object(line4)
scene.add_object(line5)
scene.add_object(line6)
scene.add_object(line7)
scene.add_object(line8)
scene.add_object(line9)
scene.add_object(line10)
scene.add_object(line11)
scene.add_object(line12)
scene.add_object(line13)
scene.add_object(line14)
scene.add_object(line15)
scene.add_object(line16)
scene.add_object(line17)
scene.add_object(line18)
scene.add_object(line19)
scene.add_object(line20)
scene.add_object(line21)
scene.add_object(line22)

# scene.add_object(circle11)
# scene.add_object(circle12)
# scene.add_object(circle13)
# scene.add_object(circle14)
# scene.add_object(circle15)
# scene.add_object(circle16)
# scene.add_object(circle17)
# scene.add_object(circle18)
# scene.add_object(circle19)
# scene.add_object(circle20)
# scene.add_object(circle21)
# scene.add_object(circle22)
# scene.add_object(circle23)
# scene.add_object(circle24)
# scene.add_object(circle25)
# scene.add_object(circle26)
# scene.add_object(circle27)
# scene.add_object(circle28)
# scene.add_object(circle29)
# scene.add_object(circle30)
# scene.add_object(circle31)
# scene.add_object(circle32)
# scene.add_object(circle33)

@scene.run_forever(interval_ms=10)
def main():
    # global points_3d, points_3d_record
    # points_3d = points_3d_record[0]
    # points_3d_record = points_3d_record[1:] 
    # if points_3d is not None:
    if smd_config['points_3d'] is not None:
        points_3d = smd_config['points_3d']
        print("updating")
        points_3d *= 5
        print(points_3d)   
        points_3d[2,:] = -points_3d[2,:]
        x = points_3d[0,:]
        z = points_3d[1,:]
        y = points_3d[2,:] 
        circle11.update_attributes(position=arena.Position(x[11], y[11], z[11]))
        circle12.update_attributes(position=arena.Position(x[12], y[12], z[12]))
        circle13.update_attributes(position=arena.Position(x[13], y[13], z[13]))
        circle14.update_attributes(position=arena.Position(x[14], y[14], z[14]))
        circle15.update_attributes(position=arena.Position(x[15], y[15], z[15]))
        circle16.update_attributes(position=arena.Position(x[16], y[16], z[16]))
        circle17.update_attributes(position=arena.Position(x[17], y[17], z[17]))
        circle18.update_attributes(position=arena.Position(x[18], y[18], z[18]))
        circle19.update_attributes(position=arena.Position(x[19], y[19], z[19]))
        circle20.update_attributes(position=arena.Position(x[20], y[20], z[20]))
        circle21.update_attributes(position=arena.Position(x[21], y[21], z[21]))
        circle22.update_attributes(position=arena.Position(x[22], y[22], z[22]))
        circle23.update_attributes(position=arena.Position(x[23], y[23], z[23]))
        circle24.update_attributes(position=arena.Position(x[24], y[24], z[24]))
        circle25.update_attributes(position=arena.Position(x[25], y[25], z[25]))
        circle26.update_attributes(position=arena.Position(x[26], y[26], z[26]))
        circle27.update_attributes(position=arena.Position(x[27], y[27], z[27]))
        circle28.update_attributes(position=arena.Position(x[28], y[28], z[28]))
        circle29.update_attributes(position=arena.Position(x[29], y[29], z[29]))
        circle30.update_attributes(position=arena.Position(x[30], y[30], z[30]))
        circle31.update_attributes(position=arena.Position(x[31], y[31], z[31]))
        circle32.update_attributes(position=arena.Position(x[32], y[32], z[32]))
        # circle33.update_attributes(position=arena.Position(points_3d[0][33], points_3d[1][33], points_3d[2][33]))
        # scene.update_object(circle11)
        # scene.update_object(circle12)
        # scene.update_object(circle13)
        # scene.update_object(circle14)
        # scene.update_object(circle15)
        # scene.update_object(circle16)
        # scene.update_object(circle17)
        # scene.update_object(circle18)
        # scene.update_object(circle19)
        # scene.update_object(circle20)
        # scene.update_object(circle21)
        # scene.update_object(circle22)
        # scene.update_object(circle23)
        # scene.update_object(circle24)
        # scene.update_object(circle25)
        # scene.update_object(circle26)
        # scene.update_object(circle27)
        # scene.update_object(circle28)
        # scene.update_object(circle29)
        # scene.update_object(circle30)
        # scene.update_object(circle31)
        # scene.update_object(circle32)
        # scene.update_object(circle33)
        line1.update_attributes(start=(x[11], y[11], z[11]), end=(x[23], y[23], z[23]))
        print(x[11], y[11], z[11], x[23], y[23], z[23])
        line2.update_attributes(start=(x[23], y[23], z[23]), end=(x[25], y[25], z[25]))
        line3.update_attributes(start=(x[25], y[25], z[25]), end=(x[27], y[27], z[27]))
        line4.update_attributes(start=(x[27], y[27], z[27]), end=(x[29], y[29], z[29]))
        line5.update_attributes(start=(x[29], y[29], z[29]), end=(x[31], y[31], z[31]))
        line6.update_attributes(start=(x[11], y[11], z[11]), end=(x[12], y[12], z[12]))
        line7.update_attributes(start=(x[12], y[12], z[12]), end=(x[14], y[14], z[14]))
        line8.update_attributes(start=(x[14], y[14], z[14]), end=(x[16], y[16], z[16]))
        line9.update_attributes(start=(x[16], y[16], z[16]), end=(x[18], y[18], z[18]))
        line10.update_attributes(start=(x[18], y[18], z[18]), end=(x[20], y[20], z[20]))
        line11.update_attributes(start=(x[20], y[20], z[20]), end=(x[16], y[16], z[16]))
        line12.update_attributes(start=(x[12], y[12], z[12]), end=(x[24], y[24], z[24]))
        line13.update_attributes(start=(x[24], y[24], z[24]), end=(x[26], y[26], z[26]))
        line14.update_attributes(start=(x[26], y[26], z[26]), end=(x[28], y[28], z[28]))
        line15.update_attributes(start=(x[28], y[28], z[28]), end=(x[30], y[30], z[30]))
        line16.update_attributes(start=(x[30], y[30], z[30]), end=(x[32], y[32], z[32]))
        line17.update_attributes(start=(x[24], y[24], z[24]), end=(x[23], y[23], z[23]))
        line18.update_attributes(start=(x[11], y[11], z[11]), end=(x[13], y[13], z[13]))
        line19.update_attributes(start=(x[13], y[13], z[13]), end=(x[15], y[15], z[15]))
        line20.update_attributes(start=(x[15], y[15], z[15]), end=(x[17], y[17], z[17]))
        line21.update_attributes(start=(x[17], y[17], z[17]), end=(x[19], y[19], z[19]))
        line22.update_attributes(start=(x[19], y[19], z[19]), end=(x[15], y[15], z[15]))
        scene.update_object(line1)
        scene.update_object(line2)
        scene.update_object(line3)
        scene.update_object(line4)
        scene.update_object(line5)
        scene.update_object(line6)
        scene.update_object(line7)
        scene.update_object(line8)
        scene.update_object(line9)
        scene.update_object(line10)
        scene.update_object(line11)
        scene.update_object(line12)
        scene.update_object(line13)
        scene.update_object(line14)
        scene.update_object(line15)
        scene.update_object(line16)
        scene.update_object(line17)
        scene.update_object(line18)
        scene.update_object(line19)
        scene.update_object(line20)
        scene.update_object(line21)
        scene.update_object(line22)
    else:
        print("empty")

if __name__ == '__main__':
    scene.run_tasks()

