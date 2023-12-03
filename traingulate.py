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

received_data = {}
points_3d = None
resultscoords1 = None
resultscoords2 = None
points_3d_record = []
scene = arena.Scene(host="arenaxr.org", scene="final")

def traingulate(projection1, projection2):
    global points_3d, resultscoords1, resultscoords2
    points_3dtemp = cv2.triangulatePoints(projection1, projection2, resultscoords1, resultscoords2)
    points_3d = (points_3dtemp / points_3dtemp[3])[:3]
    points_3d_record.append(points_3d)


def handle_client(client_socket, client_id):
    while True:
        data = client_socket.recv(2048)
        resultscoords = pickle.loads(data)
        received_data[client_id] = resultscoords
        if len(received_data) == 2:
            process_data()

def process_data():
    global resultscoords1, resultscoords2
    projection1 = received_data[0]["projection"]
    projection2 = received_data[1]["projection"]
    if received_data[0]["coords"] is not None:
        resultscoords1 = received_data[0]["coords"]
    if received_data[1]["coords"] is not None:
        resultscoords2 = received_data[1]["coords"]
    if (resultscoords1 is not None) and (resultscoords2 is not None):
        traingulate(projection1, projection2)
        print(len(points_3d_record))
        if (len(points_3d_record) == 100):
            np.save('points_3d_record.npy', points_3d_record)
            print("saved")
            sys.exit()


def server():
    host = socket.gethostname()
    port = 25517
    server_socket = socket.socket()
    server_socket.bind((host, port))
    server_socket.listen(2)
    client_id = 0
    while True:
        conn, address = server_socket.accept()
        print("Connection from: " + str(address))
        threading.Thread(target=handle_client, args=(conn, client_id)).start()
        client_id += 1

faketag = arena.Box(
    object_id='faketag',
    position=(-2, -2, 0),
    rotation=(0, 0, 1, 0),
)
circle11 = arena.Circle(
    object_id='circle11',
    position=(0, 0, 0),
    radius=0.01,
    color=(255, 0, 0),
    parent='faketag'
)
circle12 = arena.Circle(
    object_id='circle12',
    position=(0, 0, 0),
    radius=0.01,
    color=(255, 0, 0),
    parent='faketag'
)
circle13 = arena.Circle(
    object_id='circle13',
    position=(0, 0, 0),
    radius=0.01,
    color=(255, 0, 0),
    parent='faketag'
)
circle14 = arena.Circle(
    object_id='circle14',
    position=(0, 0, 0),
    radius=0.01,
    color=(255, 0, 0),
    parent='faketag'
)
circle15 = arena.Circle(
    object_id='circle15',
    position=(0, 0, 0),
    radius=0.01,
    color=(255, 0, 0),
    parent='faketag'
)
circle16 = arena.Circle(
    object_id='circle16',
    position=(0, 0, 0),
    radius=0.01,
    color=(255, 0, 0),
    parent='faketag'
)
circle17 = arena.Circle(
    object_id='circle17',
    position=(0, 0, 0),
    radius=0.01,
    color=(255, 0, 0),
    parent='faketag'
)
circle18 = arena.Circle(
    object_id='circle18',
    position=(0, 0, 0),
    radius=0.01,
    color=(255, 0, 0),
    parent='faketag'
)
circle19 = arena.Circle(
    object_id='circle19',
    position=(0, 0, 0),
    radius=0.01,
    color=(255, 0, 0),
    parent='faketag'
)
circle20 = arena.Circle(
    object_id='circle20',
    position=(0, 0, 0),
    radius=0.01,
    color=(255, 0, 0),
)
circle21 = arena.Circle(
    object_id='circle21',
    position=(0, 0, 0),
    radius=0.01,
    color=(255, 0, 0),
)
circle22 = arena.Circle(
    object_id='circle22',
    position=(0, 0, 0),
    radius=0.01,
    color=(255, 0, 0),
)
circle23 = arena.Circle(
    object_id='circle23',
    position=(0, 0, 0),
    radius=0.01,
    color=(255, 0, 0),
)
circle24 = arena.Circle(
    object_id='circle24',
    position=(0, 0, 0),
    radius=0.01,
    color=(255, 0, 0),
)
circle25 = arena.Circle(
    object_id='circle25',
    position=(0, 0, 0),
    radius=0.01,
    color=(255, 0, 0),
)
circle26 = arena.Circle(
    object_id='circle26',
    position=(0, 0, 0),
    radius=0.01,
    color=(255, 0, 0),
)
circle27 = arena.Circle(
    object_id='circle27',
    position=(0, 0, 0),
    radius=0.01,
    color=(255, 0, 0),
)
circle28 = arena.Circle(
    object_id='circle28',
    position=(0, 0, 0),
    radius=0.01,
    color=(255, 0, 0),
)
circle29 = arena.Circle(
    object_id='circle29',
    position=(0, 0, 0),
    radius=0.01,
    color=(255, 0, 0),
)
circle30 = arena.Circle(
    object_id='circle30',
    position=(0, 0, 0),
    radius=0.01,
    color=(255, 0, 0),
)
circle31 = arena.Circle(
    object_id='circle31',
    position=(0, 0, 0),
    radius=0.01,
    color=(255, 0, 0),
)
circle32 = arena.Circle(
    object_id='circle32',
    position=(0, 0, 0),
    radius=0.01,
    color=(255, 0, 0),
)
scene.add_object(faketag)
scene.add_object(circle11)
scene.add_object(circle12)
scene.add_object(circle13)
scene.add_object(circle14)
scene.add_object(circle15)
scene.add_object(circle16)
scene.add_object(circle17)
scene.add_object(circle18)
scene.add_object(circle19)
scene.add_object(circle20)
scene.add_object(circle21)
scene.add_object(circle22)
scene.add_object(circle23)
scene.add_object(circle24)
scene.add_object(circle25)
scene.add_object(circle26)
scene.add_object(circle27)
scene.add_object(circle28)
scene.add_object(circle29)
scene.add_object(circle30)
scene.add_object(circle31)
scene.add_object(circle32)
# scene.add_object(circle33)

@scene.run_forever(interval_ms=25)
def main():
    global points_3d
    if points_3d is not None:
        print("updating")
        circle11.update_attributes(position=arena.Position(points_3d[0][11], points_3d[1][11], points_3d[2][11]))
        circle12.update_attributes(position=arena.Position(points_3d[0][12], points_3d[1][12], points_3d[2][12]))
        circle13.update_attributes(position=arena.Position(points_3d[0][13], points_3d[1][13], points_3d[2][13]))
        circle14.update_attributes(position=arena.Position(points_3d[0][14], points_3d[1][14], points_3d[2][14]))
        circle15.update_attributes(position=arena.Position(points_3d[0][15], points_3d[1][15], points_3d[2][15]))
        circle16.update_attributes(position=arena.Position(points_3d[0][16], points_3d[1][16], points_3d[2][16]))
        circle17.update_attributes(position=arena.Position(points_3d[0][17], points_3d[1][17], points_3d[2][17]))
        circle18.update_attributes(position=arena.Position(points_3d[0][18], points_3d[1][18], points_3d[2][18]))
        circle19.update_attributes(position=arena.Position(points_3d[0][19], points_3d[1][19], points_3d[2][19]))
        circle20.update_attributes(position=arena.Position(points_3d[0][20], points_3d[1][20], points_3d[2][20]))
        circle21.update_attributes(position=arena.Position(points_3d[0][21], points_3d[1][21], points_3d[2][21]))
        circle22.update_attributes(position=arena.Position(points_3d[0][22], points_3d[1][22], points_3d[2][22]))
        circle23.update_attributes(position=arena.Position(points_3d[0][23], points_3d[1][23], points_3d[2][23]))
        circle24.update_attributes(position=arena.Position(points_3d[0][24], points_3d[1][24], points_3d[2][24]))
        circle25.update_attributes(position=arena.Position(points_3d[0][25], points_3d[1][25], points_3d[2][25]))
        circle26.update_attributes(position=arena.Position(points_3d[0][26], points_3d[1][26], points_3d[2][26]))
        circle27.update_attributes(position=arena.Position(points_3d[0][27], points_3d[1][27], points_3d[2][27]))
        circle28.update_attributes(position=arena.Position(points_3d[0][28], points_3d[1][28], points_3d[2][28]))
        circle29.update_attributes(position=arena.Position(points_3d[0][29], points_3d[1][29], points_3d[2][29]))
        circle30.update_attributes(position=arena.Position(points_3d[0][30], points_3d[1][30], points_3d[2][30]))
        circle31.update_attributes(position=arena.Position(points_3d[0][31], points_3d[1][31], points_3d[2][31]))
        circle32.update_attributes(position=arena.Position(points_3d[0][32], points_3d[1][32], points_3d[2][32]))
        # circle33.update_attributes(position=arena.Position(points_3d[0][33], points_3d[1][33], points_3d[2][33]))
        scene.update_object(circle11)
        scene.update_object(circle12)
        scene.update_object(circle13)
        scene.update_object(circle14)
        scene.update_object(circle15)
        scene.update_object(circle16)
        scene.update_object(circle17)
        scene.update_object(circle18)
        scene.update_object(circle19)
        scene.update_object(circle20)
        scene.update_object(circle21)
        scene.update_object(circle22)
        scene.update_object(circle23)
        scene.update_object(circle24)
        scene.update_object(circle25)
        scene.update_object(circle26)
        scene.update_object(circle27)
        scene.update_object(circle28)
        scene.update_object(circle29)
        scene.update_object(circle30)
        scene.update_object(circle31)
        scene.update_object(circle32)
        # scene.update_object(circle33)

points_3ds = np.load('points_3d_record.npy')
points_3ds *= 5
for point_3d in points_3ds:
    points_3d = point_3d
scene.run_tasks()
# server()
# tests
