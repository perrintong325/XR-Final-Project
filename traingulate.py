import socket
import threading
import pickle
import cv2
import matplotlib.pyplot as plt 
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import sys

received_data = {}
points_3d = None
resultscoords1 = None
resultscoords2 = None
points_3d_record = []

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

server_program()