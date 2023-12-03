import cv2
from dt_apriltags import Detector
import numpy as np
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import matplotlib.pyplot as plt
import socket
import pickle
from calibrate import calibrate
from picamera2 import Picamera2
import time

s = socket.socket()
port = 25517
addr = '192.168.8.121'
s.connect((addr, port))

def detection():
    picam = Picamera2()
    config = picam.create_video_configuration({"size": (1920, 1080)})
    picam.configure(config)
    picam.start_preview()
    picam.start()
    image = cv2.rotate(cv2.cvtColor(picam.capture_array("main"), cv2.COLOR_BGR2GRAY), cv2.ROTATE_180)
    projection = calibrate(image)
    base_options = python.BaseOptions(model_asset_path='pose_landmarker_lite.task')
    options = vision.PoseLandmarkerOptions(base_options=base_options, 
                                       output_segmentation_masks=False,
                                       min_pose_detection_confidence=0.8,
                                       min_tracking_confidence=0.8,
                                       min_pose_presence_confidence=0.8)
    detector = vision.PoseLandmarker.create_from_options(options)

    while True:
        # start_time = time.time()
        status = True
        try:
            image = picam.capture_array("main")
        except:
            print('L')
            status = False
        # end_time = time.time()
        # latency = end_time - start_time
        # print(f"Latency of picam.capture_array: {latency} seconds")
        image = cv2.rotate(image, cv2.ROTATE_180)
        if status:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            imagemp = mp.Image(image_format=mp.ImageFormat.SRGB, data=image)
            detection_result = detector.detect(imagemp)
            resultscoords = []
            if detection_result.pose_landmarks:
                for x in range(0, 33):
                    pose = [detection_result.pose_landmarks[0][x].x * imagemp.width,
                            detection_result.pose_landmarks[0][x].y * imagemp.height]
                    resultscoords.append(pose)
                    cv2.circle(image, (int(pose[0]), int(pose[1])), 5, (0, 255, 0), -1)
                resultscoords = (np.array(resultscoords)).T
                # print(resultscoords)
                # end_time2 = time.time()
                # latency2 = end_time2 - end_time
                # print(f"Latency of pose estimation: {latency2} seconds")
            else:
                resultscoords = None
            sendData = {
                "coords": resultscoords,
                "projection": projection
            }
            data = pickle.dumps(sendData)
            s.send(data)
            # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # cv2.imshow('Webcam Footage', image)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     break
        else:
            print("camera not working")

if __name__ == "__main__":
    detection()
    s.close()
    cv2.destroyAllWindows()