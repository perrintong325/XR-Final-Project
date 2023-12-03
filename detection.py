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

# s = socket.socket()
# port = 517
# addr = '192.168.188.0.1'
# s.connect((addr, port))

def detection():
    picam = Picamera2()
    config = picam.create_video_configuration({"size": (1920, 1080)})
    picam.configure(config)
    picam.start()
    # projection = calibrate(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
    base_options = python.BaseOptions(model_asset_path='final/pose_landmarker_lite.task')
    options = vision.PoseLandmarkerOptions(base_options=base_options, 
                                       output_segmentation_masks=True)
    detector = vision.PoseLandmarker.create_from_options(options)

    while True:
        start_time = time.time()
        image = picam.capture_array("main")
        image = cv2.rotate(image, cv2.ROTATE_180)
        status = True
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
            else:
                resultscoords = np.full((2, 33), -1)
            # data = pickle.dumps([projection, resultscoords])
            # s.send(data)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            # cv2.imshow('Webcam Footage', image)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            print("camera not working")

if __name__ == "__main__":
    detection()
    # s.close()
    cv2.destroyAllWindows()