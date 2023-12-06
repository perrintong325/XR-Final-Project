import cv2
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

def yolo_detection(frame):
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    layers_names = net.getUnconnectedOutLayersNames()
    
    height, width, _ = frame.shape
    blob = cv2.dnn.blobFromImage(frame, scalefactor=1/255.0, size=(416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(layers_names)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 0:  # Assuming class 0 corresponds to humans
                center_x, center_y, w, h = (detection[0:4] * np.array([width, height, width, height])).astype(int)
                x, y = int(center_x - w / 2), int(center_y - h / 2)
                class_ids.append(class_id)
                confidences.append(float(confidence))
                boxes.append([x, y, w, h])

    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    return [boxes[i] for i in indices.flatten()] if indices is not None else []

def detection():
    picam = Picamera2()
    config = picam.create_video_configuration({"size": (1920, 1080)})
    picam.configure(config)
    picam.start_preview()
    picam.start()

    projection = calibrate(image)  # Assuming this function returns the calibration matrix

    base_options = python.BaseOptions(model_asset_path='pose_landmarker_lite.task')
    options = vision.PoseLandmarkerOptions(base_options=base_options, 
                                       output_segmentation_masks=False,
                                       min_pose_detection_confidence=0.8,
                                       min_tracking_confidence=0.8,
                                       min_pose_presence_confidence=0.8)
    detector = vision.PoseLandmarker.create_from_options(options)

    while True:
        status = True
        try:
            image = picam.capture_array("main")
        except:
            print('Failed to capture image')
            status = False

        if status:
            image = cv2.rotate(image, cv2.ROTATE_180)

            # YOLO detection
            yolo_boxes = yolo_detection(image)

            for box in yolo_boxes:
                x, y, w, h = box
                roi = image[y:y+h, x:x+w]

                image_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
                imagemp = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

                detection_result = detector.detect(imagemp)
                resultscoords = []

                if detection_result.pose_landmarks:
                    for i in range(33):
                        pose = [detection_result.pose_landmarks[0][i].x * imagemp.width + x,
                                detection_result.pose_landmarks[0][i].y * imagemp.height + y]
                        resultscoords.append(pose)
                        cv2.circle(image, (int(pose[0]), int(pose[1])), 5, (0, 255, 0), -1)

                    sendData = {
                        "coords": resultscoords,
                        "projection": projection
                    }

                    data = pickle.dumps(sendData)
                    s.send(data)

        else:
            print("Camera not working")

if __name__ == "__main__":
    detection()
    s.close()
    cv2.destroyAllWindows()
