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

def calibrate(image):
    # image = cv2.cvtColor(cv2.VideoCapture(0), cv2.COLOR_BGR2GRAY)
    atDetector = Detector(families="tag36h11", nthreads=1,
                          quad_decimate=1.0, quad_sigma=0.0,
                          refine_edges=1, decode_sharpening=0.25, debug=0)
    mtx = np.loadtxt('cam_intrinsics.txt')
    zeros = np.zeros((mtx.shape[0], 1))
    mtx2 = np.hstack((mtx, zeros))
    camera_params = (mtx[0,0], mtx[1,1], mtx[0,2], mtx[1,2])
    results = atDetector.detect(image, estimate_tag_pose=True, 
                                camera_params=camera_params, tag_size=0.168)
    extrinsics = np.vstack((np.hstack((results[0].pose_R, results[0].pose_t)), [0,0,0,1]))
    extrinsics = np.linalg.inv(extrinsics)
    projection = np.matmul(mtx2, extrinsics)
    return projection