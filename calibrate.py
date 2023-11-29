import cv2
from pupil_apriltags import Detector
import numpy as np
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import matplotlib.pyplot as plt

def calibrate():
    imageL = cv2.cvtColor(cv2.VideoCapture(0), cv2.COLOR_BGR2GRAY)
    imageR = cv2.cvtColor(cv2.VideoCapture(1), cv2.COLOR_BGR2GRAY)
    atDetector = Detector(families="tag36h11", nthreads=1,
                          quad_decimate=1.0, quad_sigma=0.0,
                          refine_edges=1, decode_sharpening=0.25, debug=0)
    mtx = np.loadtxt('cam_intrinsics.txt')
    camera_params = (mtx[0,0], mtx[1,1], mtx[0,2], mtx[1,2])
    resultsL = atDetector.detect(imageL, estimate_tag_pose=True, 
                                 camera_params=camera_params, tag_size=0.249)
    resultsR = atDetector.detect(imageR, estimate_tag_pose=True,
                                 camera_params=camera_params, tag_size=0.249)
    zeros = np.zeros((mtx.shape[0], 1))
    mtx2 = np.hstack((mtx, zeros))
    extrinsicsL = np.vstack((np.hstack((resultsL[0].pose_R, resultsL[0].pose_t)), [0,0,0,1]))
    extrinsicsR = np.vstack((np.hstack((resultsR[0].pose_R, resultsR[0].pose_t)), [0,0,0,1]))
    extrinsicsL = np.linalg.inv(extrinsicsL)
    extrinsicsR = np.linalg.inv(extrinsicsR)
    projectionL = np.matmul(mtx2, extrinsicsL)
    projectionR = np.matmul(mtx2, extrinsicsR)
    return projectionL, projectionR