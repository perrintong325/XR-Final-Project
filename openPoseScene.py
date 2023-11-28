import bpy
import cv2
from pupil_apriltags import Detector
import numpy as np
import os
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

def draw_landmarks_on_image(rgb_image, detection_result):
  pose_landmarks_list = detection_result.pose_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected poses to visualize.
  for idx in range(len(pose_landmarks_list)):
    pose_landmarks = pose_landmarks_list[idx]

    # Draw the pose landmarks.
    pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    pose_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in pose_landmarks
    ])
    solutions.drawing_utils.draw_landmarks(
      annotated_image,
      pose_landmarks_proto,
      solutions.pose.POSE_CONNECTIONS,
      solutions.drawing_styles.get_default_pose_landmarks_style())
  return annotated_image

imageL = cv2.imread("image_L.jpg")
imageLmp = mp.Image.create_from_file("image_L.jpg")
imageR = cv2.imread("image_R.jpg")
imageRmp = mp.Image.create_from_file("image_R.jpg")

# april tag detection
grayL = cv2.cvtColor(imageL, cv2.COLOR_BGR2GRAY)
grayR = cv2.cvtColor(imageR, cv2.COLOR_BGR2GRAY)
atDetector = Detector(
   families="tag36h11",
   nthreads=1,
   quad_decimate=1.0,
   quad_sigma=0.0,
   refine_edges=1,
   decode_sharpening=0.25,
   debug=0
)
mtx = np.loadtxt('cam_intrinsics.txt')
camera_params = (mtx[0,0], mtx[1,1], mtx[0,2], mtx[1,2])
resultsL = atDetector.detect(grayL, estimate_tag_pose=True, camera_params=camera_params, tag_size=0.249)
resultsR = atDetector.detect(grayR, estimate_tag_pose=True, camera_params=camera_params, tag_size=0.249)
zeros = np.zeros((mtx.shape[0], 1))
mtx2 = np.hstack((mtx, zeros))
extrinsicsL = np.vstack((np.hstack((resultsL[0].pose_R, resultsL[0].pose_t)), [0,0,0,1]))
extrinsicsR = np.vstack((np.hstack((resultsR[0].pose_R, resultsR[0].pose_t)), [0,0,0,1]))
extrinsicsL = np.linalg.inv(extrinsicsL)
extrinsicsR = np.linalg.inv(extrinsicsR)
projectionL = np.matmul(mtx2, extrinsicsL)
projectionR = np.matmul(mtx2, extrinsicsR)

# mediapipe pose detection
base_options = python.BaseOptions(model_asset_path='pose_landmarker_heavy.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
output_segmentation_masks=True)
detector = vision.PoseLandmarker.create_from_options(options)
detection_resultL = detector.detect(imageLmp)
detection_resultR = detector.detect(imageRmp)
resultsLcoords = []
resultsRcoords = []
# 11 face handmarks
for x in range(0, 33, 1):
  l = [detection_resultL.pose_landmarks[0][x].x * imageLmp.width, detection_resultL.pose_landmarks[0][x].y * imageLmp.height]
  r = [detection_resultR.pose_landmarks[0][x].x * imageRmp.width, detection_resultR.pose_landmarks[0][x].y * imageRmp.height]
  resultsLcoords.append(l)
  resultsRcoords.append(r)

resultsLcoords = (np.array(resultsLcoords)).T
resultsRcoords = (np.array(resultsRcoords)).T

annotated_imageL = draw_landmarks_on_image(imageLmp.numpy_view(), detection_resultL)
annotated_imageR = draw_landmarks_on_image(imageRmp.numpy_view(), detection_resultR)

cv2.imwrite("annotated_imageL1.png", annotated_imageL)
cv2.imwrite("annotated_imageR.png", annotated_imageR)

points_3d = cv2.triangulatePoints(projectionL, projectionR, resultsLcoords, resultsRcoords)
points_3d = points_3d / points_3d[3]
print(points_3d)