import bpy
import cv2
import apriltag
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

imageL = cv2.imread("imageL.jpg")
imageLmp = mp.Image.create_from_file("imageL.jpg")
imageR = cv2.imread("imageR.jpg")
imageRmp = mp.Image.create_from_file("imageR.jpg")

# april tag detection
grayL = cv2.cvtColor(imageL, cv2.COLOR_BGR2GRAY)
grayR = cv2.cvtColor(imageR, cv2.COLOR_BGR2GRAY)
options = apriltag.DetectorOptions(families="tag36h11")
atDetector = apriltag.Detector(options)
resultsL = atDetector.detect(grayL)
resultsR = atDetector.detect(grayR)

# mediapipe pose detection
base_options = python.BaseOptions(model_asset_path='pose_landmarker_heavy.task')
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
output_segmentation_masks=True)
detector = vision.PoseLandmarker.create_from_options(options)
detection_resultL = detector.detect(imageLmp)
detection_resultR = detector.detect(imageRmp)
annotated_imageL = draw_landmarks_on_image(imageLmp.numpy_view(), detection_resultL)
annotated_imageR = draw_landmarks_on_image(imageRmp.numpy_view(), detection_resultR)
cv2.imwrite("annotated_imageL.png", annotated_imageL)
cv2.imwrite("annotated_imageR.png", annotated_imageR)