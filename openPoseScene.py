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
import matplotlib.pyplot as plt

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
points_3d = (points_3d / points_3d[3])[:3]
print(points_3d)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

# Extract the x, y, and z coordinates
x = points_3d[0,:]
y = points_3d[1,:]
z = points_3d[2,:]

ax.plot(xs = [x[11], x[23]], ys = [y[11], y[23]], zs = [z[11], z[23]], linewidth = 4, c = 'r')
ax.plot(xs = [x[23], x[25]], ys = [y[23], y[25]], zs = [z[23], z[25]], linewidth = 4, c = 'r')
ax.plot(xs = [x[25], x[27]], ys = [y[25], y[27]], zs = [z[25], z[27]], linewidth = 4, c = 'r')
ax.plot(xs = [x[27], x[29]], ys = [y[27], y[29]], zs = [z[27], z[29]], linewidth = 4, c = 'r')
ax.plot(xs = [x[29], x[31]], ys = [y[29], y[31]], zs = [z[29], z[31]], linewidth = 4, c = 'r')
ax.plot(xs = [x[11], x[12]], ys = [y[11], y[12]], zs = [z[11], z[12]], linewidth = 4, c = 'r')
ax.plot(xs = [x[12], x[14]], ys = [y[12], y[14]], zs = [z[12], z[14]], linewidth = 4, c = 'r')
ax.plot(xs = [x[14], x[16]], ys = [y[14], y[16]], zs = [z[14], z[16]], linewidth = 4, c = 'r')
ax.plot(xs = [x[16], x[18]], ys = [y[16], y[18]], zs = [z[16], z[18]], linewidth = 4, c = 'r')
ax.plot(xs = [x[18], x[20]], ys = [y[18], y[20]], zs = [z[18], z[20]], linewidth = 4, c = 'r')
ax.plot(xs = [x[20], x[16]], ys = [y[20], y[16]], zs = [z[20], z[16]], linewidth = 4, c = 'r')
ax.plot(xs = [x[12], x[24]], ys = [y[12], y[24]], zs = [z[12], z[24]], linewidth = 4, c = 'r')
ax.plot(xs = [x[24], x[26]], ys = [y[24], y[26]], zs = [z[24], z[26]], linewidth = 4, c = 'r')
ax.plot(xs = [x[26], x[28]], ys = [y[26], y[28]], zs = [z[26], z[28]], linewidth = 4, c = 'r')
ax.plot(xs = [x[28], x[30]], ys = [y[28], y[30]], zs = [z[28], z[30]], linewidth = 4, c = 'r')
ax.plot(xs = [x[30], x[32]], ys = [y[30], y[32]], zs = [z[30], z[32]], linewidth = 4, c = 'r')
ax.plot(xs = [x[24], x[23]], ys = [y[24], y[23]], zs = [z[24], z[23]], linewidth = 4, c = 'r')
ax.plot(xs = [x[11], x[13]], ys = [y[11], y[13]], zs = [z[11], z[13]], linewidth = 4, c = 'r')
ax.plot(xs = [x[13], x[15]], ys = [y[13], y[15]], zs = [z[13], z[15]], linewidth = 4, c = 'r')
ax.plot(xs = [x[15], x[17]], ys = [y[15], y[17]], zs = [z[15], z[17]], linewidth = 4, c = 'r')
ax.plot(xs = [x[17], x[19]], ys = [y[17], y[19]], zs = [z[17], z[19]], linewidth = 4, c = 'r')
ax.plot(xs = [x[19], x[15]], ys = [y[19], y[15]], zs = [z[19], z[15]], linewidth = 4, c = 'r')


ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')

plt.show()