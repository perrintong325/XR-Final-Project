from picamera2 import Picamera2
import cv2

picam = Picamera2()
config = picam.create_video_configuration({"size": (1920, 1080)})
picam.configure(config)
picam.start_preview()
picam.start()
cv2.imwrite("test.jpg", cv2.rotate(cv2.cvtColor(picam.capture_array("main"), cv2.COLOR_BGR2RGB), cv2.ROTATE_180))