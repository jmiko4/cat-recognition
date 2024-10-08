from picamera2 import Picamera2
import time

picam2 = Picamera2()
picam2.start()
time.sleep(2)  # Allow the camera to warm up
picam2.capture_file("test_image.jpg")
picam2.stop()

print("Image captured and saved as test_image.jpg")
