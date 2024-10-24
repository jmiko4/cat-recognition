import cv2
import numpy as np
from picamera2 import Picamera2

# Initialize the camera
picam2 = Picamera2()
# Configure camera with a compatible format for OpenCV
config = picam2.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)})
picam2.configure(config)
picam2.start()

# Define color range for detection 
lower_bound = np.array([10, 40, 60])  # Lower bound for the color
upper_bound = np.array([25, 255, 200])  # Upper bound for the color

while True:
    # Capture image
    frame = picam2.capture_array()

    # Convert BGR to HSV
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Create a mask for the defined color range
    color_mask = cv2.inRange(hsv_frame, lower_bound, upper_bound)

    # Bitwise-AND mask and original image to extract the color
    color_detection = cv2.bitwise_and(frame, frame, mask=color_mask)
    
    # Check how much of the frame is brown
    brown_pixels = cv2.countNonZero(color_mask)
    total_pixels = frame.shape[0] * frame.shape[1]

    # Percentage of brown in the frame
    brown_percentage = (brown_pixels / total_pixels) * 100

    # If a significant amount of brown is detected (threshold can be adjusted)
    if brown_percentage > 10:  # Threshold for detection, adjust as needed
        print("No food for you! (Brown cat detected)")
    else:
        print("Feed cat (Not the brown cat)")
        print(brown_percentage)

    # Display the original and color-detected images
    cv2.imshow('Original', frame)
    cv2.imshow('Color Detected', color_detection)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
picam2.stop()
cv2.destroyAllWindows()
