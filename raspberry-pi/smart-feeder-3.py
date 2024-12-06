import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter
from picamera2 import Picamera2
import time
import serial
from gpiozero import MotionSensor, Button
import pygame
from datetime import datetime, timedelta

# Initialize the pygame mixer for sound
pygame.mixer.init()
pygame.mixer.music.load('alarm.mp3')  # Load the alarm sound

# Set up the PIR motion sensor and stop button
pir = MotionSensor(17)  # PIR sensor connected to GPIO 17
stop_button = Button(27, pull_up=True)  # Stop button connected to GPIO 27

print("PIR sensor is active. Waiting for motion...")

# Set up the TFLite model
interpreter = Interpreter(model_path='cat_classifier2.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

class_indices = ['Black Cat', 'Brown Cat', 'No Cat']  # Include "No Cat" class

# Initialize the Pi Camera
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)})
picam2.configure(config)

# Function to capture image from the camera
def capture_image():
    frame = picam2.capture_array()
    frame_resized = cv2.resize(frame, (224, 224))
    img_array = np.array(frame_resized).astype('float32')
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = (img_array / 127.5) - 1  # Preprocess normalization for MobileNetV2
    return img_array, frame

# Function to classify the cat
def classify_cat(img_array):
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    confidence = predictions[0][predicted_class_index] * 100
    return class_indices[predicted_class_index], confidence

# Function to send signal to Arduino via serial
def send_servo_signal():
    try:
        ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
        time.sleep(2)
        ser.write(b'r')
        ser.close()
    except Exception as e:
        print(f"Error with serial communication: {e}")

# Function to play the alarm sound
def play_alarm():
    pygame.mixer.music.play()

# Tracking Variables
feed_interval = timedelta(hours=4)
max_feedings_per_day = 4
last_feed_time = None
feeding_times = []
brown_cat_alert_window = timedelta(minutes=10)
alert_active = False

# Main loop
picam2.start()

try:
    while True:

        pir.wait_for_motion()
        print("Motion detected! Capturing image...")

        # Capture and classify the cat
        img_array, original_frame = capture_image()
        cat_type, confidence = classify_cat(img_array)

        print(f"The model predicts: {cat_type} with {confidence:.2f}% confidence.")

        current_time = datetime.now()
        
        # Define feedback based on last feed time
        if last_feed_time is None:
            feed_time_text = "Waiting for cat detection..."
        else:
            time_since_last_feed = (current_time - last_feed_time).total_seconds() // 60
            feed_time_text = f"Time since last feed: {int(time_since_last_feed)} min"

        # Only proceed if it's not a "No Cat" result
        if cat_type == 'Black Cat' and confidence > 95:
            if (last_feed_time is None or (current_time - last_feed_time) >= feed_interval) and len(feeding_times) < max_feedings_per_day:
                print("Black Cat detected with high confidence. Dispensing food...")
                send_servo_signal()
                last_feed_time = current_time
                feeding_times.append(current_time)
                alert_active = True

                feeding_times = [ft for ft in feeding_times if (current_time - ft) < timedelta(days=1)]

        elif cat_type == 'Brown Cat' and confidence > 95 and alert_active:
            if (current_time - last_feed_time) <= brown_cat_alert_window:
                print("Brown Cat detected within 10 minutes of feeding. Playing alarm...")
                play_alarm()
                
        # Display the frame with the prediction and feed time information
        display_frame = original_frame.copy()
        text = f"Predicted: {cat_type} ({confidence:.2f}%)"
        cv2.putText(display_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(display_frame, feed_time_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imshow("Cat Detection", display_frame)
        
        # Check for stop button press
        if stop_button.is_pressed:
            print("Stop button pressed. Exiting program...")
            break  # Exit the main loop

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Exit signal received from keyboard.")
            break

        pir.wait_for_no_motion()
        print("No motion detected. Waiting for next motion...")

except KeyboardInterrupt:
    print("Exiting program due to KeyboardInterrupt...")

finally:
    # Cleanup
    picam2.stop()
    cv2.destroyAllWindows()
    pygame.mixer.quit()
    print("System has been safely shut down.")
