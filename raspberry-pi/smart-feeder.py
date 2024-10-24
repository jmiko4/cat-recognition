import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter
from picamera2 import Picamera2
import time
import serial
from gpiozero import MotionSensor
import pygame

# --- Initialize the pygame mixer for sound ---
pygame.mixer.init()
# Load the alarm sound (you can use any sound file like "alarm.mp3")
pygame.mixer.music.load('alarm.mp3')

# --- Set up the PIR motion sensor ---
pir = MotionSensor(17)  # PIR sensor connected to GPIO 17
print("PIR sensor is active. Waiting for motion...")

# --- Set up the TFLite model ---
interpreter = Interpreter(model_path='cat_classifier.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

class_indices = ['Black Cat', 'Brown Cat']

# --- Initialize the Pi Camera ---
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)})
picam2.configure(config)

# --- Function to capture image from the camera ---
def capture_image():
    frame = picam2.capture_array()
    frame_resized = cv2.resize(frame, (224, 224))
    img_array = np.array(frame_resized).astype('float32')
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = (img_array / 127.5) - 1  # Preprocess normalization for MobileNetV2
    return img_array, frame  # Return both the processed image and the original frame

# --- Function to classify the cat ---
def classify_cat(img_array):
    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    predictions = interpreter.get_tensor(output_details[0]['index'])
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    confidence = predictions[0][predicted_class_index] * 100
    return class_indices[predicted_class_index], confidence

# --- Function to send signal to Arduino via serial ---
def send_servo_signal():
    try:
        ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1)  # Adjust the port if needed
        time.sleep(2)  # Give time to establish connection
        ser.write(b'r')  # Send signal 'r' to rotate the motor
        ser.close()
    except Exception as e:
        print(f"Error with serial communication: {e}")

# --- Function to play the alarm sound ---
def play_alarm():
    pygame.mixer.music.play()

# --- Main loop ---
picam2.start()
food_dispensed = False  # Track whether food has been dispensed

try:
    while True:
        pir.wait_for_motion()  # Wait for motion detection
        print("Motion detected! Capturing image...")

        # Capture and classify the cat
        img_array, original_frame = capture_image()
        cat_type, confidence = classify_cat(img_array)

        # Display the classification result
        print(f"The model predicts: {cat_type} with {confidence:.2f}% confidence.")

        if cat_type == 'Black Cat' and confidence > 95 and not food_dispensed:
            # If a Black Cat is detected with high confidence and food has not been dispensed
            print("Black Cat detected with high confidence. Dispensing food...")
            send_servo_signal()
            food_dispensed = True  # Mark that food has been dispensed

        elif cat_type == 'Brown Cat' and confidence > 95 and food_dispensed:
            # If a Brown Cat is detected after food has been dispensed, play alarm
            print("Brown Cat detected after food was dispensed. Playing alarm...")
            play_alarm()

        # Display the frame with the prediction
        display_frame = original_frame.copy()
        text = f"Predicted: {cat_type} ({confidence:.2f}%)"
        cv2.putText(display_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Cat Detection", display_frame)

        # Exit the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        pir.wait_for_no_motion()  # Wait for no motion
        print("No motion detected. Waiting for next motion...")

except KeyboardInterrupt:
    print("Exiting program...")

finally:
    picam2.stop()
    cv2.destroyAllWindows()
