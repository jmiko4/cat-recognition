import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter
from picamera2 import Picamera2
import time
import serial
from gpiozero import MotionSensor
import pygame
from datetime import datetime, timedelta

# Initialize the pygame mixer for sound
pygame.mixer.init()
pygame.mixer.music.load('alarm.mp3')  # Load the alarm sound

# Set up the PIR motion sensor
pir = MotionSensor(17)  # PIR sensor connected to GPIO 17
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

# Function to send servo signal to Arduino via serial
def send_servo_signal(motor):
    try:
        ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
        time.sleep(2)
        if motor == 1:
            ser.write(b'r')  # Signal for motor1 (Black Cat)
        elif motor == 2:
            ser.write(b's')  # Signal for motor2 (Scheduled feed)
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

# Schedule for motor2 automatic feeding
scheduled_feed_times = [
    "08:30", "12:30", "16:30", "21:00"
]

# Function to calculate time until next scheduled feed for motor2
def get_next_scheduled_feed_time():
    now = datetime.now()
    for feed_time_str in scheduled_feed_times:
        feed_time = datetime.strptime(feed_time_str, "%H:%M").replace(year=now.year, month=now.month, day=now.day)
        if now < feed_time:
            return feed_time
    # If all times have passed, use the first time on the next day
    return datetime.strptime(scheduled_feed_times[0], "%H:%M").replace(year=now.year, month=now.month, day=now.day) + timedelta(days=1)

# Main loop
picam2.start()

try:
    while True:
        current_time = datetime.now()

        # Calculate countdown for motor1
        if last_feed_time is None or (current_time - last_feed_time) >= feed_interval:
            motor1_countdown_text = "Ready to feed"
        else:
            time_until_next_feed = feed_interval - (current_time - last_feed_time)
            motor1_countdown_text = f"Next feed in {time_until_next_feed.seconds // 3600}h {(time_until_next_feed.seconds // 60) % 60}m"

        # Calculate countdown for motor2
        next_scheduled_feed = get_next_scheduled_feed_time()
        time_until_scheduled_feed = next_scheduled_feed - current_time
        motor2_countdown_text = f"Next scheduled feed in {time_until_scheduled_feed.seconds // 3600}h {(time_until_scheduled_feed.seconds // 60) % 60}m"

        # Check if current time matches any scheduled feed times
        if current_time.strftime("%H:%M") in scheduled_feed_times:
            print(f"Scheduled feeding at {current_time.strftime('%H:%M')}. Dispensing food for motor2...")
            send_servo_signal(2)
            time.sleep(60)  # Wait a minute to prevent repeated dispensing within the same minute

        # Check for motion and capture image
        pir.wait_for_motion()
        print("Motion detected! Capturing image...")

        # Capture and classify the cat
        img_array, original_frame = capture_image()
        cat_type, confidence = classify_cat(img_array)

        print(f"The model predicts: {cat_type} with {confidence:.2f}% confidence.")

        # Define feedback based on last feed time for motor1
        if last_feed_time is None:
            feed_time_text = "Waiting for cat detection..."
        else:
            time_since_last_feed = (current_time - last_feed_time).total_seconds() // 60
            feed_time_text = f"Time since last feed: {int(time_since_last_feed)} min"

        # Only proceed if it's not a "No Cat" result
        if cat_type == 'Black Cat' and confidence > 95:
            if (last_feed_time is None or (current_time - last_feed_time) >= feed_interval) and len(feeding_times) < max_feedings_per_day:
                print("Black Cat detected with high confidence. Dispensing food...")
                send_servo_signal(1)  # Activate motor1 for Black Cat
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
        cv2.putText(display_frame, motor1_countdown_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 128, 255), 2)
        cv2.putText(display_frame, motor2_countdown_text, (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 128, 128), 2)
        cv2.imshow("Cat Detection", display_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        pir.wait_for_no_motion()
        print("No motion detected. Waiting for next motion...")

except KeyboardInterrupt:
    print("Exiting program...")

finally:
    picam2.stop()
    cv2.destroyAllWindows()
