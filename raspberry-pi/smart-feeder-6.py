import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter
from picamera2 import Picamera2
import time
import serial
from datetime import datetime, timedelta
import pygame
import threading
from tkinter import Tk, Label, Button, Frame
from PIL import Image, ImageTk

# Initialize pygame for alarm sound
pygame.mixer.init()
pygame.mixer.music.load('alarm.mp3')  # Load the alarm sound

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

# Function to capture and preprocess an image
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

# Function to send signal to Arduino
def send_servo_signal(motor_id):
    try:
        ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1)
        time.sleep(2)
        if motor_id == 1:
            ser.write(b'l')  # Motor 1 signal
        elif motor_id == 2:
            ser.write(b's')  # Motor 2 signal
        ser.close()
    except Exception as e:
        print(f"Error with serial communication: {e}")

# Function to play alarm sound
def play_alarm():
    pygame.mixer.music.play()

# Main loop for detection
running = True
max_feedings_per_day = 4
last_feed_time = None
feed_interval = timedelta(hours = 4)
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


# GUI Application
class CatFeederApp:
    def __init__(self, master):
        self.master = master
        self.master.title("Cat Feeder System")

        # Set the window size to 800x480
        self.master.geometry("800x480")

        # Disable resizing to prevent elements from spilling outside
        self.master.resizable(False, False)

        # Video feed frame
        self.video_frame = Frame(master, width=800, height=300, bg="black")  # Adjusted size
        self.video_frame.pack_propagate(False)  # Prevent resizing of the frame
        self.video_frame.pack()

        self.video_label = Label(self.video_frame, bg="black")
        self.video_label.pack(expand=True)

        # Manual feed buttons
        self.manual_feed_frame = Frame(master)
        self.manual_feed_frame.pack(pady=10)

        self.feed_button1 = Button(self.manual_feed_frame, text="Feed Motor 1", width=12, height=2, command=lambda: send_servo_signal(1))
        self.feed_button1.pack(side="left", padx=10)

        self.feed_button2 = Button(self.manual_feed_frame, text="Feed Motor 2", width=12, height=2, command=lambda: send_servo_signal(2))
        self.feed_button2.pack(side="left", padx=10)

        # Stop system button
        self.stop_button = Button(master, text="Stop System", width=20, height=2, command=self.stop_system)
        self.stop_button.pack(pady=10)

        self.running = True
        self.camera_active = True  # New flag to track camera state
        self.update_video_feed()

    def stop_system(self):
        global running
        print("Stopping system...")
        running = False
        self.running = False
        if self.camera_active:
            picam2.stop()
        pygame.mixer.quit()
        self.master.destroy()

    def update_video_feed(self):
        """Continuously updates the video feed."""
        if not self.running:
            return

        global last_feed_time, feeding_times, alert_active

        current_time = datetime.now()

        # Check if the camera should be turned off
        if alert_active and (current_time - last_feed_time) > brown_cat_alert_window:
            print("Turning off the camera to save power...")
            picam2.stop()
            self.camera_active = False
            alert_active = False

        # Check if it's time to turn the camera back on
        if not self.camera_active:
            next_scheduled_feed = get_next_scheduled_feed_time()
            if (current_time - last_feed_time) >= feed_interval:  #Turn on for next available feeding
                print("Turning the camera back on...")
                picam2.start()
                self.camera_active = True

        if self.camera_active:
            try:
                # Capture and classify
                img_array, original_frame = capture_image()
                if img_array is None or original_frame is None:
                    raise ValueError("No image captured.")

                # Rotate the frame 90 degrees counter-clockwise
                rotated_frame = cv2.rotate(original_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)

                cat_type, confidence = classify_cat(img_array)
                print(f"Detected: {cat_type} ({confidence:.2f}%)")
                
                # Check feeding conditions
                if cat_type == 'Black Cat' and confidence > 98:
                    if (last_feed_time is None or (current_time - last_feed_time) >= feed_interval) and len(feeding_times) < 4:
                        send_servo_signal(1)
                        last_feed_time = current_time
                        feeding_times.append(current_time)
                        alert_active = True
                elif cat_type == 'Brown Cat' and confidence > 98 and alert_active:
                    if (current_time - last_feed_time) <= brown_cat_alert_window:
                        play_alarm()

                # Overlay classification text on the video frame
                display_frame = rotated_frame.copy()
                text = f"Predicted: {cat_type} ({confidence:.2f}%)"
                cv2.putText(display_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            except Exception as e:
                print(f"Error during image capture or classification: {e}")
                # Display a blank frame with a message
                blank_frame = np.zeros((480, 300, 3), dtype=np.uint8)  # Black frame
                text = "Camera is off or not capturing images."
                cv2.putText(blank_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                rotated_frame = cv2.rotate(blank_frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
                display_frame = rotated_frame.copy()
            
        else:
            # Display a blank frame with a message when the camera is off
            blank_frame = np.zeros((480, 300, 3), dtype=np.uint8)  # Black frame
            text = "Camera is off."
            cv2.putText(blank_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            display_frame = blank_frame.copy()

        # Calculate countdowns
        if last_feed_time is None or (current_time - last_feed_time) >= feed_interval:
            motor1_countdown_text = "Ready to feed"
        else:
            time_until_next_feed = feed_interval - (current_time - last_feed_time)
            motor1_countdown_text = f"Next Lady feed in {time_until_next_feed.seconds // 3600}h {(time_until_next_feed.seconds // 60) % 60}m"

        next_scheduled_feed = get_next_scheduled_feed_time()
        time_until_scheduled_feed = next_scheduled_feed - current_time
        motor2_countdown_text = f"Next Stinky feed in {time_until_scheduled_feed.seconds // 3600}h {(time_until_scheduled_feed.seconds // 60) % 60}m"

        # Add countdown text to the frame
        cv2.putText(display_frame, motor1_countdown_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 128, 255), 2)
        cv2.putText(display_frame, motor2_countdown_text, (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 128, 128), 2)

        # Convert frame to ImageTk format
        image = Image.fromarray(cv2.cvtColor(display_frame, cv2.COLOR_BGR2RGB))
        photo = ImageTk.PhotoImage(image=image)
        self.video_label.config(image=photo)
        self.video_label.image = photo

        # Schedule next frame update
        self.master.after(100, self.update_video_feed)


# Run the GUI
if __name__ == "__main__":
    picam2.start()
    root = Tk()
    app = CatFeederApp(root)

    try:
        root.mainloop()
    finally:
        running = False
        if app.camera_active:
            picam2.stop()
        cv2.destroyAllWindows()
        pygame.mixer.quit()
        print("System shut down.")

