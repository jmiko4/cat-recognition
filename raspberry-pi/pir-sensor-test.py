from gpiozero import MotionSensor
from time import sleep

# Set up the motion sensor on GPIO 17 (Pin 11)
pir = MotionSensor(17)

print("PIR sensor is active. Waiting for motion...")

while True:
    pir.wait_for_motion()
    print("Motion detected!")
    sleep(1)  # Pause for a second after motion is detected
    pir.wait_for_no_motion()
    print("No motion detected.")
