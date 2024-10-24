from gpiozero import Servo
from time import sleep

# GPIO Pin connected to the servo (BCM mode)
servo = Servo(18)

# Function to move the servo to a specific angle
def move_servo(angle):
    # Convert angle to a value between -1 (min) and 1 (max) for gpiozero's Servo
    # Assuming that -1 is 0 degrees, 0 is 90 degrees, and 1 is 180 degrees
    value = (angle / 90) - 1
    servo.value = value

try:
    while True:
        print("Moving to 0 degrees")
        move_servo(0)
        sleep(1)

        print("Moving to 90 degrees")
        move_servo(90)
        sleep(1)

        print("Moving to 180 degrees")
        move_servo(180)
        sleep(1)

except KeyboardInterrupt:
    print("Program interrupted")
