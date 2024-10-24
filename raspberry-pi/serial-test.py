import serial
import time

# Adjust the port based on what your Pi assigns to the Arduino (e.g., ttyUSB0 or ttyACM0)
ser = serial.Serial('/dev/ttyACM0', 9600, timeout=1)  
time.sleep(2)  # Wait for the connection to be established

# Send the command to rotate the motor counterclockwise
ser.write(b'r')  # Send the character 'r' as a signal

# Close the serial connection
ser.close()
