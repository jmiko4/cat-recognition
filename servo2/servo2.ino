#include <Servo.h>

// Create a Servo object
Servo myServo;
char receivedCommand;

void setup() {
  // Attach the servo to pin 9
  myServo.attach(9);

  // Start Serial communication for debugging and receiving data from Raspberry Pi
  Serial.begin(9600);

  // Stop the motor initially
  myServo.write(90); // Stop motor (neutral position)
  Serial.println("Ready to receive commands...");
}

void loop() {
  // Check if any data is available to read from the serial port
  if (Serial.available() > 0) {
    // Read the incoming byte
    receivedCommand = Serial.read();

    // If the command is 'r', rotate counterclockwise for 4 seconds
    if (receivedCommand == 'r') {
      Serial.println("Rotating counterclockwise...");
      myServo.write(0);  // Rotate counterclockwise
      delay(4000);       // Rotate for 4 seconds

      // Stop the motor
      Serial.println("Stopping motor...");
      myServo.write(90); // Stop motor
    }
  }
}
