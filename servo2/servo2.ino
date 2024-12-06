#include <Servo.h>

// Create Servo objects for each motor
Servo motor1;
Servo motor2;
char receivedCommand;

void setup() {
  // Attach motor1 to pin 9 and motor2 to pin 10
  motor1.attach(9);
  motor2.attach(10);

  // Start Serial communication for debugging and receiving data from Raspberry Pi
  Serial.begin(9600);

  // Stop both motors initially
  motor1.write(90); // Stop motor1 (neutral position)
  motor2.write(90); // Stop motor2 (neutral position)
  Serial.println("Ready to receive commands...");
}

void loop() {
  // Check if any data is available to read from the serial port
  if (Serial.available() > 0) {
    // Read the incoming byte
    receivedCommand = Serial.read();

    // Command for motor1
    if (receivedCommand == 'l') {
      Serial.println("Motor1 rotating counterclockwise...");
      motor1.write(0);   // Rotate motor1 counterclockwise
      delay(4000);       // Rotate for 4 seconds

      // Stop motor1
      Serial.println("Stopping motor1...");
      motor1.write(90);  // Stop motor1
    }

    // Command for motor2
    else if (receivedCommand == 's') {
      Serial.println("Motor2 rotating counterclockwise...");
      motor2.write(0);   // Rotate motor2 counterclockwise
      delay(4000);       // Rotate for 4 seconds

      // Stop motor2
      Serial.println("Stopping motor2...");
      motor2.write(90);  // Stop motor2
    }
  }
}
