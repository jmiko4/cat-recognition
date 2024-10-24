#include <Servo.h>

Servo myServo;

void setup() {
  myServo.attach(6);  // Attach to pin D9
}

void loop() {
  myServo.write(0);   // Move to 0 degrees
  delay(2000);        // Wait for 2 seconds

  myServo.write(90);  // Move to 90 degrees
  delay(2000);        // Wait for 2 seconds

  myServo.write(180); // Move to 180 degrees
  delay(2000);        // Wait for 2 seconds
}
