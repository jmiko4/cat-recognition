import cv2
import numpy as np
from tflite_runtime.interpreter import Interpreter
from picamera2 import Picamera2
import time

# Load the TFLite model
interpreter = Interpreter(model_path='cat_classifier.tflite')
interpreter.allocate_tensors()

# Get input and output details for the model
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

# Class labels (based on your training dataset)
class_indices = ['Black Cat', 'Brown Cat']

def capture_image():
    # Capture an image using the Pi Camera
    frame = picam2.capture_array()

    # Resize the frame to match the input size of the model (224x224)
    frame_resized = cv2.resize(frame, (224, 224))

    # Convert the image to a format suitable for the model (float32 and normalization)
    img_array = np.array(frame_resized).astype('float32')
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = (img_array / 127.5) - 1  # MobileNetV2 preprocess normalization

    return img_array, frame  # Return the original frame as well

def classify_cat(img_array):
    # Set the tensor to the interpreter for inference
    interpreter.set_tensor(input_details[0]['index'], img_array)

    # Run inference
    interpreter.invoke()

    # Get the prediction results
    predictions = interpreter.get_tensor(output_details[0]['index'])

    # Determine the predicted class and confidence level
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    confidence = predictions[0][predicted_class_index] * 100

    return class_indices[predicted_class_index], confidence

# Initialize the camera
picam2 = Picamera2()
# Configure camera with a compatible format for OpenCV
config = picam2.create_preview_configuration(main={"format": "RGB888", "size": (640, 480)})
picam2.configure(config)
picam2.start()

try:
    while True:
        # Capture and classify the cat in front of the camera
        img_array, original_frame = capture_image()
        cat_type, confidence = classify_cat(img_array)

        # Prepare the display frame with classification results
        display_frame = original_frame.copy()  # Copy the original frame for display

        # Draw the prediction text on the frame
        text = f"The model predicts: {cat_type} with {confidence:.2f}% confidence."
        cv2.putText(display_frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # Show the frame with the prediction
        cv2.imshow("Cat Detection", display_frame)

        # Wait for 2 seconds before taking another picture
        time.sleep(2)

        # Check for 'q' key to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("Exiting program...")
    picam2.stop()

# Cleanup
cv2.destroyAllWindows()
