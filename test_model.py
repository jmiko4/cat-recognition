import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

# Load your trained model
model = tf.keras.models.load_model('cat_classifier.h5')

# Path to the image you want to test
img_path = 'both2.jpg'

# Load the image with the same size you used for training
img = image.load_img(img_path, target_size=(224, 224))  # Use the same size used during training

# Convert the image to an array
img_array = image.img_to_array(img)

# Expand dimensions to match the expected input format of the model (batch size, height, width, channels)
img_array = np.expand_dims(img_array, axis=0)

# Preprocess the image (same preprocessing you did during training)
img_array = preprocess_input(img_array)

# Run prediction
predictions = model.predict(img_array)

# Output the predicted class and confidence
class_indices = ['Black Cat', 'Brown Cat']  # Corresponds to the order of your classes

# Get the index of the highest probability (the predicted class)
predicted_class_index = np.argmax(predictions, axis=1)[0]

# Get the confidence of the prediction (as a percentage)
confidence = predictions[0][predicted_class_index] * 100

# Output the result
print(f"The model predicts: {class_indices[predicted_class_index]} with {confidence:.2f}% confidence.")
