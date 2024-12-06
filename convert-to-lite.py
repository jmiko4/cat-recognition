import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model('cat_classifier3.h5')

# Convert the model to TensorFlow Lite format
converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

# Save the TensorFlow Lite model
with open('cat_classifier3.tflite', 'wb') as f:
    f.write(tflite_model)
