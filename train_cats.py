import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Load the pre-trained MobileNetV2 model without the top layer
base_model = tf.keras.applications.MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the base model layers so we don't modify pre-trained weights
base_model.trainable = False

# Add custom classification layers on top of the base model
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation='relu'),
    layers.Dense(2, activation='softmax')  # Two output classes: Brown Cat, Black Cat
])

# Compile the model (we're using categorical cross-entropy for 2-class classification)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Prepare the data using ImageDataGenerator (for training/validation split)
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)  # Normalize pixel values
train_generator = train_datagen.flow_from_directory(
    'cat_dataset',  # Path to your dataset
    target_size=(224, 224),  # Resize images to 224x224 for MobileNetV2
    batch_size=32,
    class_mode='categorical',
    subset='training'  # Training set (80% of the data)
)
validation_generator = train_datagen.flow_from_directory(
    'cat_dataset',
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'  # Validation set (20% of the data)
)

# Train the model
history = model.fit(train_generator, validation_data=validation_generator, epochs=10)

# Save the model
model.save('cat_classifier.h5')
