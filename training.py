import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import os

# Path to the dataset
dataset_path = '/content/surfacesa/surfaces'  # Ensure your dataset is correctly uploaded

# Image Data Generator (handles resizing dynamically)
datagen = ImageDataGenerator(rescale=1.0/255.0, validation_split=0.2)  # Scaling pixel values between 0 and 1

# Define target size (the input size expected by MobileNetV2 is 224x224)
target_size = (224, 224)

# Load training and validation data, resizing images as needed
train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=target_size,  # Resize images to 224x224
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

val_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=target_size,  # Resize images to 224x224
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Load pre-trained MobileNetV2 without the top layers
base_model = MobileNetV2(input_shape=(224, 224, 3), include_top=False, weights='imagenet')

# Add custom layers for classification
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)  # Add a dense layer with 128 neurons
predictions = Dense(3, activation='softmax')(x)  # 3 classes: Even, Uneven, Slippery

# Create the model
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the layers of the base model
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_generator, validation_data=val_generator, epochs=10)

# Save the trained model
model.save('/content/surface_classification_model.h5')
#surface_classification_model.h5 is the classifier file downloaded in your environmentt.
