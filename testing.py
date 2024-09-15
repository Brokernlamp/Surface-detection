from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# Load the trained model
model = load_model('/content/surface_classification_model.h5')  # Path to your saved model

# Define the class labels (same as during training)
class_labels = ['Even', 'Uneven', 'Slippery']

# List of image paths you want to test (replace these with the actual paths)
image_paths = [
    '/content/even.jpeg',
    '/content/images (1).jpeg',
    '/content/slippery.jpeg',
    '/content/uneven.jpeg',
    '/content/eneven2.jpeg'
]

# Function to load and preprocess a single image
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))  # Resize the image
    img_array = image.img_to_array(img)  # Convert to array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize pixel values
    return img_array

# Loop through all the images
for image_path in image_paths:
    # Preprocess the image
    img_array = preprocess_image(image_path)
    
    # Predict the class of the image
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction[0])  # Get the class index with the highest probability
    
    # Get the predicted label
    predicted_label = class_labels[class_index]
    
    # Print and display the result
    print(f'Image: {image_path} --> Predicted Surface: {predicted_label}')
    img = image.load_img(image_path)  # Load the image for display
    plt.imshow(img)
    plt.title(f'Predicted Surface: {predicted_label}')
    plt.show()
