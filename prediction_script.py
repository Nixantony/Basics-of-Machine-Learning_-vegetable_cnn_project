# prediction_script.py
# This script helps to check what vegetable is in the image using our trained model

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Load the trained model file
model_path = "vegetable_classifier_model.keras"
model = load_model(model_path)

# List of vegetable names (same order used during training)
class_labels = [
    "Beans", "Bitter_Gourd", "Bottle_Gourd", "Brinjal", "Broccoli",
    "Cabbage", "Capsicum", "Carrot", "Cauliflower", "Cucumber",
    "Papaya", "Potato", "Pumpkin", "Radish", "Tomato"
]

# Set the name of the image we want to test
image_filename = "image3.jpeg"  # You can change this to another image file

# Load the image and resize it so the model can read it
image = load_img(image_filename, target_size=(150, 150))

# Convert the image to numbers and scale it between 0 and 1
image_array = img_to_array(image) / 255.0

# Add one more dimension to make it fit the model input
image_array = np.expand_dims(image_array, axis=0)

# Use the model to predict the vegetable
prediction = model.predict(image_array)

# Find the class with the highest score
predicted_index = np.argmax(prediction[0])

# Get the name of the vegetable from the list
predicted_label = class_labels[predicted_index]

# Print out the result
print("Prediction Result:", predicted_label)    