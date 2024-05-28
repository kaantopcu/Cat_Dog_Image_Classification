from keras.models import load_model
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np


# Load the trained model
model_path = r"C:\Users\User\Desktop\Projects\Cat_Dog_Image_Classification\src\models\cats_and_dogs_small_1.h5"
model = load_model(model_path)

# Function to preprocess the image
def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(150, 150))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# Define the labels
labels = {0: 'Cat', 1: 'Dog'}


uploaded_file = r"C:\Users\User\Desktop\Projects\Cat_Dog_Image_Classification\data\cat1.jpg"

if uploaded_file is not None:
        # Display the uploaded image
    # Make a prediction
    img_array = preprocess_image(uploaded_file)
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)
    predicted_label = labels[predicted_class]