# predict.py
import argparse
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

# Load the trained model
model = load_model("ImranIsicModle.h5")

# Preprocess the input image
def preprocess_image(image_path, target_size=(28, 28)):
    img = load_img(image_path, target_size=target_size)
    img_array = img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = img_array / 255.0  # Normalize pixel values
    return img_array

# Make a prediction
def predict(image_path):
    img_array = preprocess_image(image_path)
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    class_names = ['AK', 'BCC', 'BKL', 'DF', 'MEL', 'NV', 'SCC', 'VASC']
    print(f"Predicted Class: {class_names[predicted_class]}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Predict skin lesion class from an image.")
    parser.add_argument('--image_path', type=str, required=True, help="Path to the input image.")
    args = parser.parse_args()
    predict(args.image_path)
