# evaluate_model.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("skin_lesion_model.h5")

# Load test data
def load_test_data():
    test_dataset_path = "data/ISIC_2019_Test_Input"  # Path to test dataset folder
    test_metadata_path = "data/ISIC_2019_Test_GroundTruth.csv"  # Path to test metadata file

    # Load metadata
    test_metadata = pd.read_csv(test_metadata_path)
    
    test_images, test_labels = [], []
    for _, row in test_metadata.iterrows():
        image_path = os.path.join(test_dataset_path, row['image'] + ".jpg")
        if os.path.exists(image_path):
            img = load_img(image_path, target_size=(28, 28))  # Resize image
            img_array = img_to_array(img) / 255.0  # Normalize pixel values
            test_images.append(img_array)
            test_labels.append(np.argmax(row[1:].values))  # Get class label

    # Convert to numpy arrays
    X_test = np.array(test_images)
    y_test = np.array(test_labels)

    # Encode labels
    label_encoder = LabelEncoder()
    y_test_encoded = label_encoder.fit_transform(y_test)
    y_test_one_hot = to_categorical(y_test_encoded)

    return X_test, y_test_one_hot, label_encoder

# Evaluate the model
def evaluate():
    X_test, y_test, label_encoder = load_test_data()
    
    test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=1)
    print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
    print(f"Test Loss: {test_loss:.4f}")

    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_true = np.argmax(y_test, axis=1)

    print(classification_report(y_true, y_pred, target_names=label_encoder.classes_))

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=label_encoder.classes_)
    disp.plot(cmap='Blues', xticks_rotation='vertical')
    plt.show()

if __name__ == '__main__':
    evaluate()
