# evaluate_model.py
from tensorflow.keras.models import load_model
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import os
import pandas as pd
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

# Load the trained model
model = load_model("ImranIsicModle.h5")

# Load test data
def load_test_data():
    dataset_path = "data/ISIC_2019_Training_Input"  # Path to the dataset folder
    metadata_path = "data/ISIC_2019_Training_GroundTruth.csv"  # Path to the metadata file

    # Load metadata
    metadata = pd.read_csv(metadata_path)
    class_names = metadata.columns[1:]  # Skip the first column (image ID)
    
    # Load images and labels
    all_images, all_labels = [], []
    for idx, row in metadata.iterrows():
        image_path = os.path.join(dataset_path, row['image'] + ".jpg")
        if os.path.exists(image_path):
            img = load_img(image_path, target_size=(28, 28))
            img_array = img_to_array(img)
            all_images.append(img_array)
            all_labels.append(np.argmax(row[1:].values))  # Get the class label

    # Convert to numpy arrays
    X_all = np.array(all_images)
    y_all = np.array(all_labels)

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_all)
    y_one_hot = to_categorical(y_encoded)

    # Reshape data for oversampling
    X_reshaped = X_all.reshape(X_all.shape[0], -1)
    oversampler = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = oversampler.fit_resample(X_reshaped, y_one_hot)

    # Reshape back to image format
    X_resampled = X_resampled.reshape(-1, 28, 28, 3)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
    return X_test, y_test

# Evaluate the model
def evaluate():
    X_test, y_test = load_test_data()
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
