# train_model.py
import os
import pandas as pd
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from imblearn.over_sampling import RandomOverSampler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

# Load and preprocess data
def load_data():
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

    # Balance the dataset
    X_reshaped = X_all.reshape(X_all.shape[0], -1)
    oversampler = RandomOverSampler(random_state=42)
    X_resampled, y_resampled = oversampler.fit_resample(X_reshaped, y_one_hot)

    # Reshape back to image format
    X_resampled = X_resampled.reshape(-1, 28, 28, 3)

    # Split into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

# Define the model
def create_model():
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(0.01), input_shape=(28, 28, 3)),
        MaxPooling2D(),
        BatchNormalization(),
        Conv2D(64, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(0.01)),
        MaxPooling2D(),
        BatchNormalization(),
        Conv2D(128, (3, 3), activation='relu', padding='same', kernel_initializer='he_normal', kernel_regularizer=l2(0.01)),
        MaxPooling2D(),
        BatchNormalization(),
        Flatten(),
        Dropout(0.5),
        Dense(256, activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dense(128, activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dense(64, activation='relu', kernel_initializer='he_normal', kernel_regularizer=l2(0.01)),
        BatchNormalization(),
        Dense(8, activation='softmax')
    ])
    return model

# Train the model
def train():
    X_train, X_test, y_train, y_test = load_data()
    model = create_model()
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    callbacks = [
        ReduceLROnPlateau(monitor='val_loss', patience=3, verbose=1),
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)
    ]
    
    history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100, batch_size=64, callbacks=callbacks)
    model.save("ImranIsicModle.h5")

if __name__ == '__main__':
    train()
