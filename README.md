# Skin Lesion Classification Using CNN

This repository contains the code and resources for training a Convolutional Neural Network (CNN) to classify skin lesions into 8 categories using the ISIC 2019 dataset. The model is designed to assist in the automated diagnosis of skin diseases, which can be useful for medical professionals and researchers.

---

## Table of Contents
1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Data Preprocessing](#data-preprocessing)
4. [Model Architecture](#model-architecture)
5. [Training](#training)
6. [Evaluation](#evaluation)
7. [Results](#results)
8. [Usage](#usage)
9. [Contributing](#contributing)
10. [License](#license)

---

## Introduction
This project involves training a CNN model to classify skin lesions into 8 categories:
- **AK** (Actinic Keratosis)
- **BCC** (Basal Cell Carcinoma)
- **BKL** (Benign Keratosis)
- **DF** (Dermatofibroma)
- **MEL** (Melanoma)
- **NV** (Melanocytic Nevus)
- **SCC** (Squamous Cell Carcinoma)
- **VASC** (Vascular Lesion)

The model is trained on the **ISIC 2019 dataset**, which contains 25,331 images. The goal is to achieve high accuracy in classifying skin lesions to assist in early diagnosis and treatment.

---

## Dataset
The dataset used in this project is the **ISIC 2019 Training Dataset**, which can be downloaded from [Kaggle](https://www.kaggle.com/datasets/salviohexia/isic-2019-skin-lesion-images-for-classification) and contains 25,331 images of skin lesions. The dataset is split into 8 classes, and the images are resized to 28x28 pixels for training.

### Dataset Summary
- **Total Images**: 25,331
- **Classes**: 8 (AK, BCC, BKL, DF, MEL, NV, SCC, VASC)
- **Image Size**: 28x28 pixels
- **Dataset Source**: [ISIC 2019 Challenge](https://challenge.isic-archive.com/data/)

### Preprocessing
- Images are resized to 28x28 pixels.
- Labels are one-hot encoded.
- Data is balanced using **Random Oversampling** to address class imbalance.

---

## Data Preprocessing
The dataset is preprocessed to prepare it for training:
1. **Image Loading**: Images are loaded and resized to 28x28 pixels.
2. **Label Encoding**: Labels are converted to one-hot encoded vectors.
3. **Data Balancing**: Random oversampling is applied to balance the dataset.
4. **Train-Test Split**: The dataset is split into training (80%) and testing (20%) sets.

---

## Model Architecture
The CNN model is designed with the following architecture:
- **Input Layer**: 28x28x3 (RGB images)
- **Convolutional Layers**: 3 layers with 32, 64, and 128 filters respectively.
- **Max Pooling**: Applied after each convolutional layer.
- **Batch Normalization**: Used to stabilize training.
- **Dropout**: Added to prevent overfitting.
- **Dense Layers**: 3 fully connected layers with 256, 128, and 64 units.
- **Output Layer**: 8 units with softmax activation for multi-class classification.

### Model Summary
```python
Model: Sequential
_________________________________________________________________
Layer (type)                 Output Shape              Param #   
=================================================================
Conv2D (Conv2D)              (None, 28, 28, 32)        896       
MaxPooling2D (MaxPooling2D)  (None, 14, 14, 32)        0         
BatchNormalization (BatchNor (None, 14, 14, 32)        128       
Conv2D (Conv2D)              (None, 14, 14, 64)        18496     
MaxPooling2D (MaxPooling2D)  (None, 7, 7, 64)          0         
BatchNormalization (BatchNor (None, 7, 7, 64)          256       
Conv2D (Conv2D)              (None, 7, 7, 128)         73856     
MaxPooling2D (MaxPooling2D)  (None, 3, 3, 128)         0         
BatchNormalization (BatchNor (None, 3, 3, 128)         512       
Flatten (Flatten)            (None, 1152)              0         
Dropout (Dropout)            (None, 1152)              0         
Dense (Dense)                (None, 256)               295168    
BatchNormalization (BatchNor (None, 256)               1024      
Dense (Dense)                (None, 128)               32896     
BatchNormalization (BatchNor (None, 128)               512       
Dense (Dense)                (None, 64)                8256      
BatchNormalization (BatchNor (None, 64)                256       
Dense (Dense)                (None, 8)                 520       
=================================================================
Total params: 432,776
Trainable params: 431,240
Non-trainable params: 1,536
_________________________________________________________________
```

---

## Training
The model is trained for 100 epochs with the following configuration:
- **Optimizer**: Adam
- **Loss Function**: Categorical Crossentropy
- **Batch Size**: 64
- **Callbacks**: ReduceLROnPlateau, EarlyStopping

### Training Metrics
- **Training Accuracy**: 98.03%
- **Validation Accuracy**: 96.75%
- **Test Accuracy**: 96.86%

---

## Evaluation
The model is evaluated on the test set, and the following metrics are reported:
- **Test Accuracy**: 96.86%
- **Test Loss**: 0.2712

### Classification Report
```plaintext
              precision    recall  f1-score   support

          AK       0.99      1.00      1.00      2582
         BCC       0.94      0.99      0.97      2665
         BKL       0.96      0.99      0.97      2582
          DF       1.00      1.00      1.00      2517
         MEL       0.92      0.93      0.92      2612
          NV       0.94      0.84      0.89      2557
         SCC       0.99      1.00      1.00      2550
        VASC       1.00      1.00      1.00      2535

    accuracy                           0.97     20600
   macro avg       0.97      0.97      0.97     20600
weighted avg       0.97      0.97      0.97     20600
```

### Confusion Matrix
![image](https://github.com/user-attachments/assets/b885f4ad-c8b5-4908-9eaa-1622d25b8b52)

---

## Results
### What You Did
- **Data Preprocessing**: Resized images to 28x28 pixels, one-hot encoded labels, and balanced the dataset using random oversampling.
- **Model Development**: Designed and trained a CNN model with 3 convolutional layers, batch normalization, dropout, and dense layers.
- **Training**: Trained the model for 100 epochs using the Adam optimizer and categorical crossentropy loss.
- **Evaluation**: Evaluated the model on a test set and reported accuracy, precision, recall, and F1-score.

### Why You Did It
- **Medical Application**: Skin lesion classification is crucial for early diagnosis and treatment of skin diseases. Automating this process can assist medical professionals in making faster and more accurate diagnoses.
- **Class Imbalance**: The dataset had imbalanced classes, so random oversampling was used to ensure the model learns from all classes equally.
- **Model Performance**: By using a CNN with batch normalization and dropout, we aimed to achieve high accuracy while preventing overfitting.

### What Were the Results
- **High Accuracy**: The model achieved a **test accuracy of 96.86%**, demonstrating its ability to classify skin lesions effectively.
- **Strong Performance Across Classes**: The model performed exceptionally well for classes like AK, DF, SCC, and VASC, with precision and recall close to 1.
- **Challenging Classes**: Classes like MEL and NV had slightly lower performance, likely due to their visual similarity in the dataset.
- **Confusion Matrix**: The confusion matrix shows that the model makes very few misclassifications, with most errors occurring between visually similar classes like MEL and NV.

---

## Usage
To use this model, follow these steps:

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/skin-lesion-classification.git
   cd skin-lesion-classification
   ```

2. **Install Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Train the Model**:
   ```bash
   python train_model.py
   ```

4. **Evaluate the Model**:
   ```bash
   python evaluate_model.py
   ```

5. **Make Predictions**:
   ```bash
   python predict.py --image_path path/to/image.jpg
   ```

---

## Contributing
Contributions to this project are welcome! If you have any improvements or suggestions, feel free to create a pull request or open an issue.

---

## License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
