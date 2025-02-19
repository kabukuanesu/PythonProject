# Import necessary libraries
import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from common_utils import load_and_preprocess_data, evaluate_model

# Load and preprocess the dataset
data_dir = "C://Users//Anesu Kabuku//PycharmProjects//PythonProject//Datasets//ph2_dataset"
image_size = (64, 64)

# Load and preprocess the dataset
X, y = load_and_preprocess_data(data_dir, image_size)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the CNN architecture
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(image_size[0], image_size[1], 3)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(1, activation='sigmoid')
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

# Train the model
train_generator = datagen.flow(X_train, y_train, batch_size=32)
model.fit(train_generator, steps_per_epoch=len(X_train) // 32, epochs=10)

# Evaluate the model
predictions = model.predict(X_test)
predictions = (predictions > 0.5).astype(int)

# Calculate evaluation metrics
accuracy, precision, recall, f1, conf_matrix = evaluate_model(y_test, predictions)

print("Accuracy: {:.2f}".format(accuracy))
print("Precision: {:.2f}".format(precision))
print("Recall: {:.2f}".format(recall))
print("F1 Score: {:.2f}".format(f1))
print("Confusion Matrix:")
print(conf_matrix)

# Save the model for later use
model.save("cnn_ph2_model.keras")
print("Model saved as cnn_ph2_model.keras")
