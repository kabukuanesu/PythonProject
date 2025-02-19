# Import necessary libraries
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from common_utils import load_and_preprocess_data, evaluate_model

# Load and preprocess the dataset
data_dir = "C://Users//Anesu Kabuku//PycharmProjects//PythonProject//Datasets//ph2_dataset"
image_size = (64, 64)

# Load and preprocess the dataset
X, y = load_and_preprocess_data(data_dir, image_size)

# Flatten the images for SVM
X = X.reshape(len(X), -1)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the SVM model
svm_model = SVC(kernel='linear')  # You can experiment with different kernels like 'rbf', 'poly', etc.

# Train the SVM model
svm_model.fit(X_train, y_train)

# Predict on the test set
predictions = svm_model.predict(X_test)

# Calculate evaluation metrics
accuracy, precision, recall, f1, conf_matrix = evaluate_model(y_test, predictions)

print("Accuracy: {:.2f}".format(accuracy))
print("Precision: {:.2f}".format(precision))
print("Recall: {:.2f}".format(recall))
print("F1 Score: {:.2f}".format(f1))
print("Confusion Matrix:")
print(conf_matrix)

# Save the model for later use
import joblib
joblib.dump(svm_model, "svm_ph2_model.pkl")
print("Model saved as svm_ph2_model.pkl")
