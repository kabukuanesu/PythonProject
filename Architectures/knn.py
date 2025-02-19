# Import necessary libraries
import os
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from common_utils import load_and_preprocess_data, evaluate_model

# Load and preprocess the dataset
data_dir = "C://Users//Anesu Kabuku//PycharmProjects//PythonProject//Datasets//ph2_dataset"
image_size = (64, 64)

# Load and preprocess the dataset
X, y = load_and_preprocess_data(data_dir, image_size)

# Flatten the images for KNN
X = X.reshape(len(X), -1)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the KNN model
knn_model = KNeighborsClassifier(n_neighbors=3)  # You can experiment with different values for n_neighbors

# Train the KNN model
knn_model.fit(X_train, y_train)

# Predict on the test set
predictions = knn_model.predict(X_test)

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
joblib.dump(knn_model, "knn_ph2_model.pkl")
print("Model saved as knn_ph2_model.pkl")
