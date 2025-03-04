from sklearn.svm import SVC
import joblib
from common_utils import prompt_user, load_dataset, evaluate_model, split_data, get_data_paths

# Prompt user for subset size and preprocessed data preference
subset_size, use_preprocessed = prompt_user()

# Get dataset paths
data_dir, metadata_path = get_data_paths()

# Load and preprocess the dataset
X, y, label_encoder = load_dataset(data_dir, metadata_path, subset_size, use_preprocessed, image_size=(64, 64))

# Flatten the images for SVM
X = X.reshape(len(X), -1)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = split_data(X, y)

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
joblib.dump(svm_model, f"svm_model_{subset_size}.pkl")
print(f"Model saved as svm_model_{subset_size}.pkl")
