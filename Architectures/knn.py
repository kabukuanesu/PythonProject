from sklearn.neighbors import KNeighborsClassifier
import joblib
from common_utils import prompt_user, load_dataset, evaluate_model, split_data, get_data_paths

# Prompt user for subset size and preprocessed data preference
subset_size, use_preprocessed = prompt_user()

# Get dataset paths
data_dir, metadata_path = get_data_paths()

# Load and preprocess the dataset
X, y, label_encoder = load_dataset(data_dir, metadata_path, subset_size, use_preprocessed, image_size=(64, 64))

# Flatten the images for KNN
X = X.reshape(len(X), -1)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = split_data(X, y)

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

# Save the model for later use in JavaScript
joblib.dump(knn_model, f"knn_model_{subset_size}.pkl")
print(f"Model saved as knn_model_{subset_size}.pkl")
