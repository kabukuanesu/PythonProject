import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from common_utils import prompt_user, load_dataset, evaluate_model, split_data, get_data_paths

# Prompt user for subset size and preprocessed data preference
subset_size, use_preprocessed = prompt_user()

# Get dataset paths
data_dir, metadata_path = get_data_paths()

# Load and preprocess the dataset
X, y, label_encoder = load_dataset(data_dir, metadata_path, subset_size, use_preprocessed, image_size=(64, 64))

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = split_data(X, y)

# Define the CNN architecture
num_classes = len(np.unique(y))
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=X_train.shape[1:]),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.1,
    zoom_range=0.1,
    horizontal_flip=True
)

# Apply data augmentation to the training data
train_generator = datagen.flow(X_train, y_train, batch_size=32)

# Train the model
model.fit(train_generator, steps_per_epoch=len(X_train) // 32, epochs=10)

# Evaluate the model
predictions = model.predict(X_test)
predictions = np.argmax(predictions, axis=1)

# Calculate evaluation metrics
accuracy, precision, recall, f1, conf_matrix = evaluate_model(y_test, predictions)

print("Accuracy: {:.2f}".format(accuracy))
print("Precision: {:.2f}".format(precision))
print("Recall: {:.2f}".format(recall))
print("F1 Score: {:.2f}".format(f1))
print("Confusion Matrix:")
print(conf_matrix)

# Save the model for later use
model.save(f"cnn_model_{subset_size}.keras")
print(f"Model saved as cnn_model_{subset_size}.keras in Keras")
