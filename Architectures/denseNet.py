import numpy as np
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from common_utils import prompt_user, load_dataset, evaluate_model, split_data, get_data_paths

# Prompt user for subset size and preprocessed data preference
subset_size, use_preprocessed = prompt_user()

# Get dataset paths
data_dir, metadata_path = get_data_paths()

# Load and preprocess the dataset
X, y, label_encoder = load_dataset(data_dir, metadata_path, subset_size, use_preprocessed, image_size=(224, 224))

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = split_data(X, y)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    rescale=1./255
)

# Create data generators for training and testing
train_generator = datagen.flow(X_train, y_train, batch_size=32)
test_generator = datagen.flow(X_test, y_test, batch_size=32)

# Load the DenseNet model with pre-trained ImageNet weights
base_model = DenseNet121(weights='imagenet', include_top=False)

# Add custom classification head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(len(np.unique(y)), activation='softmax')(x)

# Create the final model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_generator, steps_per_epoch=len(X_train) // 32, epochs=10)

# Evaluate the model
predictions = model.predict(test_generator)
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
model.save(f"densenet_model_{subset_size}.keras")
print(f"Model saved as densenet_model_{subset_size}.keras in Keras")
