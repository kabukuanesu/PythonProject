from common_utils import load_and_preprocess_data, evaluate_model
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

# Load and preprocess the dataset
data_dir = "C://Users//Anesu Kabuku//PycharmProjects//PythonProject//Datasets//ph2_dataset"
image_size = (224, 224)

# Load and preprocess the dataset
data = load_and_preprocess_data(data_dir, image_size)

if data.empty:
    raise ValueError("No valid images found in the specified directory.")

# Split the dataset into training and testing sets
train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    rescale=1./255
)

# Create data generators for training and testing
train_generator = datagen.flow_from_dataframe(
    dataframe=train_data,
    x_col='image_path',
    y_col='label',
    target_size=image_size,
    batch_size=32,
    class_mode='binary'  # Use 'binary' for binary classification
)

test_generator = datagen.flow_from_dataframe(
    dataframe=test_data,
    x_col='image_path',
    y_col='label',
    target_size=image_size,
    batch_size=32,
    class_mode='binary'  # Use 'binary' for binary classification
)

# Load the DenseNet model with pre-trained ImageNet weights
base_model = DenseNet121(weights='imagenet', include_top=False)

# Add custom classification head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(1024, activation='relu')(x)
predictions = Dense(1, activation='sigmoid')(x)

# Create the final model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer=Adam(), loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(train_generator, steps_per_epoch=len(train_data)//32, epochs=10)

# Evaluate the model
predictions = model.predict(test_generator)
predictions = (predictions > 0.5).astype(int)

# Calculate evaluation metrics
accuracy, precision, recall, f1, conf_matrix = evaluate_model(test_data['label'], predictions)

print("Accuracy: {:.2f}".format(accuracy))
print("Precision: {:.2f}".format(precision))
print("Recall: {:.2f}".format(recall))
print("F1 Score: {:.2f}".format(f1))
print("Confusion Matrix:")
print(conf_matrix)

# Save the model for later use
model.save("densenet_ph2_model.keras")
print("Model saved as densenet_ph2_model.keras")
