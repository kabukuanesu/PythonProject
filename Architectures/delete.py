import os
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (accuracy_score, precision_score,
                             recall_score, f1_score,
                             confusion_matrix, classification_report)
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# Define the dataset paths
def get_data_paths():
    data_dir = "C://Users//Anesu Kabuku//PycharmProjects//PythonProject//Datasets//HAM10000//HAM10000"
    metadata_path = "C://Users//Anesu Kabuku//PycharmProjects//PythonProject//Datasets//HAM10000//HAM10000_metadata.csv"
    return data_dir, metadata_path


def load_dataset(data_dir, metadata_path, subset_size, use_preprocessed, image_size=(64, 64)):
    """
    Load and preprocess the dataset based on user preferences.
    """
    if use_preprocessed:
        preprocessed_file = "C://Users//Anesu Kabuku//PycharmProjects//PythonProject//Datasets//HAM10000//hmnist_28_28_RGB.csv"
        X, y, label_encoder = load_and_preprocess_data(data_dir, metadata_path, subset_size, image_size,
                                                       use_preprocessed=True, preprocessed_file=preprocessed_file)
    else:
        X, y, label_encoder = load_and_preprocess_data(data_dir, metadata_path, subset_size, image_size)

    return X, y, label_encoder


def prompt_user():
    subset_size = int(input("Enter the subset size (100, 600, 1600, 3100, 5100, 7600, 10000, 10015): "))
    use_preprocessed = input("Use preprocessed data? (yes/no): ").lower() == "yes"
    return subset_size, use_preprocessed


def load_and_preprocess_data(data_dir, metadata_path, subset_size, image_size, use_preprocessed=False,
                             preprocessed_file=None):
    if use_preprocessed:
        # Load preprocessed data from CSV
        preprocessed_data = pd.read_csv(preprocessed_file)
        X = preprocessed_data.iloc[:, :-1].values  # All columns except the last one
        y = preprocessed_data.iloc[:, -1].values  # Last column is the label

        # Reshape X to match the image dimensions
        if "8_8" in preprocessed_file:
            X = X.reshape(-1, 8, 8, 3 if "RGB" in preprocessed_file else 1)
        elif "28_28" in preprocessed_file:
            X = X.reshape(-1, 28, 28, 3 if "RGB" in preprocessed_file else 1)

        # Normalize pixel values
        X = X / 255.0

        # Encode labels to integers
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y)

        return X, y, label_encoder

    else:
        # Load metadata
        metadata = pd.read_csv(metadata_path)

        # Ensure subset_size is valid
        valid_sizes = [100, 200, 600, 1000, 1600, 3100, 5000, 5100, 7600, 10000, 10015]
        if subset_size not in valid_sizes:
            raise ValueError(f"Invalid subset size. Choose from {valid_sizes}.")

        # Sample the dataset while maintaining class ratio
        subset = metadata.groupby('dx', group_keys=False).apply(
            lambda x: x.sample(int(np.rint(subset_size * len(x) / len(metadata)))).reset_index(drop=True))

        # Debug: Print class distribution in the subset
        print("\nClass distribution in the subset:")
        print(subset['dx'].value_counts())

        # Load images and labels
        image_paths = []
        labels = []

        for index, row in subset.iterrows():
            image_path = os.path.join(data_dir, row['image_id'] + '.jpg')
            try:
                with Image.open(image_path) as img:
                    img = img.resize(image_size)
                    img = np.array(img) / 255.0
                    if len(img.shape) == 2:  # Convert grayscale to RGB
                        img = np.stack([img] * 3, axis=-1)
                    image_paths.append(img)
                    labels.append(row['dx'])
            except (IOError, SyntaxError, PermissionError) as e:
                print(f"Invalid image file {image_path}: {e}")

        # Encode labels to integers
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(labels)
        X = np.array(image_paths)

        return X, y, label_encoder


def evaluate_model(y_test, predictions, label_encoder=None):
    """
    Evaluate model performance with comprehensive metrics.

    Args:
        y_test: True labels
        predictions: Model predictions
        label_encoder: Optional label encoder to show class names in report

    Returns:
        Tuple of (accuracy, precision, recall, f1, conf_matrix)
    """
    # Get class names if label_encoder is provided
    target_names = label_encoder.classes_ if label_encoder else None

    # Calculate metrics
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='macro', zero_division=0)
    recall = recall_score(y_test, predictions, average='macro')
    f1 = f1_score(y_test, predictions, average='macro')
    conf_matrix = confusion_matrix(y_test, predictions)

    # Print detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_test, predictions, target_names=target_names, zero_division=0))

    # Print class distributions
    print("\nClass distribution in predictions:")
    unique_pred, counts_pred = np.unique(predictions, return_counts=True)
    for cls, count in zip(unique_pred, counts_pred):
        cls_name = label_encoder.inverse_transform([cls])[0] if label_encoder else cls
        print(f"{cls_name}: {count}")

    print("\nClass distribution in test set:")
    unique_test, counts_test = np.unique(y_test, return_counts=True)
    for cls, count in zip(unique_test, counts_test):
        cls_name = label_encoder.inverse_transform([cls])[0] if label_encoder else cls
        print(f"{cls_name}: {count}")

    return accuracy, precision, recall, f1, conf_matrix


def split_data(X, y):
    """
    Split data into training and test sets with stratification.
    """
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def get_augmentation_generator():
    """
    Create an image data generator for data augmentation.
    """
    return ImageDataGenerator(
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest'
    )
