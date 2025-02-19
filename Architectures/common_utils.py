import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from PIL import Image

def load_and_preprocess_data(data_dir, image_size):
    image_paths = []
    labels = []

    classes = os.listdir(data_dir)
    for label, class_name in enumerate(classes):
        class_dir = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_dir):
            continue  # Skip if it's not a directory

        for image_name in os.listdir(class_dir):
            image_path = os.path.join(class_dir, image_name)
            try:
                # Attempt to open the image to check if it's valid
                with Image.open(image_path) as img:
                    img.verify()  # Verify that it is, in fact, an image
                image_paths.append(image_path)
                labels.append(label)
            except (IOError, SyntaxError) as e:
                print(f"Invalid image file {image_path}: {e}")

    return pd.DataFrame({'image_path': image_paths, 'label': labels})

def evaluate_model(y_test, predictions):
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions, average='binary')
    recall = recall_score(y_test, predictions, average='binary')
    f1 = f1_score(y_test, predictions, average='binary')
    conf_matrix = confusion_matrix(y_test, predictions)

    return accuracy, precision, recall, f1, conf_matrix
