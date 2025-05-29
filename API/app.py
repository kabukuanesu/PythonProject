from flask import Flask, request, jsonify
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Load the pre-trained model
MODEL_PATH = "C://Users//Anesu Kabuku//PycharmProjects//PythonProject//API//model.keras"
model = load_model(MODEL_PATH)

# Define allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(img_path, target_size=(28, 28)):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # Normalize the image
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join('uploads', filename)
        file.save(file_path)

        # Preprocess the image
        img_array = preprocess_image(file_path)

        # Make prediction
        predictions = model.predict(img_array)
        predicted_class = np.argmax(predictions, axis=1)[0]
        confidence_score = float(np.max(predictions))

        # Map the predicted class to a label (classes are 0: nv, 1: mel, 2: bkl, 3: bcc, 4: akiec, 5: vasc, 6: df.)
        class_labels = ['nv', 'mel', 'bkl', 'bcc', 'akiec', 'vasc', 'df']  # Update this based on your model's classes
        predicted_label = class_labels[predicted_class]

        # Clean up the uploaded file
        os.remove(file_path)

        return jsonify({
            'predicted_class': predicted_label,
            'confidence_score': confidence_score
        })
    else:
        return jsonify({'error': 'Allowed file types are png, jpg, jpeg'}), 400

if __name__ == '__main__':
    # Create uploads directory if it doesn't exist
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(host='0.0.0.0', port=5000)
