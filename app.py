from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
import os
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import tensorflow as tf

# Set TensorFlow logging level to suppress warnings
tf.get_logger().setLevel('ERROR')

app = Flask(__name__)

# Load the model
model_path = os.path.join(os.getcwd(), 'bcd_model.h5')
model = load_model(model_path)

IMAGE_SIZE = (150, 150)  # Same as the model's expected input size

# Directory to temporarily store uploaded files
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def load_and_preprocess_image(img_path, target_size):
    img = image.load_img(img_path, target_size=target_size)
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0
    return img_array

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if the file is in the request
        if 'image' not in request.files:
            return jsonify({'error': 'No file provided'}), 400

        file = request.files['image']

        # Check if the file has a valid name
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400

        # Save the file to the upload directory
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Preprocess the image
        img_array = load_and_preprocess_image(file_path, IMAGE_SIZE)

        # Make prediction
        predictions = model.predict(img_array)
        predicted_class = (predictions[0] > 0.5).astype("int32")
        class_labels = ['benign', 'malignant']
        predicted_label = class_labels[predicted_class[0]]

        # Remove the file after prediction
        os.remove(file_path)

        # Respond with the prediction
        if predicted_label == 'malignant':
            return jsonify({
                'prediction': 'Malignant',
                'message': 'The patient may be suffering from Breast Cancer.'
            })
        else:
            return jsonify({
                'prediction': 'Benign',
                'message': 'The patient is not suffering from Breast Cancer.'
            })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Use Render's dynamic port
    app.run(host='0.0.0.0', port=port)
