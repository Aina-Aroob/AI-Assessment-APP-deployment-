from flask import Flask, render_template, request
import numpy as np
import os
import cv2
import tempfile
from tensorflow.keras.models import load_model

app = Flask(__name__)
model = load_model('glasses_model.h5')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return render_template('index.html', prediction='No image uploaded')

    file = request.files['image']
    if file.filename == '':
        return render_template('index.html', prediction='No image selected')

    try:
        # Save file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as temp:
            file.save(temp.name)
            img_path = temp.name

        # Load and preprocess image (for 64x64 RGB input)
        img = cv2.imread(img_path)
        img = cv2.resize(img, (64, 64))  # match model input
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # convert BGR to RGB
        img = img / 255.0  # normalize
        img = np.expand_dims(img, axis=0)  # shape becomes (1, 64, 64, 3)

        prediction = model.predict(img)
        os.remove(img_path)

        prob = prediction[0][0]
        result = "Wearing Glasses" if prob > 0.5 else "Not Wearing Glasses"
        confidence = f"{prob * 100:.2f}% confidence"
        result_class = "success" if prob > 0.5 else "danger"

        return render_template('index.html', prediction=f"{result} ({confidence})", result_class=result_class)

    except Exception as e:
        return render_template('index.html', prediction=f"Error: {str(e)}", result_class="warning")

if __name__ == '__main__':
    app.run(debug=True)