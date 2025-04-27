from flask import Flask, request, render_template, jsonify
import numpy as np
import tensorflow as tf
from tensorflow import keras
from PIL import Image
import os
import json

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Charger le modèle
model = keras.models.load_model('mobilenet_model.h5')

# Charger le fichier de correspondance des classes
with open('class_names.json', 'r') as f:
    idx_to_class = json.load(f)

# Page d'accueil
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

# Route pour faire la prédiction
@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return "No file part", 400

    file = request.files['file']
    if file.filename == '':
        return "No selected file", 400

    # Sauvegarder l'image uploadée
    img_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(img_path)

    # Préparer l'image
    img = Image.open(img_path).resize((224, 224))
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array / 255.0

    # Prédire
    preds = model.predict(img_array)
    predicted_class_index = np.argmax(preds[0])
    predicted_class_label = idx_to_class[str(predicted_class_index)]

    # Nettoyer
    os.remove(img_path)

    return render_template('result.html', 
                           predicted_class=predicted_class_label,
                           confidence=round(float(np.max(preds[0])) * 100, 2))

if __name__ == '__main__':
    app.run(debug=True)
