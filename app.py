import sys
import os
from flask import Flask, request, render_template, jsonify
import cv2
import numpy as np

# Ensure 'src' is recognized as a module
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from src.face_recognizer import initialize_camera, load_names

app = Flask(__name__)

# Load the camera
recognizer = cv2.face.LBPHFaceRecognizer_create()
names = load_names('names.json')
recognizer.read('trainer.yml')
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/recognize', methods=['POST'])
def recognize():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    # Save uploaded image
    file_path = os.path.join('media', file.filename)
    file.save(file_path)
    
    # Read image
    image = cv2.imread(file_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    results = []
    for (x, y, w, h) in faces:
        id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
        name = names.get(str(id), "Unknown")
        results.append({'name': name, 'confidence': f'{confidence:.1f}%'})
    
    return jsonify({'result': results})

if __name__ == '__main__':
    app.run(debug=True)