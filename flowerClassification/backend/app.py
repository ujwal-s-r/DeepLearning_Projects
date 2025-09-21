import logging
import os
import json

from flask import Flask, request, jsonify
from flask_cors import CORS
from predict import predict, loading_model


app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the model once when the app starts
model = loading_model('project_checkpoint.pth')


@app.route('/')
def hello_world():
    return 'Hello, World!'


@app.route('/predict', methods=['POST'])
def predict_flower():
    if request.method == "POST":
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        image = request.files['image']
        
        if image.filename == '':
            return jsonify({'error': 'No image file selected'}), 400
        
        # Save the uploaded image temporarily
        image_path = os.path.join('uploads', image.filename)
        os.makedirs('uploads', exist_ok=True)
        image.save(image_path)
        
        try:
            # Get predictions
            probs, classes = predict(image_path, model, 5)
            
            # Load flower names mapping
            with open('cat_to_name.json', 'r') as f:
                cat_to_name = json.load(f)
            
            # Prepare response
            predictions = []
            for i in range(len(classes)):
                flower_name = cat_to_name.get(classes[i], 'Unknown')
                predictions.append({
                    'class': classes[i],
                    'name': flower_name,
                    'probability': float(probs[i])
                })
            
            # Clean up uploaded file
            os.remove(image_path)
            
            return jsonify({
                'success': True,
                'predictions': predictions
            })
        
        except Exception as e:
            # Clean up uploaded file if it exists
            if os.path.exists(image_path):
                os.remove(image_path)
            return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True)
