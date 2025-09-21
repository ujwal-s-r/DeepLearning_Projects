from flask import Flask, request, jsonify
from flask_cors import CORS
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
import io
import os

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Define the same model architecture as in your notebook
class TinyCNN(nn.Module):
    def __init__(self, num_classes=5):
        super().__init__()
        def block(cin, cout):
            return nn.Sequential(
                nn.Conv2d(cin, cout, 3, padding=1, bias=False),
                nn.GroupNorm(16, cout),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(2)
            )
        self.features = nn.Sequential(
            block(3, 32),    # 224 -> 112
            block(32, 64),   # 112 -> 56
            block(64, 128),  # 56 -> 28
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Dropout(0.2),
            nn.Linear(128, num_classes)  # logits   
        )

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x)
        return self.head(x)

    @torch.no_grad()
    def get_probabilities(self, x):
        logits = self.forward(x)
        probs = torch.softmax(logits, dim=1)
        return probs

# Global variables
model = None
device = None
transform = None
class_names = None

def load_model():
    """Load the trained model and class names"""
    global model, device, transform, class_names
    
    # Set device
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    # Load model info (contains class names and model state)
    model_path = '../models/flower_classifier_with_info.pth'
    
    # Check if the model file exists
    if not os.path.exists(model_path):
        # Try absolute path as fallback
        absolute_model_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'models', 'flower_classifier_with_info.pth')
        if os.path.exists(absolute_model_path):
            model_path = absolute_model_path
        else:
            raise FileNotFoundError(f"Model file not found at {model_path} or {absolute_model_path}")
    
    print(f"Loading model from: {model_path}")
    
    model_info = torch.load(model_path, map_location=device)
    
    # Extract class names and number of classes
    class_names = model_info['class_names']
    num_classes = model_info['num_classes']
    
    # Initialize model
    model = TinyCNN(num_classes=num_classes)
    model.load_state_dict(model_info['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Define the same transforms as used in training
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    print(f"Model loaded successfully!")
    print(f"Classes: {class_names}")
    print(f"Number of classes: {num_classes}")
    print(f"Device: {device}")
    
    # Test the model with a dummy input to ensure it works
    test_input = torch.randn(1, 3, 224, 224).to(device)
    with torch.no_grad():
        test_output = model(test_input)
        print(f"Model test successful! Output shape: {test_output.shape}")
    
    return True

def preprocess_image(image_file):
    """Preprocess the uploaded image"""
    try:
        # Open and convert image to RGB
        image = Image.open(io.BytesIO(image_file.read())).convert('RGB')
        
        # Apply transforms
        image_tensor = transform(image).unsqueeze(0)  # Add batch dimension
        
        return image_tensor.to(device)
    except Exception as e:
        raise ValueError(f"Error processing image: {str(e)}")

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        # Check if image file is in request
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        image_file = request.files['image']
        
        # Check if file is selected
        if image_file.filename == '':
            return jsonify({'error': 'No image file selected'}), 400
        
        # Preprocess image
        image_tensor = preprocess_image(image_file)
        
        # Make prediction
        with torch.no_grad():
            probabilities = model.get_probabilities(image_tensor)
            predicted_idx = torch.argmax(probabilities, dim=1).item()
            predicted_class = class_names[predicted_idx]
            confidence = probabilities[0][predicted_idx].item()
            
            # Get all predictions
            all_predictions = []
            for i, class_name in enumerate(class_names):
                all_predictions.append({
                    'class': class_name,
                    'probability': probabilities[0][i].item()
                })
            
            # Sort by probability (highest first)
            all_predictions.sort(key=lambda x: x['probability'], reverse=True)
        
        return jsonify({
            'predicted_class': predicted_class,
            'confidence': confidence,
            'all_predictions': all_predictions
        })
        
    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'device': str(device) if device else 'unknown',
        'classes': class_names if class_names else []
    })

if __name__ == '__main__':
    try:
        # Load the model when starting the server
        load_model()
        print("Starting Flask server...")
        app.run(debug=True, host='0.0.0.0', port=5000)
    except Exception as e:
        print(f"Error starting server: {str(e)}")
        print("Make sure the model file exists at '../models/flower_classifier_with_info.pth'")