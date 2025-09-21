# Flower Classifier Web Application

A complete web application for classifying flower images using a custom PyTorch CNN model.

## Project Structure

```
flower/
├── frontend/           # Web interface
│   ├── index.html     # Main HTML file
│   ├── style.css      # Styling
│   └── script.js      # JavaScript functionality
├── backend/           # Flask API server
│   ├── app.py         # Main Flask application
│   └── requirements.txt # Python dependencies
├── models/            # Trained models
│   └── flower_classifier_with_info.pth
└── data/              # Training data
    └── train/         # Training images organized by class
```

## Features

- **Drag & Drop Interface**: Easy image upload with drag and drop support
- **Real-time Preview**: See your uploaded image before classification
- **AI Predictions**: Get confident predictions with probability scores
- **All Classes**: View predictions for all flower classes
- **Responsive Design**: Works on desktop and mobile devices

## Setup Instructions

### 1. Install Backend Dependencies

```bash
cd backend
pip install -r requirements.txt
```

### 2. Start the Backend Server

```bash
cd backend
python app.py
```

The server will start at `http://localhost:5000`

### 3. Open the Frontend

Simply open `frontend/index.html` in your web browser, or serve it with a local server:

```bash
cd frontend
# Using Python
python -m http.server 8000

# Using Node.js (if you have it)
npx serve .
```

## Usage

1. **Upload Image**: Click the upload area or drag and drop a flower image
2. **Preview**: Review the uploaded image
3. **Classify**: Click "Classify Flower" to get AI predictions
4. **Results**: View the predicted class and confidence scores

## Supported Flower Classes

The model can classify the following flower types:
- (Will be loaded dynamically from your trained model)

## API Endpoints

- `POST /predict` - Upload image and get predictions
- `GET /health` - Check server health and model status

## Technical Details

- **Frontend**: HTML5, CSS3, JavaScript (Vanilla)
- **Backend**: Flask, PyTorch
- **Model**: Custom TinyCNN architecture
- **Image Processing**: PIL, torchvision transforms
- **CORS**: Enabled for cross-origin requests

## Troubleshooting

### Model Not Found Error
Make sure your trained model file exists at `models/flower_classifier_with_info.pth`

### CORS Issues
The backend includes CORS headers. If you still have issues, make sure both frontend and backend are running.

### Image Upload Issues
- Supported formats: JPG, PNG, GIF
- Max file size: 10MB
- Images are automatically resized to 224x224 pixels

## Development

To modify the model or add new features:

1. **Update Model**: Retrain in the Jupyter notebook and save to `models/`
2. **Frontend Changes**: Edit files in `frontend/`
3. **Backend Changes**: Modify `backend/app.py`
4. **Styling**: Update `frontend/style.css`

## Performance

- **Model Size**: Lightweight CNN (< 1MB)
- **Inference Time**: ~100ms on CPU, ~10ms on GPU
- **Memory Usage**: ~500MB with PyTorch loaded