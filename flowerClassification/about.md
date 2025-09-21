# Flower Classification Project

## Project Overview

This is a full-stack web application that classifies flower images using deep learning. The project consists of a React frontend for image upload and a Flask backend that serves a trained PyTorch model for prediction.

## Current Architecture

### Frontend (React)
- **Location**: `/src` directory
- **Technology**: React.js with Bootstrap for styling
- **Features**:
  - Image upload interface using `react-images-upload`
  - Prediction results display
  - Responsive design with Bootstrap
- **Status**: ✅ Completed and working

### Backend (Flask)
- **Location**: `/backend` directory
- **Technology**: Flask with PyTorch for ML inference
- **Features**:
  - REST API endpoint for image classification
  - Image preprocessing and prediction
  - JSON response with top 5 predictions
- **Status**: ⚠️ Needs trained model file

### Machine Learning Model
- **Architecture**: ResNet (transfer learning)
- **Dataset**: Oxford 102 Flower Dataset (102 flower categories)
- **Classes**: 102 different flower species
- **Status**: ❌ Not trained yet

## Project Structure

```
flowerClassification/
├── src/                    # React frontend
│   ├── components/
│   │   ├── App.js         # Main app component
│   │   ├── Image.js       # Image upload component
│   │   └── Prediction.js  # Results display component
│   └── index.js           # React entry point
├── public/                # Static files
├── backend/               # Flask backend
│   ├── app.py            # Flask application
│   ├── predict.py        # Prediction logic
│   ├── cat_to_name.json  # Class to flower name mapping
│   └── requirements.txt  # Python dependencies
├── package.json          # Node.js dependencies
└── README.md            # Basic setup instructions
```

## What's Working

1. ✅ **Frontend Development Server**: React app runs successfully with Node.js compatibility fix
2. ✅ **Image Upload Interface**: Users can upload flower images
3. ✅ **Predict Button**: Triggers API call to backend
4. ✅ **Flask Backend Setup**: API endpoint ready to receive requests
5. ✅ **CORS Configuration**: Frontend can communicate with backend
6. ✅ **Class Mapping**: 102 flower categories defined in `cat_to_name.json`

## What Needs to Be Done

### 1. Dataset Preparation
- [ ] Download Oxford 102 Flower Dataset
- [ ] Organize dataset into train/validation/test splits
- [ ] Create data loaders with proper transformations
- [ ] Verify dataset integrity and class distribution

### 2. Model Training (Transfer Learning)
- [ ] **Base Model**: Use pre-trained ResNet18/ResNet50 from torchvision
- [ ] **Model Architecture**:
  - Replace final fully connected layer for 102 classes
  - Add dropout for regularization
  - Implement proper feature extraction vs fine-tuning
- [ ] **Training Configuration**:
  - Loss function: CrossEntropyLoss
  - Optimizer: Adam or SGD with momentum
  - Learning rate scheduling
  - Early stopping based on validation accuracy
- [ ] **Data Augmentation**:
  - Random rotation, flip, crop
  - Color jittering
  - Normalization using ImageNet stats
- [ ] **Training Loop**:
  - Train for multiple epochs
  - Track training/validation loss and accuracy
  - Save best model checkpoint
  - Implement model evaluation metrics

### 3. Model Integration
- [ ] **Model Saving**: Save trained model as `project_checkpoint.pth`
- [ ] **Model Loading**: Update `predict.py` to properly load the checkpoint
- [ ] **Inference Pipeline**: 
  - Image preprocessing (resize, normalize)
  - Model prediction with confidence scores
  - Top-k prediction selection

### 4. Performance Optimization
- [ ] **Model Optimization**:
  - Test different ResNet architectures (ResNet18, ResNet50, ResNet101)
  - Experiment with different training strategies
  - Implement ensemble methods if needed
- [ ] **Backend Optimization**:
  - Add image validation (file type, size)
  - Implement caching for frequently predicted images
  - Add logging and error handling

### 5. Testing and Validation
- [ ] **Model Testing**:
  - Test accuracy on held-out test set
  - Confusion matrix analysis
  - Per-class accuracy evaluation
- [ ] **Integration Testing**:
  - End-to-end testing with real flower images
  - API performance testing
  - Frontend-backend integration testing

### 6. Documentation and Deployment
- [ ] **Documentation**:
  - Training process documentation
  - API documentation
  - Model performance metrics
- [ ] **Deployment Preparation**:
  - Docker containerization
  - Environment configuration
  - Production-ready error handling

## Technical Requirements

### Training Environment
- **GPU**: CUDA-capable GPU recommended for training
- **Memory**: At least 8GB RAM for dataset loading
- **Storage**: ~2GB for dataset + model checkpoints

### Dependencies
```bash
# Backend
torch>=1.9.0
torchvision>=0.10.0
flask>=2.0.0
flask-cors
pillow
numpy

# Frontend
react^16.13.1
react-bootstrap^1.2.2
react-images-upload^1.2.8
```

## Dataset Information

### Oxford 102 Flower Dataset
- **Total Images**: ~8,000 images
- **Classes**: 102 flower categories
- **Split**: ~40 training images per class
- **Image Size**: Variable (needs preprocessing)
- **Challenges**: Class imbalance, similar species, lighting variations

### Download Instructions
```bash
# Download dataset (implement this)
cd backend
python download_dataset.py
```

## Training Script Structure (To Be Implemented)

```python
# train.py structure
1. Data loading and preprocessing
2. Model architecture definition
3. Training configuration
4. Training loop with validation
5. Model checkpoint saving
6. Performance evaluation
```

## Current Issues to Resolve

1. **Missing Model File**: `project_checkpoint.pth` doesn't exist
2. **Dataset**: Need to download and prepare Oxford 102 Flower Dataset
3. **Training Script**: Need to implement transfer learning training
4. **Model Evaluation**: Need to validate model performance

## Next Steps (Priority Order)

1. **Immediate**: Create training script for ResNet transfer learning
2. **Short-term**: Download and prepare dataset
3. **Medium-term**: Train and validate model
4. **Long-term**: Deploy and optimize application

## Success Metrics

- **Model Accuracy**: Target >85% top-1 accuracy on test set
- **Response Time**: <2 seconds for prediction
- **User Experience**: Seamless image upload and result display
- **Reliability**: Robust error handling and model loading

## Contributing Guidelines

1. Follow existing code structure and naming conventions
2. Add proper error handling and logging
3. Test model changes thoroughly before integration
4. Document any new features or model modifications
5. Ensure compatibility between frontend and backend changes

---

**Note**: This project is currently in development phase. The frontend and backend infrastructure is ready, but the core ML model needs to be trained before the application can provide accurate flower classifications.