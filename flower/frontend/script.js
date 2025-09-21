class FlowerClassifier {
    constructor() {
        this.selectedFile = null;
        this.apiUrl = 'http://localhost:5000/predict';
        this.initializeElements();
        this.attachEventListeners();
    }

    initializeElements() {
        this.uploadArea = document.getElementById('upload-area');
        this.fileInput = document.getElementById('file-input');
        this.imagePreview = document.getElementById('image-preview');
        this.previewImg = document.getElementById('preview-img');
        this.removeBtn = document.getElementById('remove-btn');
        this.classifyBtn = document.getElementById('classify-btn');
        this.resultsSection = document.getElementById('results-section');
        this.errorSection = document.getElementById('error-section');
        this.predictionCard = document.getElementById('prediction-card');
        this.predictedClass = document.getElementById('predicted-class');
        this.confidence = document.getElementById('confidence');
        this.allPredictions = document.getElementById('all-predictions');
        this.errorMessage = document.getElementById('error-message');
        this.btnText = document.querySelector('.btn-text');
        this.loading = document.querySelector('.loading');
    }

    attachEventListeners() {
        // Upload area click
        this.uploadArea.addEventListener('click', () => {
            this.fileInput.click();
        });

        // File input change
        this.fileInput.addEventListener('change', (e) => {
            this.handleFileSelection(e.target.files[0]);
        });

        // Drag and drop
        this.uploadArea.addEventListener('dragover', (e) => {
            e.preventDefault();
            this.uploadArea.classList.add('dragover');
        });

        this.uploadArea.addEventListener('dragleave', (e) => {
            e.preventDefault();
            this.uploadArea.classList.remove('dragover');
        });

        this.uploadArea.addEventListener('drop', (e) => {
            e.preventDefault();
            this.uploadArea.classList.remove('dragover');
            const files = e.dataTransfer.files;
            if (files.length > 0) {
                this.handleFileSelection(files[0]);
            }
        });

        // Remove button
        this.removeBtn.addEventListener('click', (e) => {
            e.stopPropagation();
            this.removeImage();
        });

        // Classify button
        this.classifyBtn.addEventListener('click', () => {
            this.classifyImage();
        });
    }

    handleFileSelection(file) {
        if (!file) return;

        // Validate file type
        if (!file.type.startsWith('image/')) {
            this.showError('Please select a valid image file.');
            return;
        }

        // Validate file size (max 10MB)
        if (file.size > 10 * 1024 * 1024) {
            this.showError('File size too large. Please select an image under 10MB.');
            return;
        }

        this.selectedFile = file;
        this.displayImagePreview(file);
        this.hideError();
    }

    displayImagePreview(file) {
        const reader = new FileReader();
        reader.onload = (e) => {
            this.previewImg.src = e.target.result;
            this.uploadArea.style.display = 'none';
            this.imagePreview.style.display = 'block';
            this.classifyBtn.disabled = false;
            this.hideResults();
        };
        reader.readAsDataURL(file);
    }

    removeImage() {
        this.selectedFile = null;
        this.uploadArea.style.display = 'block';
        this.imagePreview.style.display = 'none';
        this.classifyBtn.disabled = true;
        this.fileInput.value = '';
        this.hideResults();
        this.hideError();
    }

    async classifyImage() {
        if (!this.selectedFile) return;

        this.setLoading(true);
        this.hideResults();
        this.hideError();

        try {
            const formData = new FormData();
            formData.append('image', this.selectedFile);

            const response = await fetch(this.apiUrl, {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                throw new Error(`HTTP error! status: ${response.status}`);
            }

            const result = await response.json();
            
            if (result.error) {
                throw new Error(result.error);
            }

            this.displayResults(result);
        } catch (error) {
            console.error('Error:', error);
            this.showError(`Classification failed: ${error.message}`);
        } finally {
            this.setLoading(false);
        }
    }

    displayResults(result) {
        // Display main prediction
        this.predictedClass.textContent = `🌸 ${result.predicted_class}`;
        this.confidence.textContent = `Confidence: ${(result.confidence * 100).toFixed(1)}%`;

        // Display all predictions
        this.allPredictions.innerHTML = '<strong>All Predictions:</strong><br>';
        result.all_predictions.forEach(pred => {
            const predictionItem = document.createElement('div');
            predictionItem.className = 'prediction-item';
            predictionItem.innerHTML = `
                <span>${pred.class}</span>
                <span>${(pred.probability * 100).toFixed(1)}%</span>
            `;
            this.allPredictions.appendChild(predictionItem);
        });

        this.resultsSection.style.display = 'block';
    }

    setLoading(isLoading) {
        this.classifyBtn.disabled = isLoading;
        if (isLoading) {
            this.btnText.style.display = 'none';
            this.loading.style.display = 'inline';
        } else {
            this.btnText.style.display = 'inline';
            this.loading.style.display = 'none';
        }
    }

    showError(message) {
        this.errorMessage.textContent = message;
        this.errorSection.style.display = 'block';
    }

    hideError() {
        this.errorSection.style.display = 'none';
    }

    hideResults() {
        this.resultsSection.style.display = 'none';
    }
}

// Initialize the application when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    new FlowerClassifier();
});