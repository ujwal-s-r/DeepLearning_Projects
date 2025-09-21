import React from "react";
import ImageUploader from "react-images-upload";
import { Button, Alert, Spinner } from "react-bootstrap";

class Image extends React.Component {
  constructor(props) {
    super(props);
    this.state = { 
      pictures: [],
      name: "",
      message: "No image uploaded",
      isLoading: false,
      error: null
    };
    this.onDrop = this.onDrop.bind(this);
    this.handlePredict = this.handlePredict.bind(this);
  }

  onDrop(picture) {
    this.setState({
      pictures: this.state.pictures.concat(picture),
      name: `${picture[0].name}`,
      message: "Image Uploaded Successfully",
      error: null
    });
  }

  async handlePredict() {
    if (this.state.pictures.length === 0) {
      this.setState({ error: "Please upload an image first" });
      return;
    }

    this.setState({ isLoading: true, error: null });

    const formData = new FormData();
    formData.append('image', this.state.pictures[this.state.pictures.length - 1]);

    try {
      const response = await fetch('http://localhost:5000/predict', {
        method: 'POST',
        body: formData,
      });

      const data = await response.json();

      if (data.success) {
        // Pass predictions to parent component
        this.props.onPrediction(data.predictions);
        this.setState({ 
          isLoading: false,
          message: "Prediction completed successfully!"
        });
      } else {
        this.setState({ 
          error: data.error || "Prediction failed",
          isLoading: false 
        });
      }
    } catch (error) {
      this.setState({ 
        error: "Failed to connect to server. Make sure the backend is running.",
        isLoading: false 
      });
    }
  }

  render() {
    return (
      <div style={{ margin: "20px 0" }}>
        <ImageUploader
          withIcon={true}
          buttonText="Choose images"
          onChange={this.onDrop}
          imgExtension={[".jpg", ".gif", ".png", ".jpeg"]}
          maxFileSize={5242880}
          singleImage={true}
        />
        
        <div style={{ textAlign: "center", marginTop: "10px" }}>
          <p><strong>{this.state.name}</strong></p>
          <p>{this.state.message}</p>
          
          {this.state.error && (
            <Alert variant="danger">
              {this.state.error}
            </Alert>
          )}
          
          {this.state.pictures.length > 0 && (
            <Button 
              variant="primary" 
              onClick={this.handlePredict}
              disabled={this.state.isLoading}
              style={{ marginTop: "10px" }}
            >
              {this.state.isLoading ? (
                <>
                  <Spinner animation="border" size="sm" className="me-2" />
                  Predicting...
                </>
              ) : (
                "Predict Flower"
              )}
            </Button>
          )}
        </div>
      </div>
    );
  }
}

export default Image;
