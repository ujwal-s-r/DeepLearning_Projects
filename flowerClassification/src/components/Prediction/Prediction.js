import React from "react";
import { Row, Col, Card, ProgressBar } from "react-bootstrap";

const Prediction = ({ predictions }) => {
  if (!predictions || predictions.length === 0) {
    return (
      <div style={{ textAlign: "center", marginTop: "20px" }}>
        <h3>Upload an image and click "Predict Flower" to see results</h3>
      </div>
    );
  }

  return (
    <div style={{ marginTop: "30px", textAlign: "center" }}>
      <h2>Top 5 Flower Predictions</h2>
      <hr />
      <Row>
        {predictions.map((prediction, index) => (
          <Col md={12} key={index} style={{ marginBottom: "15px" }}>
            <Card>
              <Card.Body>
                <Row>
                  <Col md={8}>
                    <h5 style={{ textAlign: "left", marginBottom: "5px" }}>
                      {index + 1}. {prediction.name}
                    </h5>
                    <p style={{ textAlign: "left", color: "gray", margin: "0" }}>
                      Class: {prediction.class}
                    </p>
                  </Col>
                  <Col md={4}>
                    <div style={{ textAlign: "right" }}>
                      <strong>{(prediction.probability * 100).toFixed(2)}%</strong>
                    </div>
                    <ProgressBar 
                      now={prediction.probability * 100} 
                      variant={index === 0 ? "success" : index === 1 ? "info" : "secondary"}
                      style={{ marginTop: "5px" }}
                    />
                  </Col>
                </Row>
              </Card.Body>
            </Card>
          </Col>
        ))}
      </Row>
    </div>
  );
};

export default Prediction;
