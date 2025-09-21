import React from "react";
import Homepage from "./Homepage/Homepage";
import Prediction from "./Prediction/Prediction";
import Image from "./Image/Image";
import { Container } from "react-bootstrap";
import "./style.css";

class App extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      predictions: null
    };
    this.handlePrediction = this.handlePrediction.bind(this);
  }

  handlePrediction(predictions) {
    this.setState({ predictions });
  }

  render() {
    return (
      <>
        <Container fluid className="content">
          <Homepage />
          <Image onPrediction={this.handlePrediction} />
          <Prediction predictions={this.state.predictions} />
        </Container>
      </>
    );
  }
}

export default App;
