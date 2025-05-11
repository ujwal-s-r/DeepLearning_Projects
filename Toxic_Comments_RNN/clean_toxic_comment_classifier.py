# Import required libraries
import tensorflow as tf
import numpy as np
import pandas as pd
import os
import gradio as gr

# Define constants
MAX_FEATURES = 200000
MODEL_PATH = 'toxicModel.h5'
CSV_PATH = os.path.join('data', 'train', 'train.csv')

def load_data_and_model():
    """Load the dataset, create vectorizer, and load the model"""
    print("Loading dataset...")
    df = pd.read_csv(CSV_PATH)
    col_names = df.columns[2:]  # Get toxicity category names
    
    print("Setting up text vectorization...")
    vectorizer = tf.keras.layers.TextVectorization(
        max_tokens=MAX_FEATURES,
        output_sequence_length=1800,
        output_mode='int'
    )
    vectorizer.adapt(df['comment_text'].values)
    
    print("Loading model...")
    model = tf.keras.models.load_model(MODEL_PATH)
    
    return df, vectorizer, model, col_names

def score_comment(comment, vectorizer, model, col_names):
    """Score a comment for toxicity"""
    # Vectorize the input text
    vectorized_comment = vectorizer([comment])
    
    # Make prediction
    results = model.predict(vectorized_comment)
    
    # Format results
    text = ''
    for idx, col in enumerate(col_names):
        text += '{}: {}\n'.format(col, results[0][idx] > 0.5)
    
    return text

def predict_single_comment(text):
    """Function to handle predictions from the Gradio interface"""
    return score_comment(text, vectorizer, model, col_names)

def test_predictions():
    """Test the model with sample comments"""
    test_comments = [
        "Thank you for your help, I really appreciate it!",
        "You are an idiot and should die",
        "I'm not sure if this works, let me try"
    ]
    
    for comment in test_comments:
        result = score_comment(comment, vectorizer, model, col_names)
        print(f"\nComment: {comment}")
        print(result)

# Load data and model
df, vectorizer, model, col_names = load_data_and_model()

# Test predictions
test_predictions()

# Create Gradio interface
print("\nCreating Gradio interface...")
interface = gr.Interface(
    fn=predict_single_comment,
    inputs=gr.Textbox(lines=2, placeholder='Enter a comment to check for toxicity'),
    outputs=gr.Text(),
    title="Toxic Comment Classifier",
    description="Enter a comment to check for different types of toxicity."
)

# Launch the interface
if __name__ == "__main__":
    print("Launching interface...")
    interface.launch(share=False)
