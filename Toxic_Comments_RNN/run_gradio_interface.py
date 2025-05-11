import tensorflow as tf
import numpy as np
import pandas as pd
import os
import gradio as gr

# Set path to your model and data
MODEL_PATH = os.path.join('toxicModel.h5')
CSV_PATH = os.path.join('data', 'train', 'train.csv')

# Load the dataset to get column names
print("Loading dataset...")
df = pd.read_csv(CSV_PATH)
col_names = df.columns[2:]  # Get toxicity category names

# Set up TextVectorization with the same parameters as during training
print("Setting up text vectorization...")
MAX_FEATURES = 200000
vectorizer = tf.keras.layers.TextVectorization(
    max_tokens=MAX_FEATURES,
    output_sequence_length=1800,
    output_mode='int'
)
vectorizer.adapt(df['comment_text'].values)

# Load the pre-trained model
print("Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

# Define the scoring function
def score_comment(comment):
    # Vectorize the input text
    vectorized_comment = vectorizer([comment])
    
    # Make prediction
    results = model.predict(vectorized_comment)
    
    # Format results
    text = ''
    for idx, col in enumerate(col_names):
        text += '{}: {}\n'.format(col, results[0][idx] > 0.5)
    
    return text

# Create Gradio interface
print("Creating Gradio interface...")
interface = gr.Interface(
    fn=score_comment,
    inputs=gr.Textbox(lines=2, placeholder='Enter a comment to check for toxicity'),
    outputs=gr.Text(),
    title="Toxic Comment Classifier",
    description="Enter a comment to check for different types of toxicity."
)

# Launch the interface
if __name__ == "__main__":
    print("Launching interface...")
    interface.launch(share=False)
