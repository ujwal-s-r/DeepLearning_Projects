import tensorflow as tf
import pandas as pd
import numpy as np
import os
import gradio as gr

# Load the saved model
model_path = 'toxicModel.h5'
model = tf.keras.models.load_model(model_path)

# Load the dataset to get column names and create the vectorizer
csv_path = os.path.join('data', 'train', 'train.csv')
df = pd.read_csv(csv_path)
col_names = df.columns[2:]  # Get toxicity category names

# Set up the TextVectorization layer
MAX_FEATURES = 200000
vectorizer = tf.keras.layers.TextVectorization(
    max_tokens=MAX_FEATURES,
    output_sequence_length=1800,
    output_mode='int'
)
vectorizer.adapt(df['comment_text'].values)

# Define the scoring function
def score_comment(comment):
    vectorized_comment = vectorizer([comment])
    results = model.predict(vectorized_comment)
    
    text = ''
    for idx, col in enumerate(col_names):
        text += '{}: {}\n'.format(col, results[0][idx] > 0.5)
    
    return text

# Create and launch the interface
interface = gr.Interface(
    fn=score_comment,
    inputs=gr.Textbox(lines=2, placeholder='Comment to score'),
    outputs=gr.Text(),
    title="Toxic Comment Classifier",
    description="Enter a comment to check for different types of toxicity."
)

# Launch with sharing option
if __name__ == "__main__":
    interface.launch(share=True)
