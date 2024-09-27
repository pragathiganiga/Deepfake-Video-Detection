#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model("C:\\Users\\Asus\\OneDrive\\Desktop\\deepfake_model\\my_model.h5")

# Function to extract frames from a video and save them in a specified output directory
def extract_frames(video_path, output_dir):
    cap = cv2.VideoCapture(video_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    frame_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_path = os.path.join(output_dir, f'frame_{frame_count}.jpg')
        cv2.imwrite(frame_path, frame)
        frame_count += 1
    
    cap.release()

frame_width = 128
frame_height = 128

# Preprocessing function
def preprocess_frame(frame_path):
    if not os.path.isfile(frame_path):
        raise FileNotFoundError(f"The file {frame_path} does not exist or is not a file.")
    img = load_img(frame_path, target_size=(frame_width, frame_height))
    img_array = img_to_array(img) / 255.0  # Normalize pixel values
    return img_array

def create_sequences(frames, sequence_length):
    sequences = []
    for i in range(len(frames) - sequence_length + 1):
        sequence = frames[i:i + sequence_length]
        sequences.append(sequence)
    return np.array(sequences)

def predict_video(model, video_path, sequence_length):
    output_dir = 'temp_frames'
    extract_frames(video_path, output_dir)
    
    frames = []
    for frame_file in sorted(os.listdir(output_dir)):
        frame_path = os.path.join(output_dir, frame_file)
        if frame_file.endswith('.jpg'):
            frame = preprocess_frame(frame_path)
            frames.append(frame)
    
    if len(frames) < sequence_length:
        raise ValueError(f"Video is too short to create sequences of length {sequence_length}")
    
    sequences = create_sequences(frames, sequence_length)
    predictions = model.predict(sequences)
    
    # Clean up the temporary frame directory
    for frame_file in os.listdir(output_dir):
        os.remove(os.path.join(output_dir, frame_file))
    os.rmdir(output_dir)
    
    return predictions

# Test the model with a new input video
# Note: Change this path
video_path = "C:\\Users\\Asus\\Downloads\\id0_0001.mp4"
sequence_length = 10  # Use the same sequence length as during training
predictions = predict_video(model, video_path, sequence_length)

# Print predictions
print(predictions)

# Post-process predictions to interpret results
average_prediction = np.mean(predictions)
print(f"Average Prediction: {average_prediction:.4f}")
if average_prediction >= 0.5:
    print("The video is predicted to be a deepfake.")
else:
    print("The video is predicted to be real.")

