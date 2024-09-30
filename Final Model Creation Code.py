#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, TimeDistributed, Flatten

# Directory where your videos are stored
# Note: Change this path
data_dir = "C:\\Users\\Asus\\OneDrive\\Desktop\\data_set"

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

def load_data_and_labels(data_dir, sequence_length):
    X = []
    y = []
    for subdir, _, files in os.walk(data_dir):
        for file in files:
            if file.endswith('.mp4'):
                video_path = os.path.join(subdir, file)
                output_dir = os.path.join(subdir, os.path.splitext(file)[0] + '_frames')
                extract_frames(video_path, output_dir)
                frames = []
                for frame_file in sorted(os.listdir(output_dir)):
                    frame_path = os.path.join(output_dir, frame_file)
                    if frame_file.endswith('.jpg'):
                        frame = preprocess_frame(frame_path)
                        frames.append(frame)
                if len(frames) >= sequence_length:
                    sequences = create_sequences(frames, sequence_length)
                    X.extend(sequences)
                    label = 1 if 'deepfake' in subdir else 0
                    y.extend([label] * len(sequences))
    
    X = np.array(X)
    y = np.array(y)
    print(f"Total sequences: {len(X)}, Total labels: {len(y)}")  # Debugging statement
    return X, y

sequence_length = 10  # Example sequence length

X, y = load_data_and_labels(data_dir, sequence_length)
if len(X) == 0 or len(y) == 0:
    raise ValueError("No data found. Ensure that the dataset directory is correct and contains valid video files.")

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

# Define LSTM model
model = Sequential()
model.add(TimeDistributed(Flatten(), input_shape=(sequence_length, frame_width, frame_height, 3)))
model.add(LSTM(64))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_val, y_val))

loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test loss: {loss:.4f}, Test accuracy: {accuracy:.4f}')


# # Save Model

# In[ ]:


# Note: Change the path
model.save("C:\\Users\\Asus\\OneDrive\\Desktop\\deepfake_model\\my_model.h5")

