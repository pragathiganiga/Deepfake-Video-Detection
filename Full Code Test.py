#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import cv2
import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.model_selection import train_test_split

# Function to extract frames from a video
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
    
video_path = 'C:\\Users\\Asus\\Downloads\\id0_0000.mp4'
output_dir = 'C:\\Users\\Asus\\OneDrive\\Desktop\\output'

# Call extract_frames function to split video into frames
extract_frames(video_path, output_dir)

frame_width = 128
frame_height = 128

# Preprocessing function
# Here the argument frame_path is the path of a file not a directory
def preprocess_frame(frame_path):
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
            frame_path = os.path.join(subdir, file)
            frame = preprocess_frame(frame_path)
            X.append(frame)
            label = 1 if 'deepfake' in subdir else 0
            y.append(label)
    
    X = np.array(X)
    y = np.array(y)
    X_sequences = create_sequences(X, sequence_length)
    return X_sequences, y

data_dir = 'path_to_your_data_directory'
sequence_length = 10  # Example sequence length

X, y = load_data_and_labels(data_dir, sequence_length)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

# Define LSTM model
model = Sequential()
model.add(LSTM(64, input_shape=(sequence_length, frame_width, frame_height, 3)))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, batch_size=32, epochs=10, validation_data=(X_val, y_val))

loss, accuracy = model.evaluate(X_test, y_test)
print(f'Test loss: {loss:.4f}, Test accuracy: {accuracy:.4f}')


# In[3]:


import numpy as np
print("NumPy version:", np.__version__)


# In[1]:


import tensorflow as tf

print("TensorFlow version:", tf.__version__)


# In[ ]:




