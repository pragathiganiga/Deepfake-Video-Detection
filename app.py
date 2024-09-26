from flask import Flask, request, render_template, redirect, url_for
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import io
import base64
from PIL import Image
from tensorflow.keras.layers import Flatten

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Load the saved model
model_path = "C:\\Users\\praga\\OneDrive\\Pictures\\Desktop\\deepfake_model\\my_model.h5"
model = load_model(model_path, custom_objects={'Flatten': Flatten})

# Define constants
sequence_length = 10
frame_width = 128
frame_height = 128
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


# Preprocessing function with face detection
def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Crop the first detected face
    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        break
    else:
        face = frame  # Use the entire frame if no face is detected
    
    face = cv2.resize(face, (frame_width, frame_height))
    face_array = img_to_array(face) / 255.0  # Normalize pixel values
    return face_array

def extract_frames(video_path):
    print("----- Frame extraction started -----")
    cap = cv2.VideoCapture(video_path)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    print("----- Frame extraction ended -----")
    return frames

def create_sequences(frames, sequence_length):
    sequences = []
    for i in range(len(frames) - sequence_length + 1):
        sequence = frames[i:i + sequence_length]
        sequences.append(sequence)
    return np.array(sequences)

def predict_video(video_path, sequence_length):
    frames = extract_frames(video_path)
    
    preprocessed_frames = []
    print("----- Pre processing started -----")
    for frame in frames:
        preprocessed_frame = preprocess_frame(frame)
        preprocessed_frames.append(preprocessed_frame)
    print("----- Pre processing ended -----")
    
    if len(preprocessed_frames) < sequence_length:
        raise ValueError(f"Video is too short to create sequences of length {sequence_length}")
    
    sequences = create_sequences(preprocessed_frames, sequence_length)
    predictions = model.predict(sequences)
    return predictions, frames

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'video' not in request.files:
            return redirect(request.url)
        file = request.files['video']
        if file.filename == '':
            return redirect(request.url)
        
        if file:
            video_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(video_path)
            
            try:
                predictions, frames = predict_video(video_path, sequence_length)
                
                # Convert frames to base64 for display
                frames_base64 = []
                for frame in frames:
                    _, buffer = cv2.imencode('.jpg', frame)
                    frame_bytes = buffer.tobytes()
                    frame_base64 = base64.b64encode(frame_bytes).decode('utf-8')
                    frames_base64.append(frame_base64)
                
                average_prediction = np.mean(predictions)
                result = "deepfake" if average_prediction >= 0.5 else "real"
                return render_template('index.html', frames=frames_base64, result=result, average_prediction=average_prediction)
            except Exception as e:
                return str(e)
    
    return render_template('index.html')

if __name__ == '__main__':
    if not os.path.exists(app.config['UPLOAD_FOLDER']):
        os.makedirs(app.config['UPLOAD_FOLDER'])
    app.run(debug=True, port=8001)
