from flask import Flask, render_template, request,Response
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from threading import Thread
import random
from datetime import datetime
import time
import smtplib

app = Flask(__name__)
HOST = "smtp-mail.outlook.com"
PORT = 587
FROM_EMAIL = "chedhly.ghorbel@esprit.tn"
TO_EMAIL = "medali.farhat@esprit.tn"
PASSWORD = "211JMT9635C "

MESSAGE = """Subject: Violence Detection
Hi there,

it seems there was some violence detected in the video you reviewed.

Please take action!

DO NOT REPLY TO THIS MAIL!"""
# Load the saved model
loaded_model = load_model("model.h5")

output_folder = "vid_dump"
os.makedirs(output_folder, exist_ok=True)

# Flag to indicate if the video capture thread is running
capture_thread_running = False

# Open the camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open camera.")

# Function to predict the class of a video
def predict_single_video(video_file_path, sequence_length, image_height, image_width, min_file_size_mb=1):
    # Check file size
    file_size_mb = os.path.getsize(video_file_path) / (1024 * 1024)  # Convert to megabytes
    if file_size_mb < 1:
        print(f"Skipping video {video_file_path} - File size is less than {min_file_size_mb} MB.")
        return "Unknown", 0.0

    preprocessed_data = process(video_file_path, CLASSES_LIST, sequence_length, image_height, image_width)[0]

    # Check if preprocessed_data is empty
    if not preprocessed_data.any():
        print(f"Error: Preprocessed data is empty for video {video_file_path}")
        return "Unknown", 0.0

    prediction = loaded_model.predict(preprocessed_data)

    # Determine the predicted class based on the highest probability
    predicted_class_index = np.argmax(prediction)
    predicted_class_name = CLASSES_LIST[predicted_class_index]
    print(predicted_class_name)
    return predicted_class_name, prediction[0][predicted_class_index]


# Load the saved preprocessing functions
def frames_extraction(video_path, sequence_length, image_height, image_width):
    frames_list = []
    
    video_reader = cv2.VideoCapture(video_path)
    video_frames_count = int(video_reader.get(cv2.CAP_PROP_FRAME_COUNT))
    skip_frames_window = max(int(video_frames_count / sequence_length), 1)
 
    for frame_counter in range(sequence_length):
        video_reader.set(cv2.CAP_PROP_POS_FRAMES, frame_counter * skip_frames_window)
        success, frame = video_reader.read() 
 
        if not success:
            break
 
        resized_frame = cv2.resize(frame, (image_height, image_width))
        normalized_frame = resized_frame / 255
        frames_list.append(normalized_frame)
    
    video_reader.release()
    return frames_list

def process(video_file_path, classes_list, sequence_length, image_height, image_width):
    features = []
    labels = []
    video_files_paths = []
    
    # Extract the frames of the video file.
    frames = frames_extraction(video_file_path, sequence_length, image_height, image_width)
 
    # Check if the extracted frames are equal to the SEQUENCE_LENGTH specified.
    # So ignore the videos having frames less than the SEQUENCE_LENGTH.
    if len(frames) == sequence_length:
        # Iterate through all the classes.
        for class_index, class_name in enumerate(classes_list):
            # Append the data to their respective lists.
            features.append(frames)
            labels.append(class_index)
            video_files_paths.append(video_file_path)
 
    features = np.asarray(features)
    labels = np.array(labels)  

    return features, labels, video_files_paths

# Specify necessary parameters
SEQUENCE_LENGTH = 16
IMAGE_HEIGHT, IMAGE_WIDTH = 64, 64
CLASSES_LIST = ["NonViolence", "Violence"]

# Function to check for three consecutive "Violence" videos and raise an alert
def check_consecutive_violence_alert(video_files):
    consecutive_count = 0

    for video_file in video_files:
        predicted_class, _ = predict_single_video(video_file, SEQUENCE_LENGTH, IMAGE_HEIGHT, IMAGE_WIDTH)

        if predicted_class == "Violence":
            consecutive_count += 1
            if consecutive_count == 3:
                print("Call the police! Consecutive violence detected.")
                smtp = smtplib.SMTP(HOST, PORT)

                status_code, response = smtp.ehlo()
                status_code, response = smtp.starttls()
                status_code, response = smtp.login(FROM_EMAIL, PASSWORD)
        
                smtp.sendmail(FROM_EMAIL, TO_EMAIL, MESSAGE)
        
                break
        else:
            consecutive_count = 0

# Function to capture and save video
def capture_and_save_video(output_folder):
    global capture_thread_running

    while capture_thread_running:
        current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
        random_name = f"{current_time}.mp4"

        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        video_name = os.path.join(output_folder, random_name)
        out = cv2.VideoWriter(video_name, fourcc, 20.0, (640, 480))

        start_time = time.time()
        frames_captured = 0

        while time.time() - start_time < 7:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture frame.")
                break
            out.write(frame)
            frames_captured += 1

        out.release()
        print(f"Video saved as: {video_name}")

        if frames_captured >= SEQUENCE_LENGTH:
            # Check for three consecutive "Violence" videos
            video_files = [os.path.join(output_folder, file) for file in os.listdir(output_folder) if file.endswith(".mp4")]
            check_consecutive_violence_alert(video_files)

        # Sleep for 5 seconds to ensure only one video is saved per 5 seconds
        time.sleep(2)
  
    cap.release()

# Start the video capture thread only if it's not already running
if not capture_thread_running:
    capture_thread_running = True
    video_thread = Thread(target=capture_and_save_video, args=(output_folder,))
    video_thread.start()

def generate():
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture frame.")
            break

        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            print("Error: Failed to encode frame.")
            break

        frame_bytes = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
