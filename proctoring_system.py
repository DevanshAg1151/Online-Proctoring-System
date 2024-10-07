import cv2
import numpy as np
import dlib
import pygetwindow as gw
import tkinter as tk
from tkinter import ttk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from datetime import datetime
import time
import threading
import speech_recognition as sr
import torch
import pyaudio
import wave

print("Starting Proctoring System...")

# Load the YOLOv5 model
model = torch.hub.load('ultralytics/yolov5', 'yolov5s')

# Load the facial landmarks predictor
predictor_path = "shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(predictor_path)
detector = dlib.get_frontal_face_detector()

# Global variables to hold data
switch_data = []
timestamps = []
current_window = None
switch_count = 0
voice_detected = False
head_movement_detected = False
recording = True
cheat_percent = 0.0
persons_detected = 0

# Tkinter GUI setup
print("Initializing Tkinter GUI...")
root = tk.Tk()
root.title("Proctoring System")

frame = ttk.Frame(root, padding="10")
frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

switch_count_label = ttk.Label(frame, text="Switch Count: 0")
switch_count_label.grid(row=0, column=0, pady=10)

voice_status_label = ttk.Label(frame, text="Voice Detected: False")
voice_status_label.grid(row=1, column=0, pady=10)

head_status_label = ttk.Label(frame, text="Head Movement: False")
head_status_label.grid(row=2, column=0, pady=10)

distance_label = ttk.Label(frame, text="Cheat Percent: 0.00")
distance_label.grid(row=3, column=0, pady=10)

fig, ax = plt.subplots()
canvas = FigureCanvasTkAgg(fig, master=frame)
canvas.get_tk_widget().grid(row=4, column=0, pady=10)

# Function to update the GUI
def update_gui():
    try:
        switch_count_label.config(text=f"Switch Count: {switch_count}")
        voice_status_label.config(text=f"Voice Detected: {voice_detected}")
        head_status_label.config(text=f"Head Movement: {head_movement_detected}")
        distance_label.config(text=f"Cheat Percent: {cheat_percent:.2f}")
        ax.clear()
        ax.plot(timestamps, switch_data, marker='o', linestyle='-')
        ax.set_title("Window Switches Over Time")
        ax.set_xlabel("Time")
        ax.set_ylabel("Switch Count")
        canvas.draw()
    except Exception as e:
        print(f"Error in update_gui: {e}")

# Function to estimate distance from the camera
def estimate_distance(width):
    # Placeholder function: replace with actual distance estimation logic
    return 1.0 / width

# Function to monitor window switches
def monitor_window_switches():
    global current_window, switch_count
    try:
        new_window = gw.getActiveWindow()
        if new_window and new_window != current_window:
            current_window = new_window
            switch_count += 1
            switch_data.append(switch_count)
            timestamps.append(datetime.now().strftime("%H:%M:%S"))
            root.after(0, update_gui)  # Ensure GUI updates run in the main thread
    except Exception as e:
        print(f"Error in monitor_window_switches: {e}")
    root.after(1000, monitor_window_switches)

# Function to detect voice activity
def detect_voice():
    global voice_detected, cheat_percent
    recognizer = sr.Recognizer()
    microphone = sr.Microphone()
    while True:
        try:
            with microphone as source:
                recognizer.adjust_for_ambient_noise(source)
                audio = recognizer.listen(source, timeout=5)
            try:
                speech = recognizer.recognize_google(audio)
                voice_detected = True
                cheat_percent += 0.1  # Increase cheat percent for detected voice
                print(f"[{datetime.now()}] Speech detected: {speech}")
            except sr.UnknownValueError:
                voice_detected = False
            root.after(0, update_gui)  # Ensure GUI updates run in the main thread
            time.sleep(1)
        except Exception as e:
            print(f"Error in detect_voice: {e}")

# Function to detect head movement
def detect_head_movement():
    global head_movement_detected, cheat_percent
    cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    while True:
        try:
            ret, frame = cap.read()
            if not ret:
                continue
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)
            
            if faces:
                for face in faces:
                    landmarks = predictor(gray, face)
                    nose_point = (landmarks.part(30).x, landmarks.part(30).y)
                    chin_point = (landmarks.part(8).x, landmarks.part(8).y)
                    
                    # Calculate vector between nose and chin
                    dx = chin_point[0] - nose_point[0]
                    dy = chin_point[1] - nose_point[1]
                    
                    # Calculate tilt angle in radians
                    tilt_radians = np.arctan2(dy, dx)
                    
                    # Convert radians to degrees
                    tilt_degrees = np.degrees(tilt_radians)
                    
                    # Adjust to be within 0 to 360 degrees range
                    tilt_degrees = (tilt_degrees + 360) % 360
                    
                    # Calculate tilt from vertical (assuming vertical is y axis)
                    tilt_from_vertical = tilt_degrees - 90
                    tilt_from_vertical = (tilt_from_vertical + 360) % 360
                    
                    # Normalize tilt to be within -180 to 180 degrees
                    if tilt_from_vertical > 180:
                        tilt_from_vertical -= 360
                    
                    # Determine head movement detection (adjust threshold as needed)
                    if abs(tilt_from_vertical) > 40:
                        head_movement_detected = True
                        cheat_percent += abs(tilt_from_vertical) / 100  # Adjust cheat percent based on tilt angle
                    else:
                        head_movement_detected = False
                    
                    if head_movement_detected:
                        print(f"Head tilt angle: {tilt_from_vertical:.2f} degrees")
                    
                    break  # We only need to process the first detected face
            
            else:
                head_movement_detected = False
            
            root.after(0, update_gui)  # Ensure GUI updates run in the main thread
            time.sleep(1)
        
        except Exception as e:
            print(f"Error in detect_head_movement: {e}")

# Function to detect objects and estimate distance
def detect_objects_and_distance(frame):
    global cheat_percent, persons_detected
    results = model(frame)
    frame_with_boxes = results.render()[0]
    
    # Check for specific objects (like cellphone, book, laptop, keyboard, earphone)
    detected_classes = results.pandas().xyxy[0]['name'].values
    for detected_class in detected_classes:
        if detected_class in ['cell phone', 'book', 'laptop', 'keyboard', 'earphone']:
            cheat_percent += 0.1
            print(f"{detected_class} detected, increasing cheat percent")
    
    # Count persons detected
    persons_detected = 0
    if 'person' in detected_classes:
        persons_detected = np.count_nonzero(detected_classes == 'person')
    
    # Check for multiple persons
    if persons_detected > 1:
        cheat_percent += 0.1
        print("Multiple persons detected, increasing cheat percent")
    
    return frame_with_boxes


# Function to record video with audio
def record_video():
    global recording
    cap = cv2.VideoCapture(0)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    video_filename = f"output_{timestamp}.avi"
    audio_filename = f"output_audio_{timestamp}.wav"
    
    out_video = cv2.VideoWriter(video_filename, fourcc, 10.0, (640, 480))  # Decreased FPS to 10
    # Audio recording setup
    audio_format = pyaudio.paInt16
    channels = 1
    sample_rate = 44100
    chunk = 1024
    audio_writer = wave.open(audio_filename, 'wb')
    audio_writer.setnchannels(channels)
    audio_writer.setsampwidth(pyaudio.PyAudio().get_sample_size(audio_format))
    audio_writer.setframerate(sample_rate)
    
    audio_stream = pyaudio.PyAudio().open(format=audio_format,
                                          channels=channels,
                                          rate=sample_rate,
                                          input=True,
                                          frames_per_buffer=chunk)
    
    while recording:
        try:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = detect_objects_and_distance(frame)
            out_video.write(frame)
            cv2.imshow('Recording', frame)
            
            # Record audio
            audio_data = audio_stream.read(chunk)
            audio_writer.writeframes(audio_data)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                recording = False
        except Exception as e:
            print(f"Error in record_video: {e}")
    
    # Release resources
    cap.release()
    out_video.release()
    cv2.destroyAllWindows()
    
    # Close audio files
    audio_stream.stop_stream()
    audio_stream.close()
    audio_writer.close()

# Start monitoring and GUI loop
print("Starting monitoring...")
root.after(1000, monitor_window_switches)

# Start threads for voice detection, head movement detection, and video recording
print("Starting threads...")
threading.Thread(target=detect_voice, daemon=True).start()
threading.Thread(target=detect_head_movement, daemon=True).start()
threading.Thread(target=record_video, daemon=True).start()

print("Entering Tkinter main loop...")
root.mainloop()

# Release the webcam and close the window
cap.release()
cv2.destroyAllWindows()
