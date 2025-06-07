import cv2
import os
import librosa
import joblib
import tkinter as tk
from tkinter import filedialog, messagebox
from tkinter import ttk
from PIL import Image, ImageTk
from deepface import DeepFace
from collections import Counter
import threading

# Emotion Detection Functions

def detect_emotion_photo(photo_path):
    try:
        result = DeepFace.analyze(img_path=photo_path, actions=['emotion'])
        emotion = result[0]['dominant_emotion']
        messagebox.showinfo("Photo Emotion", f"Detected Emotion: {emotion}")
    except Exception as e:
        messagebox.showerror("Error", str(e))

def detect_emotion_video(video_path):
    cap = cv2.VideoCapture(video_path)
    emotions = []
    frame_count = 0

    while frame_count < 100:
        ret, frame = cap.read()
        if not ret:
            break
        try:
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            emotions.append(result[0]['dominant_emotion'])
        except:
            pass
        frame_count += 1

    cap.release()

    if emotions:
        emotion_summary = Counter(emotions).most_common(1)
        messagebox.showinfo("Video Emotion", f"Predicted Emotion: {emotion_summary[0][0]}")
    else:
        messagebox.showwarning("Warning", "No faces detected in video.")

def extract_audio_features(audio_path):
    y, sr = librosa.load(audio_path)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    return mfcc.mean(axis=1)

def detect_emotion_audio(audio_path, model_path="audio_emotion_model.pkl"):
    if not os.path.exists(model_path):
        messagebox.showerror("Error", "Audio model not found! Train and save it as 'audio_emotion_model.pkl'")
        return
    try:
        model = joblib.load(model_path)
        features = extract_audio_features(audio_path)
        prediction = model.predict([features])
        messagebox.showinfo("Audio Emotion", f"Detected Emotion: {prediction[0]}")
    except Exception as e:
        messagebox.showerror("Error", str(e))

def detect_emotion_webcam():
    cap = cv2.VideoCapture(0)
    cv2.namedWindow("Webcam Emotion Detection")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        try:
            result = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            emotion = result[0]['dominant_emotion']
            cv2.putText(frame, f"Emotion: {emotion}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        except:
            cv2.putText(frame, "Face not detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.imshow("Webcam Emotion Detection", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# GUI Implementation

def run_photo():
    file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png")])
    if file_path:
        threading.Thread(target=detect_emotion_photo, args=(file_path,)).start()

def run_video():
    file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mov")])
    if file_path:
        threading.Thread(target=detect_emotion_video, args=(file_path,)).start()

def run_audio():
    file_path = filedialog.askopenfilename(filetypes=[("Audio files", "*.wav *.mp3")])
    if file_path:
        threading.Thread(target=detect_emotion_audio, args=(file_path,)).start()

def run_webcam():
    threading.Thread(target=detect_emotion_webcam).start()

# Main GUI window
root = tk.Tk()
root.title("Emotion Detection System")
root.geometry("400x400")
root.configure(bg="#f0f0f0")

title_label = ttk.Label(root, text="Emotion Detection", font=("Helvetica", 20, "bold"))
title_label.pack(pady=20)

ttk.Button(root, text="Detect from Photo", width=30, command=run_photo).pack(pady=10)
ttk.Button(root, text="Detect from Video", width=30, command=run_video).pack(pady=10)
ttk.Button(root, text="Detect from Audio", width=30, command=run_audio).pack(pady=10)
ttk.Button(root, text="Live Detection (Webcam)", width=30, command=run_webcam).pack(pady=10)

footer = ttk.Label(root, text="Press 'q' to stop webcam", font=("Arial", 10), background="#f0f0f0", foreground="#555")
footer.pack(pady=20)

root.mainloop()
