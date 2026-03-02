import cv2
import numpy as np
import threading
import tkinter as tk
from PIL import Image, ImageTk
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder

class EmotionDetectorApp:
    def __init__(self, window, window_title):
        self.window = window
        self.window.title(window_title)

        # Load face detector and CNN emotion model
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.model = load_model('best_emotion_cnn_model.h5')
        self.class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        self.encoder = LabelEncoder()
        self.encoder.classes_ = np.array(self.class_names)

        # Open webcam video capture
        self.cap = cv2.VideoCapture(0)
        if not self.cap.isOpened():
            raise RuntimeError("Error: Could not open webcam.")

        # Create Tkinter canvas for video
        self.canvas = tk.Canvas(window, width=640, height=480)
        self.canvas.pack()

        # Start video update loop in separate thread
        self.stop_event = threading.Event()
        self.thread = threading.Thread(target=self.video_loop)
        self.thread.start()

        # On window close, clean up properly
        self.window.protocol("WM_DELETE_WINDOW", self.on_closing)

    def video_loop(self):
        while not self.stop_event.is_set():
            ret, frame = self.cap.read()
            if not ret:
                continue

            # Mirror the frame horizontally
            frame = cv2.flip(frame, 1)

            # Detect emotions on frame
            detected_frame = self.detect_emotions(frame)

            # Convert to PIL image and then to ImageTk format
            img = cv2.cvtColor(detected_frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(image=img)

            # Update the Tkinter canvas image
            self.canvas.create_image(0, 0, anchor=tk.NW, image=imgtk)
            self.canvas.image = imgtk  # Keep reference to avoid garbage collection

            # Delay for about 30 fps
            if self.stop_event.wait(1/30):
                break

    def detect_emotions(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5)

        for (x, y, w, h) in faces:
            face = gray[y:y+h, x:x+w]
            face_resized = cv2.resize(face, (48, 48))
            face_norm = face_resized.astype('float32') / 255.0
            input_data = np.expand_dims(face_norm, axis=(0, -1))

            preds = self.model.predict(input_data, verbose=0)
            pred_idx = np.argmax(preds)
            emotion = self.encoder.inverse_transform([pred_idx])[0]
            confidence = preds[0][pred_idx]

            label = f"{emotion} {confidence*100:.1f}%"
            color = (235, 206, 135)

            # Draw rectangle and label
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, label, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
        return frame

    def on_closing(self):
        self.stop_event.set()
        self.thread.join()
        self.cap.release()
        self.window.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = EmotionDetectorApp(root, "Imotions - Real-Time Emotion Detector")
    root.mainloop()
