import cv2
import numpy as np
from tensorflow.keras.models import load_model  # type: ignore
from tensorflow.keras.preprocessing.image import img_to_array  # type: ignore

# Load the face detection model (Haar Cascade)
detection_model_path = 'D:/emotion ai/models/haarcascade_frontalface_default.xml'
face_detection = cv2.CascadeClassifier(detection_model_path)

# Load the emotion classification model
emotion_model_path = 'D:/emotion ai/models/fer2013_mini_XCEPTION.110-0.65.hdf5'
emotion_classifier = load_model(emotion_model_path, compile=False)

# Emotion labels and feedback
EMOTIONS = ["angry", "disgust", "scared", "happy", "sad", "surprised", "neutral"]
FEEDBACK = {                          
    "angry": "Take a deep breath. It is okay to feel angry sometimes.",
    "disgust": "Something seems off. Want to talk about it?",
    "scared": "You look a bit scared. Everything okay?",
    "happy": "Great to see you happy! Keep smiling!",
    "sad": "Feeling down? It's okay to talk to someone.",
    "surprised": "You seem surprised! Hope it's good news!",
    "neutral": "You look calm. Stay cool!"
}

# Try available camera indices using DirectShow
camera = None
for i in range(4):
    temp_cam = cv2.VideoCapture(i, cv2.CAP_DSHOW)
    if temp_cam.isOpened():
        print(f"[INFO] Using camera index: {i}")
        camera = temp_cam
        break
    temp_cam.release()

if camera is None:
    print("[ERROR] No available webcam found.")
    exit()

cv2.namedWindow("Emotion Analysis")

while True:
    ret, frame = camera.read()
    if not ret or frame is None:
        print("[ERROR] Frame not received. Exiting...")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_detection.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE
    )

    for (x, y, w, h) in faces:
        try:
            roi = frame[y:y + h, x:x + w]
            roi = cv2.resize(roi, (64, 64))
            roi = roi.astype("float") / 255.0
            roi = img_to_array(roi)
            roi = np.expand_dims(roi, axis=0)

            preds = emotion_classifier.predict(roi, verbose=0)[0]
            emotion_probability = np.max(preds)
            label = EMOTIONS[preds.argmax()]
            confidence = f"{emotion_probability * 100:.1f}%"

            # Draw face rectangle
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)

            # Display label + confidence
            cv2.putText(frame, f"{label} ({confidence})", (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

            # Display feedback
            feedback_text = FEEDBACK[label]
            cv2.putText(frame, feedback_text, (x, y + h + 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        except Exception as e:
            print(f"[ERROR] Emotion detection failed: {e}")

    cv2.imshow("Emotion Analysis", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

camera.release()
cv2.destroyAllWindows()
