This project uses a webcam to detect human faces and analyze their emotions in real-time using a pre-trained neural network. It displays the detected emotion along with a friendly feedback message.
Features:
Real-time face detection using Haar cascades
Emotion classification using a pre-trained Keras model
Displays emotion label, confidence score, and supportive feedback on screen
Supports 7 emotions: Angry, Disgust, Scared, Happy, Sad, Surprised, Neutral

FOLDER STRUCTURE:
emotion-ai/
├── models/
│   ├── haarcascade_frontalface_default.xml
│   └── fer2013_mini_XCEPTION.110-0.65.hdf5
├── emotion_detector.py

