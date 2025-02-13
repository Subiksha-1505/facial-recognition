import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import cv2
import numpy as np
from deepface import DeepFace

# Load OpenCV face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Load and encode reference image
REFERENCE_IMAGE = "reference.jpg"  # Change this to your image
reference_embedding = DeepFace.represent(img_path=REFERENCE_IMAGE, model_name="VGG-Face")[0]["embedding"]

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale for detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(50, 50))

    for (x, y, w, h) in faces:
        face_img = frame[y:y+h, x:x+w]  # Extract face
        face_img = cv2.resize(face_img, (224, 224))  # Resize for DeepFace
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)  # Convert to RGB

        # Get embedding of the detected face
        try:
            face_embedding = DeepFace.represent(img_path=face_rgb, model_name="VGG-Face")[0]["embedding"]

            # Compare with reference embedding
            distance = np.linalg.norm(np.array(reference_embedding) - np.array(face_embedding))
            threshold = 10  # Adjust this threshold based on your needs

            label = "Matched" if distance < threshold else "Unknown"
            color = (0, 255, 0) if label == "Matched" else (0, 0, 255)
        except:
            label = "Error"
            color = (0, 0, 255)

        # Draw rectangle and label
        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Show the frame
    cv2.imshow("Facial Recognition", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
