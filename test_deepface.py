import cv2
import dlib
import numpy as np
from deepface import DeepFace


# Initialize dlib face detector
detector = dlib.get_frontal_face_detector()

# Load and encode reference image
REFERENCE_IMAGE_PATH = "reference.jpg"  # Update with your reference image path

try:
    reference_embedding = DeepFace.represent(img_path=REFERENCE_IMAGE_PATH, model_name="VGG-Face")[0]["embedding"]
    print("Reference face loaded successfully!")
except Exception as e:
    print(f"Error loading reference image: {e}")
    exit()

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)

    for face in faces:
        x, y, w, h = face.left(), face.top(), face.right(), face.bottom()

        # Extract and process face for verification
        face_img = frame[y:h, x:w]  # Crop detected face
        face_img = cv2.resize(face_img, (224, 224))  # Resize for DeepFace
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)  # Convert to RGB

        try:
            # Get embedding of the detected face
            face_embedding = DeepFace.represent(img_path=face_rgb, model_name="VGG-Face")[0]["embedding"]

            # Compare embeddings
            distance = np.linalg.norm(np.array(reference_embedding) - np.array(face_embedding))
            threshold = 10  # Set threshold (tune as needed)

            label = "Matched" if distance < threshold else "Not Matched"
            color = (0, 255, 0) if label == "Matched" else (0, 0, 255)
        except:
            label = "Error"
            color = (0, 0, 255)

        # Draw rectangle and label
        cv2.rectangle(frame, (x, y), (w, h), color, 2)
        cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Display the frame
    cv2.imshow("Face Detection & Verification", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
