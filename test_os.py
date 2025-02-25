import cv2
import dlib
import numpy as np
from deepface import DeepFace
import os

# Initialize dlib face detector
detector = dlib.get_frontal_face_detector()

# Load and encode reference image
REFERENCE_IMAGE_PATH = "subiksha.jpg"  # Update with your reference image path

try:
    reference_embedding = DeepFace.represent(img_path=REFERENCE_IMAGE_PATH, model_name="VGG-Face",enforce_detection=True)[0]["embedding"]
    print("Reference face loaded successfully!")
except Exception as e:
    print(f"Error loading reference image: {e}")
    exit()

TEMP_DIR = "temp_faces"
os.makedirs(TEMP_DIR, exist_ok=True)
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

        if x < 0 or y < 0 or w <= x or h <= y:
            continue

        # Extract and process face for verification
        face_img = frame[y:h, x:w]  # Crop detected face
        if face_img.shape[0]== 0 or face_img.shape[1]== 0 :
            continue

        face_img = cv2.resize(face_img, (224, 224))  # Resize for DeepFace
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)  # Convert to RGB

        try:
            # save temp face image 
            temp_path = os.path.join(TEMP_DIR, "temp_face.jpg")
            cv2.imwrite(temp_path, face_rgb)

            # Get embedding of the detected face
            detected_embedding = DeepFace.represent(img_path=temp_path, model_name="VGG-Face", enforce_detection=False)[0]["embedding"]


            # Compare embeddings
            distance = np.linalg.norm(np.array(reference_embedding) - np.array(detected_embedding))
            threshold = 0.9  # Set threshold (tune as needed)
          
            print(f"Distance: {distance}") 

            label = "Matched" if distance < threshold else "Not Matched"
            color = (0, 255, 0) if label == "Matched" else (0, 0, 255)
        except Exception as e:
            label = "Error"
            color = (0, 0, 255)
            print(f"Error processing face:{e}")

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

