import cv2
import face_recognition
import os
import numpy as np

# Path to reference images
REFERENCE_DIR = "reference_faces"

# Load and encode multiple reference images
known_face_encodings = []
known_face_names = []

if not os.path.exists(REFERENCE_DIR):
    print(f"Error: Reference directory '{REFERENCE_DIR}' not found!")
    exit()

for filename in os.listdir(REFERENCE_DIR):
    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        person_name = os.path.splitext(filename)[0]  # Extract name
        img_path = os.path.join(REFERENCE_DIR, filename)

        try:
            image = face_recognition.load_image_file(img_path)
            encoding = face_recognition.face_encodings(image)
            if encoding:
                known_face_encodings.append(encoding[0])
                known_face_names.append(person_name)
                print(f"Loaded encoding for: {person_name}")
            else:
                print(f"Warning: No face found in {filename}")
        except Exception as e:
            print(f"Error processing {filename}: {e}")

if not known_face_encodings:
    print("Error: No valid reference images found!")
    exit()

# Start webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert frame from BGR to RGB for face_recognition
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect face locations and encodings
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # Compare detected face with known faces
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.6)
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)

        best_match_index = np.argmin(face_distances) if len(face_distances) > 0 else -1
        name = "Unknown"

        if best_match_index != -1 and matches[best_match_index]:
            name = known_face_names[best_match_index]

        # Draw rectangle and label
        color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    # Show the frame
    cv2.imshow("Facial Recognition", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
