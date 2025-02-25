import cv2
import face_recognition
import os
import numpy as np
import mediapipe as mp
import torch
from torchvision import transforms


mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=100,
    refine_landmarks=True,
    min_detection_confidence=0.8,
    min_tracking_confidence=0.8
)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", trust_repo=True).to(device).eval()


transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((384, 384)),
    transforms.Normalize(mean=[0.5], std=[0.5])
])


REFERENCE_DIR = "reference_faces"
known_face_encodings = []
known_face_names = []


for filename in os.listdir(REFERENCE_DIR):
    if filename.lower().endswith((".png", ".jpg", ".jpeg")):
        person_name = os.path.splitext(filename)[0]
        img_path = os.path.join(REFERENCE_DIR, filename)
        try:
            image = face_recognition.load_image_file(img_path)
            encoding = face_recognition.face_encodings(image, model="hog", num_jitters=5)
            if encoding:
                known_face_encodings.append(encoding[0])
                known_face_names.append(person_name)
        except Exception:
            pass

if not known_face_encodings:
    print("No known faces found. Exiting.")
    exit()


cap = cv2.VideoCapture(0)
frame_count = 0  

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    frame_count += 1

    if frame_count % 3 != 0:
        continue

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

  
    face_locations = face_recognition.face_locations(rgb_frame, model="hog")
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations, num_jitters=1)

    results = face_mesh.process(rgb_frame) if face_locations else None

    matched_any = False  

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding, tolerance=0.4)
        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances) if len(face_distances) > 0 else -1

        if best_match_index != -1 and matches[best_match_index] and face_distances[best_match_index] < 0.5:
            name = known_face_names[best_match_index]
            matched_any = True
            print("Matched:", name) 
        else:
            name = "Unmatched"

        
        color = (0, 255, 0) if name != "Unmatched" else (0, 0, 255)
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

    
    if not matched_any:
        print("Not Matched")

    
    cv2.imshow("Optimized Face Recognition", frame)


cap.release()
cv2.destroyAllWindows()
