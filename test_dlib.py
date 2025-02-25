import cv2
import dlib

# Initialize face detector
detector = dlib.get_frontal_face_detector()

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = detector(gray)

    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()

        # Draw rectangle around face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the output
    cv2.imshow("Face Detection with dlib", frame)

    # Exit when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
