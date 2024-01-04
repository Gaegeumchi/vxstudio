import cv2
import dlib

# Load the pre-trained facial landmark predictor from dlib
predictor_path = "shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

# Open a video capture object (0 for default camera)
cap = cv2.VideoCapture(0)

while True:
    # Read a frame from the video capture
    ret, frame = cap.read()

    # Convert the frame to grayscale for better face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the grayscale frame
    faces = detector(gray)

    # Loop over each detected face
    for face in faces:
        # Get facial landmarks
        shape = predictor(gray, face)

        # Draw landmarks on the frame
        for i in range(68):
            x, y = shape.part(i).x, shape.part(i).y
            cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)  # Green circles for landmarks

    # Display the result
    cv2.imshow('Face Landmark Detection', frame)

    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture object and close all windows
cap.release()
cv2.destroyAllWindows()
