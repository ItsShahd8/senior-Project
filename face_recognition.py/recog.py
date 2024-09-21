import threading
import cv2
from deepface import DeepFace
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Initialize camera
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

counter = 0
face_match = False

# Load your reference image
reference_img = cv2.imread("c:/Users/shaha/senior-Project/face_recognition.py/shahd.jpg")
if reference_img is None:
    print("Error: Could not load reference image 'shahd.jpg'. Please check the file path.")
    exit()



# Create a face detector
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def check_face(frame, reference_img):
    global face_match
    try:
        # Convert images to RGB (DeepFace expects RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        reference_rgb = cv2.cvtColor(reference_img, cv2.COLOR_BGR2RGB)
        
        # Perform face verification
        result = DeepFace.verify(frame_rgb, reference_rgb, enforce_detection=False)
        face_match = result['verified']
    except Exception as e:
        print(f"Face verification error: {str(e)}")
        face_match = False

while True:
    ret, frame = cap.read()

    if ret:
        # Detect faces in the frame
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        if counter % 30 == 0:
            threading.Thread(target=check_face, args=(frame.copy(), reference_img.copy())).start()
        counter += 1

        if face_match:
            cv2.putText(frame, "Match!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)
        else:
            cv2.putText(frame, "No Match!", (20, 450), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)

        cv2.imshow("video", frame)

    key = cv2.waitKey(1)
    if key == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
