# face-recognition
import cv2
import numpy as np
import sys

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognizer.read("face_recognizer.yml")
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Label names should correspond to the labels assigned during training
label_names = {0: "Person 1", 1: "Person 2"}  # Add your correct labels here

cam = cv2.VideoCapture(0)

while True:
    ret, frame = cam.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (100, 100))
        label, confidence = recognizer.predict(face)

        if confidence < 69:  # Adjust threshold
            recognized_name = label_names.get(label, "Unknown")
            cv2.putText(frame, f"{recognized_name}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        else:
            recognized_name = "Unknown"
            cv2.putText(frame, "Unknown", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    cv2.imshow("Face Recognition", frame)

    # Send the recognized name to MATLAB via standard output
    if recognized_name != "Unknown":
        print(recognized_name)  # Send to MATLAB
        sys.stdout.flush()  # Ensure MATLAB reads the output

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
