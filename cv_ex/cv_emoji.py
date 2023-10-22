import cv2
import numpy as np
import random
from deepface import DeepFace


face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# Emojis
emojis = {
    "happy": "\N{slightly smiling face}",
    "neutral": "\N{neutral face}",
    "sad": "\N{slightly frowning face}",
    "disgust": "\N{face with medical mask}",
    "fear": "\N{fearful face}",
    "angry": "\N{angry face}",
    "surprise": "\N{astonished face}"
}
def print_emoji(emotion):
    if emotion in emojis:
        current_emoji = emojis[emotion]
        print(current_emoji)
    else:
        print("Emoji not found for emotion:", emotion)

video = cv2.VideoCapture(0)

if not video.isOpened():
    print("Error: Could not open video.")
    exit()

while True:
    ret, frame = video.read()

    if not ret:
        print("Error: Can't receive frame (stream end?). Exiting ...")
        break
    
    # grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
    # Draw rectangles around detected faces
    for (x, y, w, h) in faces:
        face = frame[y:y+h, x:x+w]
        
        # Use DeepFace to analyze the face for emotions
        results = DeepFace.analyze(face, actions=["emotion"], enforce_detection=False)
        
        # Extract dominant emotion and display it above the face in the frame
        dominant_emotion = results[0]['dominant_emotion']
        cv2.putText(frame, dominant_emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        print_emoji(dominant_emotion)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
    cv2.imshow("video frame", frame)
     
    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) == ord('q'):
        break
    
video.release()
cv2.destroyAllWindows()



 