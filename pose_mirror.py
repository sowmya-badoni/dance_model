import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)  # 0 = your webcam

with mp_pose.Pose(min_detection_confidence=0.5,
                  min_tracking_confidence=0.5) as pose:

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the image to RGB for MediaPipe
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(img)

        # Draw the pose landmarks
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imshow('Dance Mirror', img)

        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
