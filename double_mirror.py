import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

connections = mp_pose.POSE_CONNECTIONS

cap = cv2.VideoCapture(0)

with mp_pose.Pose(min_detection_confidence=0.5,
                  min_tracking_confidence=0.5) as pose:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process pose detection
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        # --- 1️⃣ Show the webcam video with pose overlay ---
        cam_view = frame.copy()
        if results.pose_landmarks:
            mp_drawing.draw_landmarks(
                cam_view, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        cv2.imshow("Camera View", cam_view)

        # --- 2️⃣ Create the blank stick figure view ---
        blank = np.ones((480, 640, 3), dtype=np.uint8) * 255  # white background

        if results.pose_landmarks:
            h, w, _ = blank.shape
            landmarks = results.pose_landmarks.landmark

            # Draw connections (stick figure)
            for start_idx, end_idx in connections:
                start = landmarks[start_idx]
                end = landmarks[end_idx]
                if start.visibility > 0.5 and end.visibility > 0.5:
                    x1, y1 = int(start.x * w), int(start.y * h)
                    x2, y2 = int(end.x * w), int(end.y * h)
                    cv2.line(blank, (x1, y1), (x2, y2), (0, 0, 0), 2)

            # Draw joint points
            for lm in landmarks:
                if lm.visibility > 0.5:
                    x, y = int(lm.x * w), int(lm.y * h)
                    cv2.circle(blank, (x, y), 4, (0, 0, 255), -1)

        cv2.imshow("Stick Figure View", blank)

        # Press 'q' to quit both windows
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()