import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose

# Define connections between keypoints for stick figure
connections = mp_pose.POSE_CONNECTIONS

cap = cv2.VideoCapture(0)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Process pose (we don't display this frame)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        # Create a blank white image for drawing
        blank = np.ones((480, 640, 3), dtype=np.uint8) * 255

        if results.pose_landmarks:
            h, w, _ = blank.shape
            landmarks = results.pose_landmarks.landmark

            # Draw stick figure on blank canvas
            for connection in connections:
                start_idx, end_idx = connection
                start = landmarks[start_idx]
                end = landmarks[end_idx]

                if start.visibility > 0.5 and end.visibility > 0.5:
                    x1, y1 = int(start.x * w), int(start.y * h)
                    x2, y2 = int(end.x * w), int(end.y * h)
                    cv2.line(blank, (x1, y1), (x2, y2), (0, 0, 0), 2)

            # Optionally, draw joints (dots)
            for lm in landmarks:
                if lm.visibility > 0.5:
                    x, y = int(lm.x * w), int(lm.y * h)
                    cv2.circle(blank, (x, y), 4, (0, 0, 255), -1)

        cv2.imshow("Stick Figure", blank)

        # Press Q to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()

