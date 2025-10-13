import cv2
import mediapipe as mp
import numpy as np

# --- Standard MediaPipe Setup ---
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
cap = cv2.VideoCapture(0) # Use webcam

# --- Function to draw a limb as a slightly wider rectangle ---
def draw_limb(image, start_point, end_point, color, thickness=15):
    # Calculate the angle of the limb
    angle = np.arctan2(end_point[1] - start_point[1], end_point[0] - start_point[0])
    # Calculate offset for drawing parallel lines
    dx = int(thickness / 2 * np.sin(angle))
    dy = int(thickness / 2 * np.cos(angle))

    # Define the four points of the wider rectangle/polygon
    p1 = (start_point[0] - dx, start_point[1] + dy)
    p2 = (start_point[0] + dx, start_point[1] - dy)
    p3 = (end_point[0] + dx, end_point[1] - dy)
    p4 = (end_point[0] - dx, end_point[1] + dy)

    points = np.array([p1, p2, p3, p4], np.int32)
    cv2.fillConvexPoly(image, points, color)


# --- Main Loop ---
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # --- Pose Detection ---
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)

    # --- Window 1: Your Live Camera View ---
    cam_view = frame.copy()
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(cam_view, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    cv2.imshow('Camera View', cam_view)

    # --- Window 2: The Human-like Avatar View ---
    h, w, _ = frame.shape
    avatar_view = np.ones((h, w, 3), dtype=np.uint8) * 255 # White background

    if results.pose_landmarks:
        landmarks = results.pose_landmarks.landmark

        # Get coordinates for key joints
        joint_coords = {}
        for lm_enum in mp_pose.PoseLandmark:
            lm = landmarks[lm_enum.value]
            joint_coords[lm_enum.name] = (int(lm.x * w), int(lm.y * h))

        # --- Define Colors ---
        skin_color = (203, 222, 238) # Light skin tone (BGR)
        shirt_color = (150, 75, 0) # Brownish color for shirt
        pants_color = (100, 100, 0) # Darker greenish color for pants
        shoe_color = (50, 50, 50) # Dark grey for shoes
        hair_color = (50, 50, 150) # Dark blueish for hair (can be changed)
        eye_color = (0, 0, 0) # Black
        mouth_color = (0, 0, 200) # Red

        # --- Draw the Avatar ---

        # Torso (a wider rectangle/polygon)
        # Using shoulders and hips to define a torso shape
        shoulder_mid = ((joint_coords['LEFT_SHOULDER'][0] + joint_coords['RIGHT_SHOULDER'][0]) // 2,
                        (joint_coords['LEFT_SHOULDER'][1] + joint_coords['RIGHT_SHOULDER'][1]) // 2)
        hip_mid = ((joint_coords['LEFT_HIP'][0] + joint_coords['RIGHT_HIP'][0]) // 2,
                   (joint_coords['LEFT_HIP'][1] + joint_coords['RIGHT_HIP'][1]) // 2)

        # Draw a wider torso segment
        draw_limb(avatar_view, shoulder_mid, hip_mid, shirt_color, thickness=60)


        # Arms (using the helper function for wider limbs)
        draw_limb(avatar_view, joint_coords['LEFT_SHOULDER'], joint_coords['LEFT_ELBOW'], skin_color, thickness=25)
        draw_limb(avatar_view, joint_coords['LEFT_ELBOW'], joint_coords['LEFT_WRIST'], skin_color, thickness=20)
        draw_limb(avatar_view, joint_coords['RIGHT_SHOULDER'], joint_coords['RIGHT_ELBOW'], skin_color, thickness=25)
        draw_limb(avatar_view, joint_coords['RIGHT_ELBOW'], joint_coords['RIGHT_WRIST'], skin_color, thickness=20)

        # Legs (using the helper function for wider limbs)
        draw_limb(avatar_view, joint_coords['LEFT_HIP'], joint_coords['LEFT_KNEE'], pants_color, thickness=30)
        draw_limb(avatar_view, joint_coords['LEFT_KNEE'], joint_coords['LEFT_ANKLE'], pants_color, thickness=25)
        draw_limb(avatar_view, joint_coords['RIGHT_HIP'], joint_coords['RIGHT_KNEE'], pants_color, thickness=30)
        draw_limb(avatar_view, joint_coords['RIGHT_KNEE'], joint_coords['RIGHT_ANKLE'], pants_color, thickness=25)

        # Hands and Feet (small circles)
        cv2.circle(avatar_view, joint_coords['LEFT_WRIST'], 15, skin_color, -1)
        cv2.circle(avatar_view, joint_coords['RIGHT_WRIST'], 15, skin_color, -1)
        cv2.circle(avatar_view, joint_coords['LEFT_ANKLE'], 20, shoe_color, -1) # Shoes
        cv2.circle(avatar_view, joint_coords['RIGHT_ANKLE'], 20, shoe_color, -1) # Shoes


        # Head (a larger circle)
        head_radius = 50
        cv2.circle(avatar_view, joint_coords['NOSE'], head_radius, skin_color, -1)

        # Hair (simple half-circle or oval on top of the head)
        cv2.ellipse(avatar_view, joint_coords['NOSE'], (head_radius, head_radius - 10), 0, 180, 360, hair_color, -1)


        # Basic Facial Features
        # Eyes (relative to nose)
        eye_offset_x = 15
        eye_offset_y = 10
        cv2.circle(avatar_view, (joint_coords['NOSE'][0] - eye_offset_x, joint_coords['NOSE'][1] - eye_offset_y), 5, eye_color, -1)
        cv2.circle(avatar_view, (joint_coords['NOSE'][0] + eye_offset_x, joint_coords['NOSE'][1] - eye_offset_y), 5, eye_color, -1)

        # Mouth (simple curve relative to nose)
        mouth_offset_y = 25
        mouth_start = (joint_coords['NOSE'][0] - 15, joint_coords['NOSE'][1] + mouth_offset_y)
        mouth_end = (joint_coords['NOSE'][0] + 15, joint_coords['NOSE'][1] + mouth_offset_y)
        mouth_curve_point = (joint_coords['NOSE'][0], joint_coords['NOSE'][1] + mouth_offset_y + 5) # For a slight curve
        cv2.ellipse(avatar_view, (joint_coords['NOSE'][0], joint_coords['NOSE'][1] + mouth_offset_y), (15, 7), 0, 0, 180, mouth_color, -1)


    cv2.imshow('Human-like Avatar', avatar_view)

    # --- Quit Condition ---
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# --- Cleanup ---
cap.release()
pose.close() # Important to close MediaPipe pose object
cv2.destroyAllWindows()