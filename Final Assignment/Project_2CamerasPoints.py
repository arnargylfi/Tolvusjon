import cv2
import mediapipe as mp
import numpy as np
import time

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Timing variables for FPS
start_time = time.time()
count = 0
FPS_count = 0

# Open both cameras
cap0 = cv2.VideoCapture(0)  # Laptop camera
cap1 = cv2.VideoCapture(1)  # Phone camera (change index as needed)

# Resize scale factor
resize_factor = 1  # Resize as needed (1 equals 100%)

while True:
    # Read frames from both cameras
    ret0, frame0 = cap0.read()
    ret1, frame1 = cap1.read()

    if not ret0 or not ret1:
        print("Error accessing one of the cameras.")
        break

    # Resize frames
    frame0 = cv2.resize(frame0, None, fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_AREA)
    frame1 = cv2.resize(frame1, None, fx=resize_factor, fy=resize_factor, interpolation=cv2.INTER_AREA)

    # Convert frames to RGB (MediaPipe expects RGB images)
    rgb_frame0 = cv2.cvtColor(frame0, cv2.COLOR_BGR2RGB)
    rgb_frame1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2RGB)

    # Process frames with MediaPipe Pose
    result0 = pose.process(rgb_frame0)
    result1 = pose.process(rgb_frame1)

    # Function to extract landmarks and draw lines for hands, feet, and golf club
    def draw_keypoints_and_lines(frame, result):
        if result.pose_landmarks:
            # Draw pose landmarks
            mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Extract landmarks
            landmarks = result.pose_landmarks.landmark

            # Define key points for hands, feet, and a hypothetical golf club
            key_points = {
                "left_hand": (landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y),
                "right_hand": (landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                               landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y),
                "left_foot": (landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x,
                              landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y),
                "right_foot": (landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x,
                               landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y),
                "golf_club_start": (landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                                    landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y),
                "golf_club_end": (landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x,
                                  landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y)
            }

            # Convert normalized coordinates to image coordinates
            height, width, _ = frame.shape
            for key, (x, y) in key_points.items():
                key_points[key] = (int(x * width), int(y * height))

            # Draw lines for hands, feet, and golf club
            cv2.line(frame, key_points["left_hand"], key_points["right_hand"], (0, 255, 0), 2)
            cv2.line(frame, key_points["left_foot"], key_points["right_foot"], (0, 0, 255), 2)
            cv2.line(frame, key_points["golf_club_start"], key_points["golf_club_end"], (255, 0, 0), 2)

    # Draw keypoints and lines on both frames
    draw_keypoints_and_lines(frame0, result0)
    draw_keypoints_and_lines(frame1, result1)

    # Calculate and display FPS
    count += 1
    current_time = time.time()
    if current_time - start_time >= 1:  # Update FPS every second
        FPS_count = count / (current_time - start_time)
        count = 0
        start_time = current_time

    # Add FPS to both frames
    cv2.putText(frame0, f"FPS: {FPS_count:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    cv2.putText(frame1, f"FPS: {FPS_count:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

    # Display results
    cv2.imshow("Laptop Camera View", frame0)
    cv2.imshow("Phone Camera View", frame1)

    # Exit loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release cameras and close windows
cap0.release()
cap1.release()
cv2.destroyAllWindows()
