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

# Background subtractor for motion detection
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=False)

# Function to detect the club head in the ROI
def detect_club_head(frame, roi, mask):
    """Detect the golf club head in the given region of interest (ROI)."""
    # Motion detection using background subtraction
    fg_mask_full = bg_subtractor.apply(frame)  # Apply on full frame
    fg_mask = cv2.bitwise_and(fg_mask_full, fg_mask_full, mask=mask)  # Masked motion

    # Apply thresholding to isolate moving objects
    _, thresh = cv2.threshold(fg_mask, 25, 255, cv2.THRESH_BINARY)

    # Apply edge detection
    edges = cv2.Canny(roi, 30, 100)  # Experiment with thresholds

    # Combine motion mask and edges
    combined = cv2.bitwise_and(thresh, edges)

    # Find contours of the detected moving objects
    contours, _ = cv2.findContours(combined, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours by size and location
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        # Heuristic: Filter by size and location
        if 10 < w < 50 and 10 < h < 50:  # Adjust thresholds based on club head size
            # Draw the bounding box directly on the main frame
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            # Mark the center of the club head directly on the main frame
            center = (x + w // 2, y + h // 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)

            return center
    return None


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
            coords = np.array(list(key_points.values()))
            image_coords = (coords * [width, height]).astype(int)
            key_points = {key: tuple(image_coords[i]) for i, key in enumerate(key_points)}

            # Draw lines for hands, feet, and golf club
            cv2.line(frame, key_points["left_hand"], key_points["right_hand"], (0, 255, 0), 2)
            cv2.line(frame, key_points["left_foot"], key_points["right_foot"], (0, 0, 255), 2)
            cv2.line(frame, key_points["golf_club_start"], key_points["golf_club_end"], (255, 0, 0), 2)

            # Define a circular ROI for the left wrist
            wrist_center = key_points["left_hand"]
            inner_radius = 150  # Inner radius of the ring
            outer_radius = 400  # Outer radius of the ring

            # Create a mask for the circular ring
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.circle(mask, wrist_center, outer_radius, 255, -1)  # Outer circle
            cv2.circle(mask, wrist_center, inner_radius, 0, -1)  # Inner circle (black filled)

            # Apply the mask to extract the ROI (optional for computations)
            roi_frame = cv2.bitwise_and(frame, frame, mask=mask)

            # Draw the circular ROI on the main frame for visualization
            cv2.circle(frame, wrist_center, outer_radius, (255, 255, 0), 2)  # Outer circle in cyan
            cv2.circle(frame, wrist_center, inner_radius, (255, 255, 0), 2)  # Inner circle in cyan

            # Detect the club head in the ROI and draw on the main frame
            detect_club_head(frame, roi_frame, mask)

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
