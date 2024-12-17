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
 
# Counters
total_frames = 0
club_detected_frames = 0
 
# Open video file
#cap = cv2.VideoCapture("C:/Users/kristinnh/Desktop/slomo7.mov")
cap = cv2.VideoCapture(1)
# Background subtractor for motion detection
bg_subtractor = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=25, detectShadows=False)
 
# Set maximum display dimensions
max_width = 800  # Adjust based on your screen size
max_height = 600  # Adjust based on your screen size
 
# Define Output Video Writer
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
 
# Adjust output dimensions after resizing
output_width = min(max_width, frame_width)
output_height = min(max_height, frame_height)
 
output_path = "C:/Users/kristinnh/Desktop/video_output.mp4"
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter(output_path, fourcc, 30, (output_width, output_height))
 
# Function to resize frames while maintaining aspect ratio
def resize_frame(frame, width, height):
    return cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

# Function to detect the club head in the ROI
def detect_club_head(frame, roi, mask):
    """Detect the golf club head in the given region of interest (ROI)."""
    global club_detected_frames  # Access the global counter for club detection
 
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
            # Increment club detection counter
            club_detected_frames += 1
 
            # Draw the bounding box directly on the main frame
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
 
            # Mark the center of the club head directly on the main frame
            center = (x + w // 2, y + h // 2)
            cv2.circle(frame, center, 5, (0, 0, 255), -1)
 
            return center
    return None
 
while True:
    # Read frame from video
    ret, frame = cap.read()
 
    if not ret:
        print("End of video or error reading the video.")
        break
 
    # Increment total frames counter
    total_frames += 1
 
    # Resize frame to fit within the output dimensions
    frame = resize_frame(frame, output_width, output_height)
 
    # Convert frame to RGB (MediaPipe expects RGB images)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
 
    # Process frame with MediaPipe Pose
    result = pose.process(rgb_frame)

    # Function to extract landmarks and draw lines for hands, feet, and golf club
    def draw_keypoints_and_lines(frame, result):
        if result.pose_landmarks:
            mp_drawing.draw_landmarks(frame, result.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            landmarks = result.pose_landmarks.landmark
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

            height, width, _ = frame.shape
            coords = np.array(list(key_points.values()))
            image_coords = (coords * [width, height]).astype(int)
            key_points = {key: tuple(image_coords[i]) for i, key in enumerate(key_points)}
 
            cv2.line(frame, key_points["left_hand"], key_points["right_hand"], (0, 255, 0), 2)
            cv2.line(frame, key_points["left_foot"], key_points["right_foot"], (0, 0, 255), 2)
            cv2.line(frame, key_points["golf_club_start"], key_points["golf_club_end"], (255, 0, 0), 2)
 
            wrist_center = key_points["left_hand"]
            inner_radius = 75   # Inner radius of ROI
            outer_radius = 300  # Outer radius of ROI
 
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.circle(mask, wrist_center, outer_radius, 255, -1)
            cv2.circle(mask, wrist_center, inner_radius, 0, -1)
 
            roi_frame = cv2.bitwise_and(frame, frame, mask=mask)
            detect_club_head(frame, roi_frame, mask)
 
    draw_keypoints_and_lines(frame, result)
 
    count += 1
    current_time = time.time()

    if current_time - start_time >= 1:
        FPS_count = count / (current_time - start_time)
        count = 0
        start_time = current_time

    cv2.putText(frame, f"FPS: {FPS_count:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
 
    out.write(frame)
    cv2.imshow("Processed Video", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
out.release()
cv2.destroyAllWindows()
 
# Print total frames and frames with a club detected
print(f"Total Frames: {total_frames}")
print(f"Frames with Club Detected: {club_detected_frames}")
